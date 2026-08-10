"""Stage 1 entry point: cache cutouts → fine-tune Zoobot CBM → predict → fidelity.

Requires `--extra stage1` (torch, zoobot). Base env cannot run this — the
imports below will raise ImportError with the install hint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from galaxycbm.concepts import (
    HeadSpec,
    build_head_specs,
    per_concept_fidelity,
    reliability_figure,
)
from galaxycbm.data.cutout_cache import cache_cutouts
from galaxycbm.utils import load_config, seed_everything, write_run_json
from galaxycbm.utils.io import ensure_dir, write_json

STAGE = "concepts"


def _require_torch_env() -> None:
    try:
        import torch  # noqa: F401
        import lightning  # noqa: F401
        import zoobot  # noqa: F401
    except ImportError as e:
        print(
            f"[stage1] {e}\n"
            "Stage-1 needs the `stage1` extra: `uv sync --extra dev --extra stage1`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def _make_loader(df: pd.DataFrame, cutouts_root: Path, heads: list[HeadSpec],
                 *, batch_size: int, num_workers: int, shuffle: bool, size: int):
    import torch

    from galaxycbm.concepts.dataset import collate, get_dataset

    ds = get_dataset(df, cutouts_root, heads, size=size)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
        pin_memory=False,
    )


def main() -> None:
    _require_torch_env()

    import lightning as L  # noqa: N812
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from galaxycbm.concepts.model import build_module, predict_dataframe

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    concepts_cfg = load_config("concepts")
    seed_everything(int(model_cfg.seed))

    dataset_path = Path("data/processed/dataset.parquet")
    splits_path = Path("data/processed/splits.parquet")
    for p in (dataset_path, splits_path):
        if not p.exists():
            print(f"[stage1] missing {p} — run `make data.labels` first.", file=sys.stderr)
            raise SystemExit(2)

    ds = pd.read_parquet(dataset_path)
    splits_long = pd.read_parquet(splits_path)
    splits: dict[str, pd.DataFrame] = {
        name: ds.loc[splits_long.loc[splits_long["split"] == name, "row_index"].to_numpy()]
              .reset_index(drop=True)
        for name in splits_long["split"].unique()
    }

    heads = build_head_specs(concepts_cfg)

    # 1. Cache PNG cutouts for every id we need. Reads P1 downloads.
    raw_root = Path(data_cfg.download.root) / "gz_evo"
    shards = sorted(raw_root.rglob("*.parquet"))
    needed_ids = set(ds["id_str"].astype(str))
    cutouts_root = Path("data/interim/cutouts")
    print(f"[stage1] caching PNG cutouts for {len(needed_ids)} ids …")
    n_new = cache_cutouts(shards, cutouts_root, size=224, ids=needed_ids)
    print(f"[stage1]   {n_new} new PNGs written under {cutouts_root}")

    # 2. Dataloaders.
    bs = int(model_cfg.train.batch_size)
    nw = int(model_cfg.train.num_workers)
    train_loader = _make_loader(splits["train"], cutouts_root, heads,
                                 batch_size=bs, num_workers=nw, shuffle=True, size=224)
    val_loader = _make_loader(splits["val"], cutouts_root, heads,
                               batch_size=bs, num_workers=nw, shuffle=False, size=224)

    # 3. Module + trainer.
    module = build_module(model_cfg, heads)
    ckpt_dir = ensure_dir(Path(model_cfg.logging.run_dir))
    trainer = L.Trainer(
        max_epochs=int(model_cfg.train.max_epochs),
        precision=str(model_cfg.train.precision),
        gradient_clip_val=float(model_cfg.train.gradient_clip_val),
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=int(model_cfg.train.early_stopping_patience),
                           mode="min"),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename="cbm-{epoch:02d}-{val/loss:.3f}",
                             monitor="val/loss", mode="min",
                             save_top_k=int(model_cfg.logging.save_top_k),
                             save_last=True),
        ],
        default_root_dir=str(ckpt_dir),
        log_every_n_steps=10,
    )
    # Resume from last checkpoint if one exists — safe to Ctrl-C mid-training.
    resume_ckpt = ckpt_dir / "last.ckpt"
    if resume_ckpt.exists():
        print(f"[stage1] resuming from {resume_ckpt}")
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(resume_ckpt) if resume_ckpt.exists() else None,
    )

    # 4. Predict every split.
    device = "cuda" if _cuda_available() else "cpu"
    preds_frames: list[pd.DataFrame] = []
    for name, split_df in splits.items():
        loader = _make_loader(split_df, cutouts_root, heads,
                              batch_size=bs, num_workers=nw, shuffle=False, size=224)
        pred_df = predict_dataframe(module, loader, heads, device=device)
        pred_df["split"] = name
        preds_frames.append(pred_df)
    preds = pd.concat(preds_frames, ignore_index=True)

    preds_path = Path("results/concepts/preds.parquet")
    ensure_dir(preds_path.parent)
    preds.to_parquet(preds_path, index=False)

    # 5. Fidelity table + reliability figure (evaluated on val).
    val_df = splits["val"]
    val_preds = preds[preds["split"] == "val"].set_index("id_str").reindex(val_df["id_str"]).reset_index()
    fidelity = per_concept_fidelity(val_df, val_preds, heads, y_train=splits["train"])
    fidelity_path = Path("results/concepts/fidelity.csv")
    fig_path = Path("results/concepts/reliability.png")
    fidelity.to_csv(fidelity_path, index=False)
    reliability_figure(val_df, val_preds, heads, fig_path)

    # 6. Trivial-baseline gate.
    passed = int(fidelity["beats_baseline"].sum())
    total = int(len(fidelity))
    write_run_json(
        STAGE,
        seed=int(model_cfg.seed),
        config={"model": model_cfg, "concepts": concepts_cfg},
        extra={
            "n_train": int(len(splits["train"])),
            "n_val": int(len(splits["val"])),
            "n_test": int(len(splits.get("test", pd.DataFrame()))),
            "heads_passing_baseline": passed,
            "heads_total": total,
            "preds_parquet": str(preds_path),
            "fidelity_csv": str(fidelity_path),
            "reliability_png": str(fig_path),
            "best_checkpoint": trainer.checkpoint_callback.best_model_path
                if trainer.checkpoint_callback else None,
        },
    )
    print(f"[stage1] {passed}/{total} heads beat their trivial baseline")
    print(f"[stage1] fidelity: {fidelity_path}")
    print(f"[stage1] reliability: {fig_path}")
    print(f"[stage1] preds: {preds_path}")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


if __name__ == "__main__":
    main()
