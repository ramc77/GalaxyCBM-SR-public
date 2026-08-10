"""Stage driver for P1–P3. Dispatch: `stage=raw|concepts|labels`.

    uv run python scripts/build_dataset.py stage=concepts
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

from galaxycbm.data.statmorph_concepts import (  # noqa: E402  (import order matters — see below)
    apply_nan_policy,
    assert_no_silent_nans,
    compute_concepts_for_hf_shard,
)

# Silence per-row statmorph/astropy noise. Done AFTER importing statmorph
# (which imports astropy) so astropy has already installed its AstropyLogger
# subclass — creating logging.getLogger("astropy") beforehand would return a
# plain Logger and break astropy's own _init_log().
import logging  # noqa: E402

warnings.filterwarnings("ignore", module=r"statmorph.*")
warnings.filterwarnings("ignore", module=r"astropy.*")
logging.getLogger("astropy").setLevel(logging.ERROR)

from galaxycbm.utils import load_config, seed_everything, write_run_json
from galaxycbm.utils.io import atomic_write_bytes, ensure_dir


def _parse_stage(argv: list[str]) -> str:
    for a in argv[1:]:
        if a.startswith("stage="):
            return a.split("=", 1)[1]
    raise SystemExit("usage: build_dataset.py stage=raw|concepts|labels")


def _run_concepts() -> int:
    cfg = load_config("data")
    seed_everything(int(cfg.download.subsample.seed))
    raw_root = Path(cfg.download.root) / "gz_evo"
    shards = sorted(raw_root.rglob("*.parquet"))
    if not shards:
        print(f"[concepts] no parquet shards under {raw_root} — run scripts/download_gz_evo.py first.",
              file=sys.stderr)
        return 2

    survey_suffixes = tuple(cfg.get("statmorph", {}).get("survey_suffixes", ("dr5", "dr8")))
    gain = float(cfg.get("statmorph", {}).get("gain", 1.0))
    max_rows = cfg.get("statmorph", {}).get("max_rows_per_shard", None)
    max_rows = int(max_rows) if max_rows is not None else None
    on_nan = str(cfg.quality.on_nan)
    n_workers = int(cfg.get("statmorph", {}).get("n_workers", 1))

    # Two-level resume:
    #  - final per-shard cache at data/interim/concepts_by_shard/<name>.parquet
    #    (a shard that fully completes never runs again).
    #  - intra-shard checkpoint at data/interim/concepts_by_shard/<name>.wip.parquet
    #    (flushed every 100 rows, so a Ctrl-C mid-shard picks up mid-shard).
    shard_cache = ensure_dir(Path("data/interim/concepts_by_shard"))
    checkpoint_every = int(cfg.get("statmorph", {}).get("checkpoint_every", 100))
    frames: list[pd.DataFrame] = []
    for i, shard in enumerate(shards):
        cache_path = shard_cache / shard.name
        wip_path = shard_cache / (shard.stem + ".wip.parquet")
        if cache_path.exists():
            print(f"[concepts] {i+1}/{len(shards)} {shard.name}  (cached, skipping)")
            frames.append(pd.read_parquet(cache_path))
            continue
        print(f"[concepts] {i+1}/{len(shards)} {shard.name}"
              + (f"  (capped at {max_rows} rows)" if max_rows else "")
              + (f"  ({n_workers} workers)" if n_workers > 1 else ""))
        shard_df = compute_concepts_for_hf_shard(
            shard,
            survey_suffixes=survey_suffixes,
            gain=gain,
            max_rows=max_rows,
            checkpoint_path=wip_path,
            checkpoint_every=checkpoint_every,
            n_workers=n_workers,
        )
        shard_df.to_parquet(cache_path, index=False)
        wip_path.unlink(missing_ok=True)
        frames.append(shard_df)
    concepts = pd.concat(frames, ignore_index=True)
    concepts = apply_nan_policy(concepts, policy=on_nan)
    assert_no_silent_nans(concepts)

    out_path = Path("data/interim/manifest_with_concepts.parquet")
    ensure_dir(out_path.parent)
    buf = concepts.to_parquet(index=False)
    if buf is None:  # pandas returns None when a path is passed; branch guards typing.
        concepts.to_parquet(out_path, index=False)
    else:
        atomic_write_bytes(out_path, buf)

    total = len(concepts)
    flagged = int((concepts["statmorph_quality_flag"] != 0).sum())
    write_run_json(
        "data.concepts",
        seed=int(cfg.download.subsample.seed),
        config=cfg,
        extra={
            "n_shards": len(shards),
            "n_rows": total,
            "n_flagged": flagged,
            "flagged_fraction": flagged / total if total else 0.0,
            "output_manifest": str(out_path),
        },
    )
    print(f"[concepts] wrote {out_path} — {total} rows, {flagged} flagged")
    return 0


def _run_labels() -> int:
    from galaxycbm.data.labels import build_dataset, splits_to_frame

    cfg = load_config("data")
    manifest = Path("data/interim/manifest_with_concepts.parquet")
    if not manifest.exists():
        print(f"[labels] missing {manifest} — run `make data.concepts` first.", file=sys.stderr)
        return 2

    result = build_dataset(manifest, cfg)

    dataset_path = Path("data/processed/dataset.parquet")
    splits_path = Path("data/processed/splits.parquet")
    balance_path = Path("results/data.labels/balance.csv")
    ensure_dir(dataset_path.parent)
    ensure_dir(balance_path.parent)

    result.dataset.to_parquet(dataset_path, index=False)
    splits_to_frame(result.splits).to_parquet(splits_path, index=False)
    result.balance.to_csv(balance_path, index=False)

    per_split = {name: int(len(idx)) for name, idx in result.splits.items()}
    write_run_json(
        "data.labels",
        seed=int(cfg.download.subsample.seed),
        config=cfg,
        extra={
            "n_rows_final": int(len(result.dataset)),
            "n_per_split": per_split,
            "balance_csv": str(balance_path),
            "dataset_parquet": str(dataset_path),
            "splits_parquet": str(splits_path),
        },
    )
    print(f"[labels] wrote {dataset_path} ({len(result.dataset)} rows)")
    for name, n in per_split.items():
        print(f"[labels]   {name}: {n}")
    print(f"[labels] balance report: {balance_path}")
    return 0


def _run_raw() -> int:
    """P1 — verify that at least one GZ Evo parquet shard is on disk.

    The actual download lives in scripts/download_gz_evo.py; this stage
    just gates the pipeline so make data.concepts refuses to run empty.
    """
    cfg = load_config("data")
    raw_root = Path(cfg.download.root) / "gz_evo"
    shards = sorted(raw_root.rglob("*.parquet"))
    if not shards:
        print(
            f"[raw] no parquet shards under {raw_root}.\n"
            "[raw] run:  uv run python scripts/download_gz_evo.py "
            "--config default --split train --n-files 3",
            file=sys.stderr,
        )
        return 2
    print(f"[raw] {len(shards)} shard(s) present under {raw_root}:")
    for s in shards[:5]:
        print(f"[raw]   {s.name}")
    return 0


def _not_implemented(stage: str) -> int:
    print(f"[build_dataset] stage={stage!r} not implemented.", file=sys.stderr)
    return 1


def main() -> None:
    stage = _parse_stage(sys.argv)
    if stage == "raw":
        rc = _run_raw()
    elif stage == "concepts":
        rc = _run_concepts()
    elif stage == "labels":
        rc = _run_labels()
    else:
        rc = _not_implemented(stage)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
