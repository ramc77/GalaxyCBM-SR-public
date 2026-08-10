"""(a) End-to-end ConvNeXt classifier (no bottleneck) + (d) SmoothGrad saliency.

Both live behind the `stage1` extra (torch, lightning, zoobot). Every heavy
import is inside the function bodies so this module stays importable in the
base env — the driver falls back to a "skipped" result when torch is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


class _EndToEndDataset:
    """Plain (image, label_idx, id_str) dataset for the black-box CNN baseline.

    Deliberately separate from galaxycbm.concepts.dataset.ConceptDataset:
    that one feeds our bespoke multi-head CBM Lightning module (which
    unpacks a (x, targets_dict, ids) tuple by hand), while this one feeds
    Zoobot's OWN FinetuneableZoobotClassifier, whose internal
    training_step/validation_step call batch_to_supervised_tuple() and
    expect the batch to be dict-indexable: batch['image'], batch['label'].
    Two different Lightning modules, two different batch contracts.
    """

    def __init__(self, df: pd.DataFrame, cutouts_root: str | Path,
                class_to_idx: dict[str, int], *, size: int = 224) -> None:
        from torchvision import transforms

        self.df = df.reset_index(drop=True)
        self.cutouts_root = Path(cutouts_root)
        self.class_to_idx = class_to_idx
        self.pipeline = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.df.iloc[i]
        img = Image.open(self.cutouts_root / f"{row['id_str']}.png").convert("RGB")
        x = self.pipeline(img)
        y = self.class_to_idx[str(row["hubble_type"])]
        return x, y, str(row["id_str"])


def _endtoend_fit_collate(batch):
    """For trainer.fit() — Zoobot's own step functions index batch['image']/['label']."""
    import torch

    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {"image": xs, "label": ys}


def _endtoend_predict_collate(batch):
    """For our own post-training inference loop — needs id_str, which Zoobot's format has no room for."""
    import torch

    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ids = [b[2] for b in batch]
    return xs, ys, ids


def train_endtoend_convnext(
    dataset_df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    cutouts_root: str | Path,
    model_cfg,
    *,
    seed: int = 0,
) -> tuple[object, dict, pd.DataFrame]:
    """Fine-tune a Zoobot ConvNeXt as a plain N-way classifier over hubble_type.

    Returns (module, val_metrics, val_predictions_df).
    """
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

    from galaxycbm.symbolic import compute_metrics

    classes = sorted(dataset_df["hubble_type"].dropna().astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def fit_loader(df: pd.DataFrame, shuffle: bool):
        ds = _EndToEndDataset(df, cutouts_root, class_to_idx, size=224)
        return torch.utils.data.DataLoader(
            ds, batch_size=int(model_cfg.train.batch_size),
            shuffle=shuffle, num_workers=int(model_cfg.train.num_workers),
            collate_fn=_endtoend_fit_collate,
            persistent_workers=(int(model_cfg.train.num_workers) > 0),
        )

    def predict_loader(df: pd.DataFrame):
        ds = _EndToEndDataset(df, cutouts_root, class_to_idx, size=224)
        return torch.utils.data.DataLoader(
            ds, batch_size=int(model_cfg.train.batch_size),
            shuffle=False, num_workers=int(model_cfg.train.num_workers),
            collate_fn=_endtoend_predict_collate,
            persistent_workers=(int(model_cfg.train.num_workers) > 0),
        )

    train_dl = fit_loader(splits["train"], shuffle=True)
    val_dl = fit_loader(splits["val"], shuffle=False)

    backbone = f"hf_hub:mwalmsley/zoobot-encoder-{model_cfg.backbone.name}"
    module = FinetuneableZoobotClassifier(
        num_classes=len(classes),
        name=backbone,
        learning_rate=float(model_cfg.train.lr),
        weight_decay=float(model_cfg.train.weight_decay),
        head_dropout_prob=0.3,
        prog_bar=True,
        seed=seed,
    )
    # This baseline overfits within a few epochs (15M parameters, ~24k images).
    # Without a checkpoint callback the module keeps its FINAL weights, i.e. the
    # most overfit ones, while the symbolic head is selected by cross-validated
    # accuracy. Comparing those two states biases the table in the symbolic
    # head's favour, so we checkpoint on validation loss and restore the best
    # weights before evaluating.
    ckpt_dir = Path("models/baseline_endtoend")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="cnn-{epoch:02d}",
        monitor="finetuning/val_loss", mode="min", save_top_k=1,
    )
    trainer = L.Trainer(
        max_epochs=int(model_cfg.train.max_epochs),
        precision=str(model_cfg.train.precision),
        gradient_clip_val=float(model_cfg.train.gradient_clip_val),
        callbacks=[
            EarlyStopping(monitor="finetuning/val_loss",
                          patience=int(model_cfg.train.early_stopping_patience),
                          mode="min"),
            ckpt_cb,
        ],
        default_root_dir=str(ckpt_dir),
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dl, val_dl)

    best_path = ckpt_cb.best_model_path
    if best_path and Path(best_path).exists():
        print(f"[baselines] restoring best CNN checkpoint: {best_path} "
              f"(val_loss={ckpt_cb.best_model_score})")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"])
    else:
        print("[baselines] WARNING: no CNN checkpoint found; evaluating final "
              "(likely overfit) weights.", file=sys.stderr)

    module.eval()
    rows: list[dict[str, object]] = []
    predict_dl = predict_loader(splits["val"])
    with torch.no_grad():
        for x, y, ids in predict_dl:
            logits = module(x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            for i, oid in enumerate(ids):
                rows.append({"id_str": oid, "hubble_pred": classes[int(preds[i])]})
    preds_df = pd.DataFrame(rows)

    y_true = splits["val"].set_index("id_str").loc[preds_df["id_str"], "hubble_type"].reset_index(drop=True)
    y_pred = preds_df["hubble_pred"]
    metrics = compute_metrics(y_true, y_pred)
    return module, metrics, preds_df


def smoothgrad_saliency(
    module: object,
    dataset_df: pd.DataFrame,
    cutouts_root: str | Path,
    *,
    n_samples: int = 20,
    noise_sigma: float = 0.15,
    n_images: int = 16,
    out_path: str | Path = "results/baselines/smoothgrad.png",
) -> Path:
    """Compute SmoothGrad saliency for the black-box CNN on a handful of images."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    module.eval()
    ids = dataset_df["id_str"].astype(str).head(n_images).tolist()
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.6, 3.2))
    for j, oid in enumerate(ids):
        img = Image.open(Path(cutouts_root) / f"{oid}.png").convert("RGB")
        x = pipeline(img).unsqueeze(0).requires_grad_(True)
        sal = torch.zeros_like(x)
        for _ in range(n_samples):
            xn = x + noise_sigma * torch.randn_like(x)
            xn.requires_grad_(True)
            logits = module(xn)
            top = logits.argmax(dim=-1)
            score = logits[0, top]
            grad = torch.autograd.grad(score, xn)[0]
            sal = sal + grad.abs()
        sal_map = sal.mean(dim=1).squeeze().detach().cpu().numpy()
        axes[0, j].imshow(np.asarray(img)); axes[0, j].axis("off")
        axes[1, j].imshow(sal_map, cmap="hot"); axes[1, j].axis("off")
    fig.suptitle("SmoothGrad — end-to-end CNN"); fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def interpretability_cost(module: object) -> int:
    """Trainable parameter count (proxy for black-box opacity)."""
    import torch  # noqa: F401

    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
