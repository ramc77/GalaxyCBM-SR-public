"""Concept-Bottleneck Lightning module.

Zoobot backbone (config-selectable ConvNeXt / EfficientNet) with one linear
head per concept: cross-entropy for GZ tasks, Smooth L1 for statmorph.
All torch/lightning/zoobot imports happen inside functions so this file
remains importable in the torch-free base env.
"""

from __future__ import annotations

from galaxycbm.concepts.heads import HeadSpec


def _load_zoobot_encoder(name: str, greyscale: bool = False):
    """Return (encoder_module, feature_dim). Verified against zoobot 2.9's
    FinetuneableZoobotAbstract: `.encoder` is a plain nn.Module.
    """
    import torch
    from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

    # Instantiate a throwaway classifier just to reach the pretrained encoder;
    # we build our own multi-head on top and discard `.head`.
    stub = FinetuneableZoobotClassifier(
        num_classes=2,
        name=name,
        greyscale=greyscale,
        learning_rate=0.0,
        prog_bar=False,
    )
    encoder = stub.encoder
    with torch.no_grad():
        dummy = torch.zeros(1, 1 if greyscale else 3, 224, 224)
        feat = encoder(dummy)
        if feat.ndim > 2:
            feat = feat.flatten(1)
    return encoder, int(feat.shape[-1])


def build_module(cfg_model, heads: list[HeadSpec]):
    """Assemble the Lightning module. Heavy imports live here."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import lightning as L  # noqa: N812

    backbone_name = f"hf_hub:mwalmsley/zoobot-encoder-{cfg_model.backbone.name}"
    encoder, feat_dim = _load_zoobot_encoder(backbone_name)

    head_modules = nn.ModuleDict({
        h.name: nn.Linear(feat_dim, h.n_classes if h.kind == "classification" else 1)
        for h in heads
    })
    weight_p = float(cfg_model.heads.concept_weight_perceptual)
    weight_ph = float(cfg_model.heads.concept_weight_physical)
    lr = float(cfg_model.train.lr)
    wd = float(cfg_model.train.weight_decay)

    class CBM(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.heads_mod = head_modules
            self.save_hyperparameters({
                "heads": [h.name for h in heads],
                "backbone": backbone_name,
                "feature_dim": feat_dim,
            })

        def forward(self, x):
            f = self.encoder(x)
            if f.ndim > 2:
                f = f.flatten(1)
            return {name: mod(f) for name, mod in self.heads_mod.items()}

        def _step(self, batch, stage: str):
            x, targets, _ = batch
            preds = self(x)
            total = None
            for h in heads:
                y = targets[h.name]
                p = preds[h.name]
                if h.kind == "classification":
                    loss = F.cross_entropy(p, y, ignore_index=-1)
                    w = weight_p
                else:
                    mask = ~torch.isnan(y)
                    if mask.any():
                        loss = F.smooth_l1_loss(p.squeeze(-1)[mask], y[mask])
                    else:
                        loss = torch.tensor(0.0, device=x.device)
                    w = weight_ph
                total = w * loss if total is None else total + w * loss
                self.log(f"{stage}/{h.name}", loss, on_step=False, on_epoch=True)
            self.log(f"{stage}/loss", total, on_step=False, on_epoch=True, prog_bar=True)
            return total

        def training_step(self, batch, batch_idx):
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):
            self._step(batch, "val")

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

    return CBM()


def predict_dataframe(module, dataloader, heads: list[HeadSpec], device: str = "cpu"):
    """Run inference; return a DataFrame with id_str + one column per head/class."""
    import pandas as pd
    import torch
    import torch.nn.functional as F

    module.eval().to(device)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for x, _, ids in dataloader:
            x = x.to(device)
            out = module(x)
            for i, oid in enumerate(ids):
                row: dict[str, object] = {"id_str": oid}
                for h in heads:
                    logits = out[h.name][i]
                    if h.kind == "classification":
                        probs = F.softmax(logits, dim=-1).cpu().numpy()
                        for c, name in enumerate(h.classes or ()):
                            row[f"{h.name}__{name}"] = float(probs[c])
                    else:
                        row[h.name] = float(logits.squeeze().cpu().item())
                rows.append(row)
    return pd.DataFrame(rows)
