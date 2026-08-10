"""ConceptDataset.

Module-level torch import — this file is one of the STAGE1_ONLY_MODULES in
tests/test_smoke_imports.py (torch/lightning/zoobot are the `stage1` extra).
Keeping the class at module scope means the multi-worker DataLoader's
'spawn' start method (default on macOS) can pickle it cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from galaxycbm.concepts.heads import HeadSpec


def build_class_maps(heads: list[HeadSpec]) -> dict[str, dict[str, int]]:
    return {
        h.name: {c: i for i, c in enumerate(h.classes or ())}
        for h in heads if h.kind == "classification"
    }


class ConceptDataset(Dataset):
    """Yields (image, targets_dict, id_str). Missing categorical → -1
    (torch cross_entropy ignore_index); missing regression → NaN so the
    loss can mask.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cutouts_root: str | Path,
        heads: list[HeadSpec],
        *,
        size: int = 224,
        imagenet_norm: bool = True,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.cutouts_root = Path(cutouts_root)
        self.heads = list(heads)
        self.size = int(size)
        self.imagenet_norm = bool(imagenet_norm)
        self._class_maps = build_class_maps(self.heads)
        tfms = [transforms.Resize((self.size, self.size)), transforms.ToTensor()]
        if self.imagenet_norm:
            tfms.append(transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            ))
        self._pipeline = transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.cutouts_root / f"{row['id_str']}.png").convert("RGB")
        x = self._pipeline(img)
        targets: dict[str, torch.Tensor] = {}
        for h in self.heads:
            v = row[h.name] if h.name in row.index else None
            if h.kind == "classification":
                if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
                    targets[h.name] = torch.tensor(-1, dtype=torch.long)
                else:
                    targets[h.name] = torch.tensor(
                        self._class_maps[h.name][str(v)], dtype=torch.long,
                    )
            else:
                vf = float("nan") if v is None or pd.isna(v) else float(v)
                targets[h.name] = torch.tensor(vf, dtype=torch.float32)
        return x, targets, str(row["id_str"])


def get_dataset(
    df: pd.DataFrame,
    cutouts_root: str | Path,
    heads: list[HeadSpec],
    *,
    size: int = 224,
    imagenet_norm: bool = True,
) -> ConceptDataset:
    return ConceptDataset(df, cutouts_root, heads, size=size, imagenet_norm=imagenet_norm)


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ids = [b[2] for b in batch]
    keys = list(batch[0][1].keys())
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in keys}
    return xs, targets, ids
