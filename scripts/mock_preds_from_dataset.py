"""Build a stand-in `results/concepts/preds.parquet` from ground-truth concepts.

Purpose: unblock Stage 2 development on machines that can't run Stage 1
(macOS x86_64 has no torch wheels). Perceptual concepts become one-hot
probability columns; physical concepts pass through verbatim. Results are
an upper bound on what Stage 2 can achieve — swap for the real Stage-1
output as soon as you can.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.concepts import build_head_specs, prob_columns
from galaxycbm.utils import load_config
from galaxycbm.utils.io import ensure_dir


def main() -> None:
    concepts_cfg = load_config("concepts")
    heads = build_head_specs(concepts_cfg)

    dataset_path = Path("data/processed/dataset.parquet")
    splits_path = Path("data/processed/splits.parquet")
    for p in (dataset_path, splits_path):
        if not p.exists():
            print(f"[mock] missing {p} — run `make data.labels` first.", file=sys.stderr)
            raise SystemExit(2)

    ds = pd.read_parquet(dataset_path)
    splits = pd.read_parquet(splits_path)
    row_to_split = dict(zip(splits["row_index"].to_numpy(), splits["split"].to_numpy()))
    ds = ds.reset_index(drop=True)
    ds["split"] = ds.index.map(row_to_split).astype("string")

    out = pd.DataFrame({"id_str": ds["id_str"].astype(str).to_numpy(),
                        "split": ds["split"].to_numpy()})

    for h in heads:
        if h.kind == "classification":
            if h.name not in ds.columns:
                continue
            true = ds[h.name].astype("string")
            for c in h.classes or ():
                out[f"{h.name}__{c}"] = (true == c).astype(float).where(true.notna(), np.nan)
        else:  # regression: statmorph value passes through
            if h.name in ds.columns:
                out[h.name] = ds[h.name].astype(float)

    dropped = [c for c in prob_columns_all(heads) if c not in out.columns]
    if dropped:
        print(f"[mock] warning: {len(dropped)} expected prob columns not in dataset "
              "(silently dropped — Stage 1 will fill these in for real):")
        for c in dropped[:6]:
            print(f"[mock]   {c}")

    preds_path = Path("results/concepts/preds.parquet")
    ensure_dir(preds_path.parent)
    out.to_parquet(preds_path, index=False)
    print(f"[mock] wrote {preds_path}  ({len(out)} rows, {out.shape[1]} cols)")
    print(f"[mock] {out['split'].value_counts().to_dict()}")


def prob_columns_all(heads) -> list[str]:
    return [c for h in heads for c in prob_columns(h)]


if __name__ == "__main__":
    main()
