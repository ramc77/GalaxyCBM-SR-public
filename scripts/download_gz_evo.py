"""Download GZ Evo parquet shards from the HuggingFace Hub.

Source: mwalmsley/gz_evo — https://huggingface.co/datasets/mwalmsley/gz_evo
Paper:  Walmsley et al. 2024, arXiv:2404.02973
License: CC-BY-NC-SA-4.0

Configs on the hub:
    tiny     — ~183 MB, 8k rows, 1 train file + 1 test file (debug)
    default  — ~18.3 GB, 806k rows, 30 train files + 7 test files

Each row has: image (PIL), id_str, dataset_name, ra, dec, and per-survey
vote-fraction columns (dr5, dr8, gz2, dr12, candels, hubble, ukidss).

Usage:
    uv run python scripts/download_gz_evo.py --config tiny
    uv run python scripts/download_gz_evo.py --config default --split train --n-files 3
    uv run python scripts/download_gz_evo.py --config default --split train  # all 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "mwalmsley/gz_evo"

# Per-config shard counts, confirmed from the HF parquet API:
#   https://huggingface.co/api/datasets/mwalmsley/gz_evo/parquet
N_SHARDS = {
    "default": {"train": 30, "test": 7},
    "tiny":    {"train": 1,  "test": 1},
}


def _shard_path(config: str, split: str, i: int) -> str:
    # Confirmed layout (curl of /api/datasets/mwalmsley/gz_evo/tree/main):
    #   default:  data/{split}-NNNNN-of-000{30|07}.parquet
    #   tiny:     tiny/{split}-00000-of-00001.parquet
    total = N_SHARDS[config][split]
    folder = "data" if config == "default" else "tiny"
    return f"{folder}/{split}-{i:05d}-of-{total:05d}.parquet"


def download(config: str, split: str, n_files: int | None, out: Path) -> list[Path]:
    total = N_SHARDS[config][split]
    n = total if n_files is None else min(n_files, total)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        rel = _shard_path(config, split, i)
        print(f"[{i+1}/{n}] {rel}")
        p = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=rel,
            local_dir=out,
        )
        paths.append(Path(p))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["default", "tiny"], default="tiny")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--n-files", type=int, default=None,
                    help="Cap shards downloaded (default: all shards in the split).")
    ap.add_argument("--out", type=Path, default=Path("data/raw/gz_evo"))
    args = ap.parse_args()

    paths = download(args.config, args.split, args.n_files, args.out)
    print(f"\nDone. {len(paths)} file(s) in {args.out}")


if __name__ == "__main__":
    main()
