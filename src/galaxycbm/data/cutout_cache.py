"""Decode HF `mwalmsley/gz_evo` parquet shards → per-id PNGs on disk.

Cheaper than re-decoding parquet blobs every mini-batch during training.
No torch dependency — pure PIL + pandas.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def cache_cutouts(
    shards: list[Path],
    out_root: str | Path,
    *,
    size: int = 224,
    id_col: str = "id_str",
    image_col: str = "image",
    ids: set[str] | None = None,
) -> int:
    """Save 1-per-id PNGs to `out_root/<id>.png`. Skip anything already cached.

    If `ids` is given, only rows whose id is in the set are extracted — lets
    the training driver pass in just the ids that survive P3's label build.
    Returns the number of newly written files.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for shard in shards:
        df = pd.read_parquet(shard, columns=[id_col, image_col])
        it = df.itertuples(index=False, name=None)
        for oid, raw in tqdm(it, total=len(df), desc=Path(shard).name, unit="img", leave=False):
            oid = str(oid)
            if ids is not None and oid not in ids:
                continue
            path = out_root / f"{oid}.png"
            if path.exists():
                continue
            b = raw["bytes"] if isinstance(raw, dict) else raw
            img = Image.open(io.BytesIO(b)).convert("RGB").resize((size, size))
            img.save(path, format="PNG", optimize=True)
            n_written += 1
    return n_written
