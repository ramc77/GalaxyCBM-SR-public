"""Physical concepts per cutout via statmorph.

Verified against statmorph 0.7.2 (Rodriguez-Gomez et al. 2019):
    - Entry point:  statmorph.source_morphology(image, segmap, **kwargs) -> list
    - Attributes:   .concentration, .asymmetry, .smoothness,
                    .gini, .m20,
                    .sersic_n, .rhalf_ellip (non-parametric half-light radius),
                    .flag (0=OK … 4=catastrophic),
                    .flag_sersic (0=OK, 1=warn, 2=bad).

Segmentation map is built by photutils detect_sources with a central-source
policy. PSF and weight map are optional — see cfg.statmorph.

Failure policy: per-object graceful. Any raise / silent NaN → quality_flag
gets a bit set and every concept for that row is NaN. Downstream stages call
:func:`assert_no_silent_nans` to refuse to advance if a NaN slips through
without a matching flag.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statmorph
from photutils.segmentation import detect_sources
from PIL import Image
from tqdm.auto import tqdm

CONCEPT_COLS: list[str] = [
    "concentration",
    "asymmetry",
    "smoothness",
    "gini",
    "m20",
    "sersic_n",
    "r_eff_pixels",
]

# Per-row wall-clock cap. A tiny minority of GZ Evo images push statmorph's
# Sérsic fitter into worst-case iteration loops that stall for minutes.
# When SIGALRM fires we NaN the row and set _BIT_TIMEOUT so P3 can filter.
DEFAULT_ROW_TIMEOUT_SECONDS = 60
QUALITY_FLAG_COL = "statmorph_quality_flag"

# Quality-flag bits (independent, so they OR together):
_BIT_STATMORPH_WARN = 1 << 0   # 1  — statmorph .flag ≥ 1
_BIT_SERSIC_WARN = 1 << 1      # 2  — statmorph .flag_sersic ≥ 1
_BIT_EXCEPTION = 1 << 2        # 4  — python raise inside the call
_BIT_NO_SEGMENT = 1 << 3       # 8  — segmentation found no central source
_BIT_TIMEOUT = 1 << 4          # 16 — per-row wall-clock cap exceeded


@dataclass
class ConceptRecord:
    concentration: float
    asymmetry: float
    smoothness: float
    gini: float
    m20: float
    sersic_n: float
    r_eff_pixels: float
    quality_flag: int


# ---------------------------------------------------------------------------
# Image → grayscale numpy
# ---------------------------------------------------------------------------


def _grayscale(image: Image.Image | np.ndarray | bytes) -> np.ndarray:
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.float64)
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        # ITU-R BT.709 luminance
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return arr


def _default_segmap(
    image: np.ndarray,
    *,
    threshold_sigma: float = 2.5,
    npixels: int = 25,
) -> tuple[np.ndarray, bool]:
    """Central-source segmentation via photutils.

    Returns (segmap, ok). `ok=False` if no source could be found and a
    fallback central disk was used — the caller should set the NO_SEGMENT bit.

    Background is estimated from the frame corners (galaxy fills the middle
    on pre-rendered GZ cutouts, so global median is biased upward); noise
    from the sigma-clipped MAD of that corner sample.
    """
    h, w = image.shape
    corner = min(h, w) // 6
    corners = np.concatenate([
        image[:corner, :corner].ravel(),
        image[:corner, -corner:].ravel(),
        image[-corner:, :corner].ravel(),
        image[-corner:, -corner:].ravel(),
    ])
    bkg = float(np.median(corners))
    std = float(1.4826 * np.median(np.abs(corners - bkg))) or 1.0
    threshold = bkg + threshold_sigma * std
    segm = detect_sources(image, threshold, npixels=npixels)
    if segm is None or segm.nlabels == 0:
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        r = min(h, w) // 4
        fallback = ((y - h // 2) ** 2 + (x - w // 2) ** 2 <= r * r).astype(np.int32)
        return fallback, False
    labels = segm.data
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    central = int(labels[cy, cx])
    if central == 0:
        vals, counts = np.unique(labels[labels > 0], return_counts=True)
        central = int(vals[int(np.argmax(counts))])
    return (labels == central).astype(np.int32), True


# ---------------------------------------------------------------------------
# Core: run statmorph on one array
# ---------------------------------------------------------------------------


def _nan_record(quality_flag: int) -> ConceptRecord:
    return ConceptRecord(
        *(float("nan"),) * len(CONCEPT_COLS),
        quality_flag=quality_flag,
    )


class _RowTimeout(Exception):
    pass


def _install_timeout(seconds: int):
    """Install a SIGALRM-based per-row timer. Returns the previous handler
    so the caller can restore it. Only works on Unix + main thread; a
    no-op on unsupported platforms.
    """
    import signal

    def _handler(signum, frame):
        raise _RowTimeout()

    try:
        prev = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(max(1, seconds)))
        return prev
    except (ValueError, AttributeError):
        return None


def _clear_timeout(prev) -> None:
    import signal

    try:
        signal.alarm(0)
        if prev is not None:
            signal.signal(signal.SIGALRM, prev)
    except (ValueError, AttributeError):
        pass


def compute_concepts_for_array(
    image: np.ndarray,
    *,
    gain: float = 1.0,
    psf: np.ndarray | None = None,
    weightmap: np.ndarray | None = None,
    timeout_seconds: int = DEFAULT_ROW_TIMEOUT_SECONDS,
) -> ConceptRecord:
    """Compute the concept vector for a single 2D image (grayscale, float64)."""
    quality = 0
    prev_alarm = _install_timeout(timeout_seconds) if timeout_seconds else None
    try:
        segmap, seg_ok = _default_segmap(image)
        if not seg_ok:
            quality |= _BIT_NO_SEGMENT
        # Sky-subtract using the same corner estimate the segmap used —
        # silences the "Image is not background-subtracted" warning and
        # gives statmorph a real zero-point.
        h, w = image.shape
        corner = min(h, w) // 6
        corners = np.concatenate([
            image[:corner, :corner].ravel(),
            image[:corner, -corner:].ravel(),
            image[-corner:, :corner].ravel(),
            image[-corner:, -corner:].ravel(),
        ])
        image_bs = image - float(np.median(corners))
        results = statmorph.source_morphology(
            image_bs, segmap, gain=gain, psf=psf, weightmap=weightmap,
        )
        if not results:
            return _nan_record(quality | _BIT_EXCEPTION)
        m = results[0]
        if int(getattr(m, "flag", 0)) >= 1:
            quality |= _BIT_STATMORPH_WARN
        if int(getattr(m, "flag_sersic", 0)) >= 1:
            quality |= _BIT_SERSIC_WARN
        return ConceptRecord(
            concentration=float(getattr(m, "concentration", np.nan)),
            asymmetry=float(getattr(m, "asymmetry", np.nan)),
            smoothness=float(getattr(m, "smoothness", np.nan)),
            gini=float(getattr(m, "gini", np.nan)),
            m20=float(getattr(m, "m20", np.nan)),
            sersic_n=float(getattr(m, "sersic_n", np.nan)),
            r_eff_pixels=float(getattr(m, "rhalf_ellip", np.nan)),
            quality_flag=quality,
        )
    except _RowTimeout:
        return _nan_record(quality | _BIT_TIMEOUT)
    except Exception:
        return _nan_record(quality | _BIT_EXCEPTION)
    finally:
        _clear_timeout(prev_alarm)


def compute_concepts_for_image(image: Image.Image | np.ndarray | bytes, **kwargs) -> ConceptRecord:
    return compute_concepts_for_array(_grayscale(image), **kwargs)


def _worker_compute(image_bytes: bytes, gain: float) -> ConceptRecord:
    """Module-level (picklable) entry point for ProcessPoolExecutor workers.

    Each worker is a fresh interpreter (macOS defaults to 'spawn'), so the
    per-row SIGALRM timeout in compute_concepts_for_array installs cleanly
    per-process — no special handling needed here.
    """
    return compute_concepts_for_image(image_bytes, gain=gain)


# ---------------------------------------------------------------------------
# DataFrame-level API
# ---------------------------------------------------------------------------


def concepts_dataframe(records: Iterable[ConceptRecord]) -> pd.DataFrame:
    rows = list(records)
    return pd.DataFrame(
        {c: [getattr(r, c) for r in rows] for c in CONCEPT_COLS}
        | {QUALITY_FLAG_COL: [r.quality_flag for r in rows]}
    )


def compute_concepts_for_hf_shard(
    shard_path: str | Path,
    *,
    image_col: str = "image",
    id_col: str = "id_str",
    keep_cols: Iterable[str] = ("id_str", "dataset_name", "ra", "dec", "summary"),
    keep_vote_fractions: bool = True,
    survey_suffixes: Iterable[str] = ("dr5", "dr8"),
    gain: float = 1.0,
    max_rows: int | None = None,
    progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 100,
    batch_size: int = 500,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Read one HF `mwalmsley/gz_evo` parquet shard and return a concept table.

    Memory strategy: load the shard via pandas (which reconstructs HF's
    nested ``image`` struct into a dict column), then iterate in fixed-
    size batches, dropping references to processed slices and forcing
    ``gc.collect()`` between batches. Statmorph / astropy accumulate
    internal state across calls; on a 22k-row shard without GC pauses
    macOS OOM-kills the process around row 18k.

    Intra-shard resume via ``checkpoint_path``: every ``checkpoint_every``
    rows the partial concept table is flushed to disk. On restart, that
    file is loaded and rows whose ``id_str`` is already in it are skipped.

    ``n_workers > 1`` parallelizes statmorph across a ProcessPoolExecutor
    (one process per worker — statmorph/scipy hold the GIL, so threads
    would not help). Result order matches submission order (``Executor.map``
    guarantee), so no id-matching bookkeeping is needed. BLAS thread counts
    are pinned to 1 per worker before the pool starts: N worker processes
    each spawning their own multi-threaded BLAS would oversubscribe the
    machine's cores and net *less* throughput than running serially.
    """
    import gc
    import os
    from concurrent.futures import ProcessPoolExecutor

    if n_workers and n_workers > 1:
        for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(_env, "1")

    shard_path = Path(shard_path)
    df = pd.read_parquet(shard_path)
    if image_col not in df.columns:
        raise KeyError(f"{shard_path}: missing image column {image_col!r}")
    if id_col not in df.columns:
        raise KeyError(f"{shard_path}: missing id column {id_col!r}")
    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)
    df[id_col] = df[id_col].astype(str)

    # Metadata columns to carry through.
    keep = [c for c in keep_cols if c in df.columns]
    if keep_vote_fractions:
        suffixes = tuple(f"-{s}_" for s in survey_suffixes)
        keep = keep + [
            c for c in df.columns
            if c.endswith("_fraction") and any(sfx in c for sfx in suffixes)
        ]
    if id_col not in keep:
        keep = [id_col] + keep

    # Resume: which ids are already in the checkpoint?
    done_ids: set[str] = set()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            done_df = pd.read_parquet(checkpoint_path, columns=[id_col])
            done_ids = set(done_df[id_col].astype(str))
            print(f"[concepts] {shard_path.name}: "
                  f"resuming from checkpoint, {len(done_ids)} rows already done")
            del done_df; gc.collect()

    todo_idx = df.index[~df[id_col].isin(done_ids)].to_numpy()
    n_todo = int(todo_idx.size)
    if n_todo == 0:
        assert checkpoint_path is not None
        return pd.read_parquet(checkpoint_path)

    pbar = tqdm(total=n_todo, desc=shard_path.name, unit="gal", leave=False) if progress else None

    processed_ids: list[str] = []
    processed_records: list[ConceptRecord] = []
    pending_meta_rows: list[pd.Series] = []

    def _flush() -> None:
        if not processed_ids or checkpoint_path is None:
            return
        new_concepts = concepts_dataframe(processed_records)
        new_meta = pd.DataFrame(pending_meta_rows).reset_index(drop=True)
        combined = pd.concat([new_meta, new_concepts], axis=1)
        if checkpoint_path.exists():
            prev = pd.read_parquet(checkpoint_path)
            combined = pd.concat([prev, combined], ignore_index=True)
            del prev
        combined.to_parquet(checkpoint_path, index=False)
        processed_ids.clear()
        processed_records.clear()
        pending_meta_rows.clear()
        gc.collect()

    # Dispatch granularity == checkpoint granularity. `pool.map()` (and the
    # serial list-comprehension fallback) is a BLOCKING call: nothing is
    # returned — and therefore nothing can be flushed to the checkpoint —
    # until every row in the dispatched chunk finishes. Dispatching in
    # `batch_size`-sized chunks (500) meant a Ctrl-C anywhere inside a
    # chunk lost up to 500 rows of completed work with nothing saved.
    # Dispatching in `checkpoint_every`-sized chunks (100) bounds that loss
    # back to the original ~100-row resume granularity.
    dispatch_step = max(1, int(checkpoint_every))
    gc_step = max(dispatch_step, int(batch_size))
    rows_since_gc = 0

    pool = ProcessPoolExecutor(max_workers=int(n_workers)) if n_workers and n_workers > 1 else None
    try:
        for start in range(0, n_todo, dispatch_step):
            chunk = todo_idx[start:start + dispatch_step]
            chunk_bytes = []
            for i in chunk:
                raw = df.at[int(i), image_col]
                chunk_bytes.append(raw["bytes"] if isinstance(raw, dict) else raw)

            if pool is not None:
                records = list(pool.map(_worker_compute, chunk_bytes, [gain] * len(chunk_bytes)))
            else:
                records = [compute_concepts_for_image(b, gain=gain) for b in chunk_bytes]

            for i, rec in zip(chunk, records):
                processed_records.append(rec)
                processed_ids.append(df.at[int(i), id_col])
                pending_meta_rows.append(df.loc[int(i), keep])
                if pbar:
                    pbar.update(1)

            _flush()  # every dispatch chunk IS a checkpoint chunk now
            rows_since_gc += len(chunk)
            if rows_since_gc >= gc_step:
                gc.collect()
                rows_since_gc = 0
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    _flush()
    if pbar:
        pbar.close()

    # Prefer the on-disk checkpoint (complete + durable).
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        return pd.read_parquet(checkpoint_path)
    concepts = concepts_dataframe(processed_records)
    ordered_meta = (
        pd.DataFrame(pending_meta_rows).reset_index(drop=True)
        if pending_meta_rows else pd.DataFrame(columns=keep)
    )
    return pd.concat([ordered_meta, concepts], axis=1)


# ---------------------------------------------------------------------------
# NaN / outlier policy
# ---------------------------------------------------------------------------


def apply_nan_policy(
    df: pd.DataFrame,
    *,
    policy: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """policy ∈ {'drop', 'impute', 'flag'}.

    - drop:   remove any row with a NaN in a concept column.
    - impute: median-fill NaNs. Quality flag stays set so P4 knows the
              value is synthetic.
    - flag:   leave NaNs; downstream must handle them explicitly.
    """
    cols = columns or CONCEPT_COLS
    if policy == "drop":
        return df.dropna(subset=cols).reset_index(drop=True)
    if policy == "impute":
        medians = df[cols].median(numeric_only=True)
        out = df.copy()
        out[cols] = out[cols].fillna(medians)
        return out
    if policy == "flag":
        return df.copy()
    raise ValueError(f"Unknown NaN policy: {policy!r}. Expected drop|impute|flag.")


def assert_no_silent_nans(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    quality_col: str = QUALITY_FLAG_COL,
) -> None:
    """After the NaN policy has been applied, a NaN in a concept column MUST
    be accompanied by a non-zero quality flag. A silent NaN (NaN + flag=0)
    means the concept computation lied about success — refuse to advance.
    """
    cols = columns or CONCEPT_COLS
    if quality_col not in df.columns:
        raise AssertionError(f"missing quality column {quality_col!r}")
    nan_mask = df[cols].isna().any(axis=1)
    silent = nan_mask & (df[quality_col] == 0)
    if silent.any():
        n = int(silent.sum())
        raise AssertionError(
            f"{n} row(s) have NaN concept values but quality_flag=0. "
            "This is a silent failure and blocks the pipeline."
        )
