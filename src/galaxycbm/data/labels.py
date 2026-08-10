"""Concept-label table + Hubble-type derivation + leakage-safe splits.

Input:  data/interim/manifest_with_concepts.parquet (from P2).
Output: data/processed/dataset.parquet   — concepts + hubble_type
        data/processed/splits.parquet    — (row_index, split_name) pairs
        results/data.labels/balance.csv  — per-split class counts

The Hubble-type target follows a compact version of the DECaLS GZ decision
tree (Walmsley et al. 2022). Perceptual concepts are the argmax of each
task's vote fractions, guarded by a config threshold. Physical concepts are
inherited untouched from statmorph. Splits are stratified by hubble_type
and defended by a two-layer leakage check: id equality + on-sky
near-duplicate (angular separation < dedup_radius_arcsec).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# GZ DECaLS decision-tree constants
# ---------------------------------------------------------------------------

DR5_TASK_ANSWERS: dict[str, list[str]] = {
    "smooth-or-featured": ["smooth", "featured-or-disk", "artifact"],
    "disk-edge-on": ["yes", "no"],
    "bar": ["strong", "weak", "no"],
    "has-spiral-arms": ["yes", "no"],
    "bulge-size": ["dominant", "large", "moderate", "small", "none"],
    "how-rounded": ["round", "in-between", "cigar-shaped"],
    "edge-on-bulge": ["boxy", "rounded", "none"],
    "spiral-winding": ["tight", "medium", "loose"],
    "spiral-arm-count": ["1", "2", "3", "4", "more-than-4", "cant-tell"],
    "merging": ["none", "minor-disturbance", "major-disturbance", "merger"],
}

HUBBLE_UNCLASSIFIED = "unclassified"


def _task_col(task: str, answer: str, suffix: str = "dr5") -> str:
    return f"{task}-{suffix}_{answer}_fraction"


def dominant_answer(
    row: pd.Series,
    task: str,
    answers: list[str],
    threshold: float,
    suffix: str = "dr5",
) -> str | None:
    """Argmax over answer fractions; returns None if the winner is under threshold.

    A None return is the honest "unclear" signal — the Hubble-type tree
    treats it as a soft branch (edge-on unclear → default no; winding
    unclear → default Sb), never as a silent success.
    """
    fractions: dict[str, float] = {}
    for ans in answers:
        col = _task_col(task, ans, suffix)
        if col in row.index and pd.notna(row[col]):
            fractions[ans] = float(row[col])
    if not fractions:
        return None
    winner = max(fractions, key=lambda k: fractions[k])
    return winner if fractions[winner] >= threshold else None


# ---------------------------------------------------------------------------
# Perceptual concepts
# ---------------------------------------------------------------------------


def derive_perceptual_concepts(
    df: pd.DataFrame,
    threshold: float,
    suffix: str = "dr5",
) -> pd.DataFrame:
    """One column per GZ task, holding the argmax answer or NaN if unclear."""
    out = pd.DataFrame(index=df.index)
    for task, answers in DR5_TASK_ANSWERS.items():
        out[task] = df.apply(
            lambda r, t=task, a=answers: dominant_answer(r, t, a, threshold, suffix),
            axis=1,
        )
    return out


# ---------------------------------------------------------------------------
# Hubble-type derivation
# ---------------------------------------------------------------------------


def derive_hubble_type(
    df: pd.DataFrame,
    threshold: float = 0.5,
    suffix: str = "dr5",
) -> pd.Series:
    """{E, S0, Sa, Sb, Sc, Sd, Irr, unclassified}.

    Priority: artifact → merger → smooth → edge-on featured → spiral vs
    non-spiral featured → winding × bulge for spiral sub-type.
    """
    labels: list[str] = []
    for _, row in df.iterrows():
        sof = dominant_answer(row, "smooth-or-featured",
                              DR5_TASK_ANSWERS["smooth-or-featured"], threshold, suffix)
        merging = dominant_answer(row, "merging",
                                  DR5_TASK_ANSWERS["merging"], threshold, suffix)

        if sof == "artifact":
            labels.append(HUBBLE_UNCLASSIFIED); continue
        if merging == "merger":
            labels.append("Irr"); continue
        if sof == "smooth":
            labels.append("E"); continue
        if sof != "featured-or-disk":
            labels.append(HUBBLE_UNCLASSIFIED); continue

        edge = dominant_answer(row, "disk-edge-on",
                               DR5_TASK_ANSWERS["disk-edge-on"], threshold, suffix)
        spirals = dominant_answer(row, "has-spiral-arms",
                                  DR5_TASK_ANSWERS["has-spiral-arms"], threshold, suffix)
        if edge == "yes":
            labels.append("S0"); continue
        if spirals != "yes":
            labels.append("S0"); continue

        winding = dominant_answer(row, "spiral-winding",
                                  DR5_TASK_ANSWERS["spiral-winding"], threshold, suffix)
        bulge = dominant_answer(row, "bulge-size",
                                DR5_TASK_ANSWERS["bulge-size"], threshold, suffix)
        large_bulge = bulge in ("dominant", "large")

        if winding == "tight":
            labels.append("Sa" if large_bulge else "Sb")
        elif winding == "medium":
            labels.append("Sb" if large_bulge else "Sc")
        elif winding == "loose":
            labels.append("Sd" if bulge in (None, "none") else "Sc")
        else:
            labels.append("Sb")  # winding unclear → canonical intermediate

    return pd.Series(labels, index=df.index, name="hubble_type")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _cut(idx: np.ndarray, fractions: dict[str, float]) -> dict[str, np.ndarray]:
    n = len(idx)
    names = list(fractions)
    out: dict[str, np.ndarray] = {}
    start = 0
    for i, name in enumerate(names):
        end = n if i == len(names) - 1 else start + int(round(n * fractions[name]))
        out[name] = idx[start:end]
        start = end
    return out


def make_splits(
    df: pd.DataFrame,
    *,
    fractions: dict[str, float],
    seed: int,
    stratify_by: str | None = "hubble_type",
) -> dict[str, np.ndarray]:
    """Row-index arrays per split. Stratified by ``stratify_by`` when present."""
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {sum(fractions.values())}")
    rng = np.random.default_rng(seed)
    if stratify_by is None or stratify_by not in df.columns:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        return _cut(idx, fractions)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in fractions}
    for _, group in df.groupby(stratify_by):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        for name, arr in _cut(idx, fractions).items():
            parts[name].append(arr)
    return {
        k: (np.concatenate(v) if v else np.array([], dtype=int))
        for k, v in parts.items()
    }


# ---------------------------------------------------------------------------
# Leakage assertion (id-equality + on-sky near-duplicate)
# ---------------------------------------------------------------------------


def assert_no_leakage(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    *,
    id_col: str = "id_str",
    ra_col: str = "ra",
    dec_col: str = "dec",
    dedup_radius_arcsec: float = 3.0,
) -> None:
    """Refuse to advance if any id appears in more than one split OR any two
    objects across splits are within ``dedup_radius_arcsec`` on the sky.
    """
    seen: dict[str, str] = {}
    for name, idx in splits.items():
        for i in idx:
            oid = str(df.at[i, id_col])
            if oid in seen and seen[oid] != name:
                raise AssertionError(
                    f"leakage: id={oid!r} appears in splits {seen[oid]!r} and {name!r}"
                )
            seen[oid] = name

    coords: dict[str, tuple[pd.DataFrame, SkyCoord]] = {}
    for name, idx in splits.items():
        sub = df.loc[idx]
        if len(sub) == 0:
            continue
        coords[name] = (
            sub,
            SkyCoord(ra=sub[ra_col].to_numpy() * u.deg,
                     dec=sub[dec_col].to_numpy() * u.deg),
        )
    names = list(coords)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sc_a, sc_b = coords[a][1], coords[b][1]
            if len(sc_a) == 0 or len(sc_b) == 0:
                continue
            _, sep2d, _ = sc_a.match_to_catalog_sky(sc_b)
            close = sep2d.arcsecond < dedup_radius_arcsec
            if close.any():
                n_pairs = int(close.sum())
                raise AssertionError(
                    f"leakage: {n_pairs} object(s) in split {a!r} lie within "
                    f'{dedup_radius_arcsec}" of an object in split {b!r}'
                )


# ---------------------------------------------------------------------------
# Class balance report
# ---------------------------------------------------------------------------


def class_balance(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    *,
    target_col: str = "hubble_type",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, idx in splits.items():
        counts = df.loc[idx, target_col].value_counts()
        for cls, n in counts.items():
            rows.append({"split": name, "class": str(cls), "n": int(n)})
    return pd.DataFrame(rows).sort_values(["split", "class"]).reset_index(drop=True)


def splits_to_frame(splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long-format (row_index, split_name) — durable on-disk representation."""
    parts = [
        pd.DataFrame({"row_index": np.asarray(idx, dtype=np.int64), "split": name})
        for name, idx in splits.items()
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["row_index", "split"]
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class LabelBuildResult:
    dataset: pd.DataFrame
    splits: dict[str, np.ndarray]
    balance: pd.DataFrame


def build_dataset(manifest_path: str | Path, cfg: DictConfig) -> LabelBuildResult:
    df = pd.read_parquet(manifest_path)
    labels_cfg = cfg.labels

    if labels_cfg.get("drop_flagged", True):
        df = df[df["statmorph_quality_flag"] == 0].reset_index(drop=True)

    primary = list(labels_cfg.get("primary_datasets", ["gz_dr5", "gz_dr8"]))
    if "dataset_name" in df.columns:
        matched = df["dataset_name"].isin(primary)
        if not matched.any():
            observed = sorted(df["dataset_name"].dropna().unique().tolist())
            raise ValueError(
                "labels.primary_datasets excludes every row. "
                f"Configured {primary!r}; observed dataset_name values {observed!r}."
            )
        df = df[matched].reset_index(drop=True)

    suffix = str(labels_cfg.get("suffix", "dr5"))
    threshold = float(labels_cfg.get("clean_sample_threshold", 0.5))
    perc = derive_perceptual_concepts(df, threshold=threshold, suffix=suffix)
    df = pd.concat([df, perc], axis=1)
    df["hubble_type"] = derive_hubble_type(df, threshold=threshold, suffix=suffix)

    if labels_cfg.get("drop_unclassified", True):
        df = df[df["hubble_type"] != HUBBLE_UNCLASSIFIED].reset_index(drop=True)

    fractions = OmegaConf.to_container(labels_cfg.splits, resolve=True)
    if not isinstance(fractions, dict):
        raise TypeError(f"labels.splits must be a mapping, got {type(fractions)}")
    splits = make_splits(
        df,
        fractions={k: float(v) for k, v in fractions.items()},
        seed=int(cfg.download.subsample.seed),
        stratify_by="hubble_type",
    )

    assert_no_leakage(
        df, splits,
        dedup_radius_arcsec=float(labels_cfg.get("dedup_radius_arcsec", 3.0)),
    )

    return LabelBuildResult(dataset=df, splits=splits, balance=class_balance(df, splits))
