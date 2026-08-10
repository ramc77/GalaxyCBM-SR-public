"""Offline tests for the concept-label build.

No network, no parquet on disk — synthesise the vote-fraction columns
directly and exercise the decision tree, splits, and leakage guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxycbm.data.labels import (
    DR5_TASK_ANSWERS,
    HUBBLE_UNCLASSIFIED,
    assert_no_leakage,
    class_balance,
    derive_hubble_type,
    derive_perceptual_concepts,
    dominant_answer,
    make_splits,
    splits_to_frame,
)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _row(**overrides: float) -> dict[str, float]:
    """Build a row with every DR5 vote fraction set to 0, then overlay overrides.

    Keys in overrides are the short form 'smooth-or-featured_smooth' etc.
    (without the '-dr5' suffix); the helper adds the suffix and '_fraction'.
    """
    row: dict[str, float] = {}
    for task, answers in DR5_TASK_ANSWERS.items():
        for ans in answers:
            row[f"{task}-dr5_{ans}_fraction"] = 0.0
    for key, val in overrides.items():
        task, ans = key.split("__")
        row[f"{task}-dr5_{ans}_fraction"] = float(val)
    return row


def _df(rows: list[dict[str, float]], *, ids=None, ras=None, decs=None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    n = len(df)
    df["id_str"] = ids if ids is not None else [f"g{i:04d}" for i in range(n)]
    df["ra"] = ras if ras is not None else np.linspace(10.0, 20.0, n)
    df["dec"] = decs if decs is not None else np.linspace(-5.0, 5.0, n)
    return df


# ---------------------------------------------------------------------------
# dominant_answer
# ---------------------------------------------------------------------------


def test_dominant_answer_picks_argmax_over_threshold():
    row = pd.Series(_row(**{"smooth-or-featured__smooth": 0.8,
                            "smooth-or-featured__featured-or-disk": 0.15,
                            "smooth-or-featured__artifact": 0.05}))
    got = dominant_answer(row, "smooth-or-featured",
                          DR5_TASK_ANSWERS["smooth-or-featured"], threshold=0.5)
    assert got == "smooth"


def test_dominant_answer_returns_none_under_threshold():
    row = pd.Series(_row(**{"smooth-or-featured__smooth": 0.4,
                            "smooth-or-featured__featured-or-disk": 0.35,
                            "smooth-or-featured__artifact": 0.25}))
    assert dominant_answer(row, "smooth-or-featured",
                           DR5_TASK_ANSWERS["smooth-or-featured"], threshold=0.5) is None


# ---------------------------------------------------------------------------
# Hubble-type decision tree
# ---------------------------------------------------------------------------


def test_smooth_row_labels_E():
    df = _df([_row(**{"smooth-or-featured__smooth": 0.9})])
    lab = derive_hubble_type(df, threshold=0.5)
    assert lab.iloc[0] == "E"


def test_edge_on_disk_labels_S0():
    df = _df([_row(**{
        "smooth-or-featured__featured-or-disk": 0.9,
        "disk-edge-on__yes": 0.9,
    })])
    assert derive_hubble_type(df, threshold=0.5).iloc[0] == "S0"


def test_featured_no_spirals_labels_S0():
    df = _df([_row(**{
        "smooth-or-featured__featured-or-disk": 0.9,
        "disk-edge-on__no": 0.9,
        "has-spiral-arms__no": 0.9,
    })])
    assert derive_hubble_type(df, threshold=0.5).iloc[0] == "S0"


def test_tight_winding_large_bulge_labels_Sa():
    df = _df([_row(**{
        "smooth-or-featured__featured-or-disk": 0.9,
        "disk-edge-on__no": 0.9,
        "has-spiral-arms__yes": 0.9,
        "spiral-winding__tight": 0.9,
        "bulge-size__large": 0.9,
    })])
    assert derive_hubble_type(df, threshold=0.5).iloc[0] == "Sa"


def test_loose_winding_no_bulge_labels_Sd():
    df = _df([_row(**{
        "smooth-or-featured__featured-or-disk": 0.9,
        "disk-edge-on__no": 0.9,
        "has-spiral-arms__yes": 0.9,
        "spiral-winding__loose": 0.9,
        "bulge-size__none": 0.9,
    })])
    assert derive_hubble_type(df, threshold=0.5).iloc[0] == "Sd"


def test_merger_labels_Irr():
    df = _df([_row(**{
        "smooth-or-featured__featured-or-disk": 0.9,
        "merging__merger": 0.9,
    })])
    assert derive_hubble_type(df, threshold=0.5).iloc[0] == "Irr"


def test_artifact_labels_unclassified():
    df = _df([_row(**{"smooth-or-featured__artifact": 0.9})])
    assert derive_hubble_type(df, threshold=0.5).iloc[0] == HUBBLE_UNCLASSIFIED


def test_derive_perceptual_concepts_has_one_column_per_task():
    df = _df([_row(**{"smooth-or-featured__smooth": 0.9,
                      "how-rounded__round": 0.9})])
    out = derive_perceptual_concepts(df, threshold=0.5)
    assert set(out.columns) == set(DR5_TASK_ANSWERS.keys())
    assert out["smooth-or-featured"].iloc[0] == "smooth"
    assert out["how-rounded"].iloc[0] == "round"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _labelled_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = rng.choice(["E", "S0", "Sa", "Sb", "Sc"], size=n)
    df = pd.DataFrame({
        "id_str": [f"g{i:04d}" for i in range(n)],
        "ra": rng.uniform(0.0, 360.0, size=n),
        "dec": rng.uniform(-30.0, 30.0, size=n),
        "hubble_type": labels,
    })
    return df


def test_splits_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        make_splits(_labelled_df(50), fractions={"train": 0.7, "val": 0.2}, seed=1)


def test_splits_cover_every_row_exactly_once():
    df = _labelled_df(200)
    splits = make_splits(df, fractions={"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    joined = np.sort(np.concatenate(list(splits.values())))
    np.testing.assert_array_equal(joined, np.arange(len(df)))


def test_splits_are_deterministic_under_seed():
    df = _labelled_df(120)
    a = make_splits(df, fractions={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    b = make_splits(df, fractions={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    for k in a:
        np.testing.assert_array_equal(np.sort(a[k]), np.sort(b[k]))


def test_stratified_splits_preserve_class_ratios():
    df = _labelled_df(500)
    splits = make_splits(df, fractions={"train": 0.6, "val": 0.2, "test": 0.2},
                         seed=0, stratify_by="hubble_type")
    global_ratios = df["hubble_type"].value_counts(normalize=True)
    for name, idx in splits.items():
        r = df.loc[idx, "hubble_type"].value_counts(normalize=True)
        # loose tolerance — 500 rows is small
        for cls in global_ratios.index:
            assert abs(r.get(cls, 0.0) - global_ratios[cls]) < 0.1, (
                f"stratification skew: split={name} class={cls}"
            )


def test_splits_to_frame_roundtrips():
    df = _labelled_df(30)
    splits = make_splits(df, fractions={"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    long = splits_to_frame(splits)
    assert set(long.columns) == {"row_index", "split"}
    assert len(long) == len(df)


# ---------------------------------------------------------------------------
# Leakage assertion
# ---------------------------------------------------------------------------


def test_leakage_guard_passes_on_disjoint_splits():
    df = _labelled_df(30)
    splits = make_splits(df, fractions={"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    assert_no_leakage(df, splits, dedup_radius_arcsec=0.1)


def test_leakage_guard_catches_shared_id():
    df = _labelled_df(20)
    splits = {"train": np.arange(0, 10), "test": np.arange(9, 20)}  # id at row 9 shared
    df.at[9, "id_str"] = "collision"
    df.at[10, "id_str"] = "collision"  # same id_str, different row → cross-split match
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(df, splits)


def test_leakage_guard_catches_near_duplicate_ra_dec():
    df = _labelled_df(20)
    # Put a row in test very close to a row in train.
    df.at[0, "ra"] = 100.0; df.at[0, "dec"] = 20.0
    df.at[15, "ra"] = 100.0 + 1e-5; df.at[15, "dec"] = 20.0
    splits = {"train": np.array([0, 1, 2, 3, 4]),
              "test":  np.array([15, 16, 17, 18, 19])}
    with pytest.raises(AssertionError, match="within"):
        assert_no_leakage(df, splits, dedup_radius_arcsec=1.0)


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------


def test_class_balance_totals_equal_split_sizes():
    df = _labelled_df(120)
    splits = make_splits(df, fractions={"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    bal = class_balance(df, splits)
    for name, idx in splits.items():
        assert bal.loc[bal["split"] == name, "n"].sum() == len(idx)
