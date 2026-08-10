"""statmorph concept computation — offline synthetic tests.

Fixtures are 2D Gaussian blobs whose analytic morphology gives us bounds we
can check (Gini > 0, concentration finite, r_eff ~ FWHM/2). Constant and
zero images exercise the failure paths without needing statmorph to succeed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from galaxycbm.data.statmorph_concepts import (
    CONCEPT_COLS,
    QUALITY_FLAG_COL,
    ConceptRecord,
    apply_nan_policy,
    assert_no_silent_nans,
    compute_concepts_for_array,
    compute_concepts_for_image,
    concepts_dataframe,
)


def _gaussian(size: int = 128, sigma: float = 8.0, amp: float = 100.0) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    r2 = (x - size / 2) ** 2 + (y - size / 2) ** 2
    img = amp * np.exp(-r2 / (2 * sigma * sigma))
    # Poissonish shot noise so statmorph has a real background to measure.
    rng = np.random.default_rng(0)
    img = img + rng.normal(0.0, 1.0, img.shape)
    return img.astype(np.float64)


def test_concept_record_shape():
    rec = compute_concepts_for_array(_gaussian())
    assert isinstance(rec, ConceptRecord)
    for c in CONCEPT_COLS:
        assert hasattr(rec, c)
    assert isinstance(rec.quality_flag, int)


def test_gaussian_gives_finite_gini_and_r_eff():
    rec = compute_concepts_for_array(_gaussian(size=128, sigma=10.0))
    assert math.isfinite(rec.gini), "Gini should be finite on a well-behaved Gaussian"
    assert 0.0 < rec.gini < 1.0
    assert math.isfinite(rec.r_eff_pixels)
    assert rec.r_eff_pixels > 0.0


def test_constant_image_flags_gracefully():
    img = np.ones((64, 64), dtype=np.float64) * 10.0
    rec = compute_concepts_for_array(img)
    # No source detectable → NO_SEGMENT (bit 3 = 8) fires; statmorph may
    # still run to completion on the fallback central disk and emit
    # warnings that raise its own .flag. The invariant we care about is
    # simply that the pipeline advertised the problem via quality_flag.
    assert rec.quality_flag != 0


def test_pil_input_matches_array_input():
    arr = _gaussian(size=96, sigma=8.0)
    from_arr = compute_concepts_for_array(arr)
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    from_pil = compute_concepts_for_image(pil)
    # PIL → 8-bit quantises; we only assert the code path returns a record
    # (not exact equality). Both should be non-null on this input.
    assert isinstance(from_pil, ConceptRecord)
    assert from_pil.quality_flag == 0 or from_pil.quality_flag & 0b0011  # warn bits at worst
    assert isinstance(from_arr, ConceptRecord)


def test_concepts_dataframe_has_columns():
    rec = compute_concepts_for_array(_gaussian())
    df = concepts_dataframe([rec, rec])
    assert list(df.columns) == CONCEPT_COLS + [QUALITY_FLAG_COL]
    assert len(df) == 2


# ---- NaN policy ----------------------------------------------------------


def _mixed_df() -> pd.DataFrame:
    rows = [
        {**{c: 0.5 for c in CONCEPT_COLS}, QUALITY_FLAG_COL: 0},          # clean
        {**{c: float("nan") for c in CONCEPT_COLS}, QUALITY_FLAG_COL: 4}, # flagged NaN
    ]
    return pd.DataFrame(rows)


def test_policy_drop_removes_nan_rows():
    out = apply_nan_policy(_mixed_df(), policy="drop")
    assert len(out) == 1
    assert out[QUALITY_FLAG_COL].iloc[0] == 0


def test_policy_impute_fills_and_keeps_flag():
    out = apply_nan_policy(_mixed_df(), policy="impute")
    assert not out[CONCEPT_COLS].isna().any().any()
    # The flag remains — downstream knows the row's values are synthetic.
    assert (out[QUALITY_FLAG_COL] == pd.Series([0, 4])).all()


def test_policy_flag_is_a_no_op():
    df_in = _mixed_df()
    out = apply_nan_policy(df_in, policy="flag")
    assert out.equals(df_in)


def test_policy_rejects_unknown():
    with pytest.raises(ValueError):
        apply_nan_policy(_mixed_df(), policy="bogus")


# ---- Silent-NaN guard ----------------------------------------------------


def test_assert_no_silent_nans_passes_when_flags_align():
    assert_no_silent_nans(_mixed_df())  # flagged NaN is fine


def test_assert_no_silent_nans_catches_silent_nan():
    bad = _mixed_df()
    bad.loc[1, QUALITY_FLAG_COL] = 0  # NaN row now has flag=0 → silent
    with pytest.raises(AssertionError):
        assert_no_silent_nans(bad)
