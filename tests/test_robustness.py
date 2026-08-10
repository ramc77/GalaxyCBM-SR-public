"""Offline robustness tests — no HF network, no PySR fit.

Covers: ECE, rule-stability Jaccard, shift-delta row builder, findings-note
markdown writer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxycbm.robustness import (
    RuleStability,
    expected_calibration_error,
    findings_note,
    rule_stability,
    shift_delta_row,
    shift_figure,
)
from galaxycbm.symbolic import ClassRule


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_zero_when_perfect_calibration():
    classes = ["A", "B"]
    # Two rows, each fully confident and correct.
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = pd.Series(["A", "B"])
    assert expected_calibration_error(probs, y, classes) == pytest.approx(0.0, abs=1e-9)


def test_ece_high_when_confidence_wrong():
    classes = ["A", "B"]
    probs = np.array([[0.99, 0.01], [0.99, 0.01]])   # both class-A predictions
    y = pd.Series(["A", "B"])                        # only half correct
    ece = expected_calibration_error(probs, y, classes)
    assert ece > 0.4


def test_ece_ignores_unknown_true_labels():
    probs = np.array([[0.7, 0.3], [0.6, 0.4]])
    y = pd.Series(["A", "C"])  # C not in classes
    assert not np.isnan(expected_calibration_error(probs, y, ["A", "B"]))


# ---------------------------------------------------------------------------
# Rule stability
# ---------------------------------------------------------------------------


def _rules(eqs: dict[str, str], cv: float = 0.8) -> list[ClassRule]:
    return [
        ClassRule(hubble_class=c, equation_str=e, latex=e,
                  complexity=len(e), pysr_score=1.0, cv_accuracy=cv)
        for c, e in eqs.items()
    ]


def test_rule_stability_perfect_match():
    features = ["fA", "fB", "fC", "fD"]
    ref = _rules({"X": "fA + fB", "Y": "fC"})
    same = _rules({"X": "fA + fB", "Y": "fC"})
    stab = rule_stability(ref, same, features)
    assert isinstance(stab, RuleStability)
    assert stab.mean_jaccard == pytest.approx(1.0)
    assert stab.top_k_overlap == pytest.approx(1.0)


def test_rule_stability_zero_overlap():
    features = ["fA", "fB", "fC", "fD"]
    ref = _rules({"X": "fA", "Y": "fB"})
    other = _rules({"X": "fC", "Y": "fD"})
    stab = rule_stability(ref, other, features)
    assert stab.mean_jaccard == pytest.approx(0.0)
    assert stab.top_k_overlap == pytest.approx(0.0)


def test_rule_stability_partial_overlap():
    features = ["fA", "fB", "fC"]
    ref = _rules({"X": "fA + fB", "Y": "fC"})
    other = _rules({"X": "fA", "Y": "fA + fC"})
    stab = rule_stability(ref, other, features)
    # X: {fA,fB} vs {fA} → 1/2
    # Y: {fC} vs {fA,fC} → 1/2
    assert stab.mean_jaccard == pytest.approx(0.5)


def test_rule_stability_handles_missing_shifted_class():
    features = ["fA", "fB"]
    ref = _rules({"X": "fA", "Y": "fB"})
    other = _rules({"X": "fA"})
    stab = rule_stability(ref, other, features)
    # Y row has NaN jaccard → mean over the one defined class = 1.0
    assert not np.isnan(stab.mean_jaccard)


# ---------------------------------------------------------------------------
# Delta row + figure + note
# ---------------------------------------------------------------------------


def test_shift_delta_row_arithmetic():
    ref = {"n": 100, "accuracy": 0.80, "macro_f1": 0.60, "cohen_kappa": 0.55}
    shift = {"n": 100, "accuracy": 0.70, "macro_f1": 0.50, "cohen_kappa": 0.40}
    stab = RuleStability(per_class=pd.DataFrame(),
                          mean_jaccard=0.4, top_k_overlap=0.5)
    row = shift_delta_row("euclid_q1", ref, shift,
                          ece_ref=0.05, ece_shift=0.11,
                          coverage_ref=0.90, coverage_shift=0.72,
                          stability=stab)
    assert row["survey"] == "euclid_q1"
    assert row["delta_accuracy"] == pytest.approx(-0.10)
    assert row["delta_kappa"] == pytest.approx(-0.15)
    assert row["delta_ece"] == pytest.approx(0.06)
    assert row["delta_coverage"] == pytest.approx(-0.18)
    assert row["rule_mean_jaccard"] == 0.4


def test_shift_figure_writes(tmp_path):
    deltas = pd.DataFrame([
        shift_delta_row("euclid_q1",
                        {"n": 10, "accuracy": 0.8, "macro_f1": 0.6, "cohen_kappa": 0.5},
                        {"n": 10, "accuracy": 0.7, "macro_f1": 0.5, "cohen_kappa": 0.4},
                        ece_ref=0.1, ece_shift=0.2, coverage_ref=0.9, coverage_shift=0.8),
        shift_delta_row("jwst_cosmos",
                        {"n": 10, "accuracy": 0.8, "macro_f1": 0.6, "cohen_kappa": 0.5},
                        {"n": 10, "accuracy": 0.5, "macro_f1": 0.4, "cohen_kappa": 0.2},
                        ece_ref=0.1, ece_shift=0.3, coverage_ref=0.9, coverage_shift=0.6),
    ])
    p = shift_figure(deltas, tmp_path / "shift.png")
    assert p.exists() and p.stat().st_size > 0


def test_findings_note_markdown(tmp_path):
    deltas = pd.DataFrame([
        shift_delta_row("euclid_q1",
                        {"n": 100, "accuracy": 0.8, "macro_f1": 0.6, "cohen_kappa": 0.5},
                        {"n": 100, "accuracy": 0.7, "macro_f1": 0.5, "cohen_kappa": 0.4},
                        ece_ref=0.05, ece_shift=0.10, coverage_ref=0.9, coverage_shift=0.8),
    ])
    p = findings_note("DECaLS", deltas, path=tmp_path / "findings.md")
    text = p.read_text()
    assert p.exists()
    assert "Cross-survey findings" in text
    assert "euclid_q1" in text
    assert "Rule stability" in text
