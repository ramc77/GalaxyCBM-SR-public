"""Eval subpackage — aggregation, tables, style, consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from galaxycbm.eval import (
    ClaimReport,
    aggregate_metrics,
    apply_paper_style,
    check_claims,
    dataframe_to_latex,
    load_claims_manifest,
    palette,
    path_exists_and_is_populated,
    resolve_path,
    write_table,
)
from galaxycbm.eval.figures import (
    concept_fidelity,
    confusion_matrix,
    robustness_shift,
    rule_pareto,
)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def test_palette_length_and_first_color():
    p = palette(5)
    assert len(p) == 5
    assert p[0] == "#000000"


def test_apply_paper_style_sets_serif():
    import matplotlib as mpl

    apply_paper_style()
    assert "serif" in mpl.rcParams["font.family"] or mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["pdf.fonttype"] == 42


# ---------------------------------------------------------------------------
# Aggregate + path helpers
# ---------------------------------------------------------------------------


def _write_stage(root: Path, name: str, files: dict[str, object]) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for fname, payload in files.items():
        p = d / fname
        if fname.endswith(".json"):
            p.write_text(json.dumps(payload) + "\n")
        elif fname.endswith(".csv"):
            pd.DataFrame(payload).to_csv(p, index=False)


def test_aggregate_reads_populated_stage(tmp_path):
    _write_stage(tmp_path, "symbolic", {
        "run.json": {"stage": "symbolic", "seed": 1},
        "metrics.json": {"accuracy": 0.7, "macro_f1": 0.5, "cohen_kappa": 0.4,
                          "labels": ["E", "S"], "confusion_matrix": [[3, 1], [0, 4]]},
        "rule_table.csv": [{"hubble_class": "E", "expression": "gini",
                             "complexity": 2, "cv_accuracy": 0.8}],
    })
    metrics = aggregate_metrics(tmp_path)
    assert metrics["stages"]["symbolic"]["metrics"]["accuracy"] == 0.7
    assert metrics["stages"]["symbolic"]["rules"][0]["hubble_class"] == "E"


def test_aggregate_marks_missing_stages_as_null(tmp_path):
    metrics = aggregate_metrics(tmp_path)   # empty dir
    assert metrics["stages"]["symbolic"]["run_json"] is None
    assert metrics["stages"]["symbolic"]["rules"] is None


def test_resolve_path_navigates_lists_and_dicts():
    obj = {"a": [{"b": 1}, {"b": 2}]}
    assert resolve_path(obj, "a.0.b") == 1
    assert resolve_path(obj, "a.1.b") == 2


def test_path_exists_is_false_for_empty_containers():
    assert not path_exists_and_is_populated({"a": []}, "a")
    assert not path_exists_and_is_populated({"a": ""}, "a")
    assert path_exists_and_is_populated({"a": 0}, "a")   # 0 is a valid value
    assert not path_exists_and_is_populated({"a": None}, "a")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_dataframe_to_latex_escapes_underscores():
    df = pd.DataFrame({"model_name": ["dense_linear_cbm"], "acc": [0.85]})
    tex = dataframe_to_latex(df, caption="c", label="l")
    assert r"dense\_linear\_cbm" in tex
    assert r"\begin{tabular}" in tex
    assert r"\bottomrule" in tex


def test_write_table_produces_both_files(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    csv, tex = write_table(df, tmp_path / "t.csv", tmp_path / "t.tex",
                            caption="c", label="tab:l")
    assert csv.exists() and tex.exists()
    assert "0.100" in tex.read_text()


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def test_check_claims_flags_missing_and_ok(tmp_path):
    metrics = {"stages": {"symbolic": {"metrics": {"accuracy": 0.7}}}}
    claims = [
        {"path": "stages.symbolic.metrics.accuracy"},
        {"path": "stages.symbolic.metrics.cohen_kappa"},
        {"path": "stages.concepts.fidelity", "optional": True},
    ]
    report = check_claims(metrics, claims)
    assert isinstance(report, ClaimReport)
    assert report.present == 1
    assert report.missing == ["stages.symbolic.metrics.cohen_kappa"]
    assert report.missing_optional == ["stages.concepts.fidelity"]
    assert not report.ok


def test_load_claims_manifest_reads_from_disk(tmp_path):
    p = tmp_path / "claims.yaml"
    p.write_text("claims:\n  - path: stages.symbolic.metrics.accuracy\n"
                  "  - path: stages.concepts.fidelity\n    optional: true\n")
    claims = load_claims_manifest(p)
    assert len(claims) == 2
    assert claims[0]["path"] == "stages.symbolic.metrics.accuracy"
    assert claims[1].get("optional") is True


# ---------------------------------------------------------------------------
# Figures — return None cleanly when source data missing
# ---------------------------------------------------------------------------


def test_figures_return_none_when_source_missing(tmp_path):
    assert rule_pareto(tmp_path / "missing.csv", tmp_path / "out.pdf") is None
    assert confusion_matrix(tmp_path / "missing.json", tmp_path / "out.pdf") is None
    assert robustness_shift(tmp_path / "missing.csv", tmp_path / "out.pdf") is None
    assert concept_fidelity(tmp_path / "missing.csv", tmp_path / "out.pdf") is None


def test_rule_pareto_writes(tmp_path):
    rule_csv = tmp_path / "rule_table.csv"
    pd.DataFrame({
        "hubble_class": ["E", "S", "Irr"],
        "expression": ["gini", "sersic_n", "asymmetry"],
        "complexity": [2, 4, 3],
        "cv_accuracy": [0.85, 0.70, 0.60],
    }).to_csv(rule_csv, index=False)
    out = rule_pareto(rule_csv, tmp_path / "out.pdf")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_confusion_matrix_writes(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({
        "accuracy": 0.6, "labels": ["E", "S"],
        "confusion_matrix": [[3, 1], [0, 4]],
    }))
    out = confusion_matrix(tmp_path / "metrics.json", tmp_path / "out.pdf")
    assert out is not None and out.exists()
