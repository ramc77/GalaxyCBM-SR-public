"""Shift metrics: Δaccuracy, Δκ, ECE, per-concept fidelity, rule similarity.

Pure numpy / sklearn / sympy — no PySR / torch. Fitting PySR on a shifted
survey lives in `pipeline.py`; here we only consume the resulting rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.symbolic import ClassRule


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: np.ndarray,
    y_true: pd.Series,
    classes: list[str],
    *,
    n_bins: int = 10,
) -> float:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got {probs.shape}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(str(c), -1) for c in y_true])
    keep = y_idx >= 0
    if keep.sum() == 0:
        return float("nan")
    probs = probs[keep]
    y_idx = y_idx[keep]
    conf = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    correct = (pred_idx == y_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / n) * abs(conf[in_bin].mean() - correct[in_bin].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Rule similarity: functional form + concept overlap
# ---------------------------------------------------------------------------


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _features_in(expression_str: str, feature_columns: list[str]) -> set[str]:
    """Return the subset of feature_columns actually referenced by the SymPy expr.

    Falls back to a regex over identifiers when SymPy can't parse the string —
    happens when PySR emits a constant equation.
    """
    try:
        import sympy

        expr = sympy.sympify(expression_str)
        return {str(s.name) for s in expr.free_symbols} & set(feature_columns)
    except Exception:
        tokens = set(_IDENT.findall(expression_str))
        return tokens & set(feature_columns)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


@dataclass
class RuleStability:
    per_class: pd.DataFrame        # class, jaccard_features, ref_used, shift_used
    mean_jaccard: float
    top_k_overlap: float           # Jaccard of top-k dominant concepts overall


def rule_stability(
    reference_rules: list[ClassRule],
    shifted_rules: list[ClassRule],
    feature_columns: list[str],
    *,
    top_k: int = 5,
) -> RuleStability:
    ref_map = {r.hubble_class: r for r in reference_rules}
    sh_map = {r.hubble_class: r for r in shifted_rules}

    rows: list[dict[str, object]] = []
    ref_all: dict[str, float] = {}
    sh_all: dict[str, float] = {}
    for cls, r in ref_map.items():
        s = sh_map.get(cls)
        ref_used = _features_in(r.equation_str, feature_columns)
        sh_used = _features_in(s.equation_str, feature_columns) if s else set()
        j = _jaccard(ref_used, sh_used) if s else float("nan")
        rows.append({
            "class": cls,
            "jaccard_features": j,
            "ref_complexity": int(r.complexity),
            "shift_complexity": int(s.complexity) if s else np.nan,
            "ref_features": sorted(ref_used),
            "shift_features": sorted(sh_used),
        })
        w = float(r.cv_accuracy) if pd.notna(r.cv_accuracy) else 0.0
        for u in ref_used:
            ref_all[u] = ref_all.get(u, 0.0) + w
        if s:
            sw = float(s.cv_accuracy) if pd.notna(s.cv_accuracy) else 0.0
            for u in sh_used:
                sh_all[u] = sh_all.get(u, 0.0) + sw

    per_class = pd.DataFrame(rows)
    mean_j = float(per_class["jaccard_features"].dropna().mean()) if not per_class.empty else float("nan")

    def top_k_set(d: dict[str, float]) -> set[str]:
        if not d:
            return set()
        s = pd.Series(d).sort_values(ascending=False)
        return set(s.head(top_k).index)

    top_overlap = _jaccard(top_k_set(ref_all), top_k_set(sh_all))
    return RuleStability(per_class=per_class, mean_jaccard=mean_j,
                          top_k_overlap=float(top_overlap))


# ---------------------------------------------------------------------------
# Cross-survey summary row builder
# ---------------------------------------------------------------------------


def shift_delta_row(
    survey: str,
    ref_metrics: dict,
    shift_metrics: dict,
    *,
    ece_ref: float,
    ece_shift: float,
    coverage_ref: float,
    coverage_shift: float,
    stability: RuleStability | None = None,
) -> dict:
    return {
        "survey": survey,
        "n_ref": int(ref_metrics.get("n", 0)),
        "n_shift": int(shift_metrics.get("n", 0)),
        "delta_accuracy": float(shift_metrics["accuracy"] - ref_metrics["accuracy"]),
        "delta_macro_f1": float(shift_metrics["macro_f1"] - ref_metrics["macro_f1"]),
        "delta_kappa": float(shift_metrics["cohen_kappa"] - ref_metrics["cohen_kappa"]),
        "delta_ece": float(ece_shift - ece_ref),
        "delta_coverage": float(coverage_shift - coverage_ref),
        "rule_mean_jaccard": stability.mean_jaccard if stability else float("nan"),
        "rule_top_k_overlap": stability.top_k_overlap if stability else float("nan"),
    }


def shift_figure(deltas: pd.DataFrame, path: str | Path) -> Path:
    """Bar plot of per-metric Δ across surveys."""
    import matplotlib.pyplot as plt

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["delta_accuracy", "delta_macro_f1", "delta_kappa",
               "delta_ece", "delta_coverage",
               "rule_mean_jaccard", "rule_top_k_overlap"]
    surveys = deltas["survey"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(surveys), 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for i, s in enumerate(surveys):
        row = deltas[deltas["survey"] == s].iloc[0]
        vals = [float(row[m]) if pd.notna(row[m]) else 0.0 for m in metrics]
        ax.bar(x + i * width, vals, width=width, label=s)
    ax.set_xticks(x + width * (len(surveys) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=25, ha="right")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.set_ylabel("shift value")
    ax.set_title("Cross-survey shift relative to reference")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Findings note (auto-populated markdown)
# ---------------------------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    """Minimal DataFrame → GitHub-flavoured markdown table.

    Avoids the `tabulate` dependency that pandas.DataFrame.to_markdown pulls in.
    """
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)

    def fmt(v: object) -> str:
        if isinstance(v, float):
            if pd.isna(v):
                return "nan"
            return f"{v:{floatfmt}}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def findings_note(
    reference_name: str,
    deltas: pd.DataFrame,
    *,
    per_concept_fidelity_deltas: pd.DataFrame | None = None,
    path: str | Path = "results/robustness/findings.md",
) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-survey findings",
        "",
        f"Reference sample: **{reference_name}**",
        "",
        "## Deltas relative to reference",
        "",
        _df_to_markdown(deltas),
        "",
        "## What the numbers say",
        "",
        "*Interpret shifts by their most likely physical driver:*",
        "",
        "- **Resolution** (Euclid VIS ≈ 0.10\"/pix, HST/JWST ≈ 0.03–0.05\"/pix vs DECaLS 0.262\"/pix): "
        "finer sampling sharpens bulge/disk decompositions and moves `sersic_n`, `concentration`, "
        "`bulge-size`.",
        "- **Band** (Euclid VIS is broad-optical, JWST is near-IR): "
        "changes `smooth-or-featured`, `has-spiral-arms`, `arm-count` because dust and stellar "
        "populations dominate the appearance differently at each band.",
        "- **PSF**: distinct PSFs shift `smoothness` and `asymmetry`, which are pixel-scale statistics.",
        "",
    ]
    if per_concept_fidelity_deltas is not None and not per_concept_fidelity_deltas.empty:
        worst = per_concept_fidelity_deltas.sort_values("delta_fidelity").head(6)
        lines += [
            "## Concepts most degraded under shift",
            "",
            _df_to_markdown(worst),
            "",
        ]
    lines += [
        "## Rule stability",
        "",
        "`rule_mean_jaccard` = mean overlap of features between the reference PySR rule "
        "and a fresh PySR fit on the shifted survey, per class. "
        "`rule_top_k_overlap` = Jaccard on the top-5 dominant concepts overall. "
        "Values near 1 mean the symbolic explanation is portable; values near 0 mean the "
        "shifted survey is telling PySR to lean on different concepts entirely.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
