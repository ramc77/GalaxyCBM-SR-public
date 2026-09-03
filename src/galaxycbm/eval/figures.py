"""Regenerate every publication figure from results/ artefacts.

Every figure is rebuilt from source data (parquet/CSV/JSON) rather than
copied, so a re-style at any point in the pipeline propagates without
running the training stages again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from galaxycbm.eval.style import apply_paper_style, palette, series_palette


def rule_pareto(rule_table_csv: Path, out: Path,
                fit_cache: Path | None = None) -> Path | None:
    """Accuracy-vs-complexity for the symbolic head.

    When the per-class PySR Pareto caches are available (written by Stage 2
    to results/symbolic/fit_cache/) the full frontier is drawn per class and
    the adopted expression is starred, which is what the caption claims.
    Without them only the adopted rules can be shown, and the axis label
    says so rather than implying a frontier that is not plotted.
    """
    if not Path(rule_table_csv).exists():
        return None
    apply_paper_style()
    import matplotlib.pyplot as plt

    df = pd.read_csv(rule_table_csv).reset_index(drop=True)
    cols = series_palette(len(df))

    # Stage 2 keys its cache on a hash of the training data, so earlier runs
    # on a different sample leave stale front files behind. Globbing by class
    # alone would pair a current adopted rule with a front computed on other
    # data. Accept a front only if it actually contains the adopted rule's
    # (complexity, cv_accuracy) point, which identifies the matching run.
    fronts: dict[str, pd.DataFrame] = {}
    if fit_cache is not None and Path(fit_cache).exists():
        for _, row in df.iterrows():
            cls = str(row["hubble_class"])
            safe = cls.replace("/", "_")
            for f in sorted(Path(fit_cache).glob(f"{safe}__*__pareto.parquet")):
                try:
                    pf = pd.read_parquet(f)
                except Exception:
                    continue
                if not {"complexity", "cv_accuracy"}.issubset(pf.columns):
                    continue
                hit = (
                    (pf["complexity"].astype(int) == int(row["complexity"]))
                    & (np.isclose(pf["cv_accuracy"].astype(float),
                                  float(row["cv_accuracy"]), atol=1e-6))
                ).any()
                if hit:
                    fronts[cls] = pf.sort_values("complexity")
                    break

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, row in df.iterrows():
        cls = str(row["hubble_class"])
        pf = fronts.get(cls)
        if pf is not None and len(pf):
            ax.plot(pf["complexity"], pf["cv_accuracy"], "-", color=cols[i],
                    lw=1.0, alpha=0.55, zorder=2)
        ax.scatter(row["complexity"], row["cv_accuracy"], s=110, color=cols[i],
                   marker="*" if fronts else "o",
                   edgecolor="black", linewidth=0.6, label=cls, zorder=5)

    ax.set_xlabel("Expression complexity (operator nodes)")
    ax.set_ylabel("Stratified $k$-fold CV accuracy")
    ax.set_title("Symbolic head: accuracy against complexity")

    # Zoom to the data. A 0-1 axis wastes most of the panel because every
    # one-vs-rest CV accuracy sits above 0.85.
    lo = float(min(df["cv_accuracy"].min(),
                   min((f["cv_accuracy"].min() for f in fronts.values()),
                       default=df["cv_accuracy"].min())))
    ax.set_ylim(max(0.0, lo - 0.03), 1.005)
    ax.legend(ncols=7, fontsize=7.5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, -0.34),
              handletextpad=0.3, columnspacing=0.8)
    if fronts:
        ax.set_title("Symbolic head: accuracy against complexity\n"
                     + r"$\mathdefault{lines:\ PySR\ Pareto\ front;\ stars:\ adopted\ rule}$",
                     fontsize=10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return Path(out)


def confusion_matrix(symbolic_metrics_json: Path, out: Path) -> Path | None:
    p = Path(symbolic_metrics_json)
    if not p.exists():
        return None
    import json

    data = json.loads(p.read_text())
    cm = np.asarray(data["confusion_matrix"], dtype=float)
    labels = list(data["labels"])
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sum > 0, cm / row_sum, 0.0)

    apply_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix — Stage-2 symbolic head")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85, label="row-normalised rate")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return Path(out)


def coverage_and_selective(uncertainty_dir: Path, out: Path) -> Path | None:
    curve_csv = Path(uncertainty_dir) / "selective_curve.csv"
    if not curve_csv.exists():
        return None
    apply_paper_style()
    import matplotlib.pyplot as plt

    curve = pd.read_csv(curve_csv)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    cols = palette(3)
    conf = curve[curve["policy"] == "by_confidence"].sort_values("abstain_fraction")
    # Beyond ~95% abstention the retained sample is tens of objects and the
    # curve is pure sampling noise; truncate rather than plot the spike.
    conf = conf[conf["abstain_fraction"] <= 0.95]
    ax.plot(conf["abstain_fraction"], conf["accuracy"],
            color=cols[1], label="rank by softmax confidence")
    sset = curve[curve["policy"] == "by_set_size"].sort_values("abstain_fraction")
    if not sset.empty:
        ax.scatter(sset["abstain_fraction"], sset["accuracy"],
                   marker="s", color=cols[2], edgecolor="black", linewidth=0.6,
                   label="accept if $|C(x)| < k$", zorder=5)
    ax.set_xlabel("Abstention rate"); ax.set_ylabel("Selective accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Selective risk — Stage-3 conformal wrapper")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return Path(out)


def robustness_shift(robustness_csv: Path, out: Path) -> Path | None:
    """Cross-survey transfer.

    Two panels, because the quantities are not commensurable: the left panel
    holds signed performance/calibration deltas (zero = no change), the right
    holds rule-stability indices bounded on [0, 1] (one = identical concept
    usage). Plotting them on a shared axis, as an earlier version did, invites
    the reader to compare a Delta-kappa against a Jaccard index.
    """
    p = Path(robustness_csv)
    if not p.exists():
        return None
    apply_paper_style()
    import matplotlib.pyplot as plt

    df = pd.read_csv(p)
    deltas  = ["delta_accuracy", "delta_macro_f1", "delta_kappa",
               "delta_ece", "delta_coverage"]
    stabil  = ["rule_mean_jaccard", "rule_top_k_overlap"]
    dlabels = [r"$\Delta$acc", r"$\Delta F_1$", r"$\Delta\kappa$",
               r"$\Delta$ECE", r"$\Delta$cov."]
    slabels = ["mean per-class $J$", "top-5 concept $J$"]

    surveys = df["survey"].tolist()
    cols = series_palette(len(surveys))
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [5, 2]})

    for ax, metrics, labels, title in (
        (axl, deltas, dlabels, "Performance and calibration shift"),
        (axr, stabil, slabels, "Rule stability"),
    ):
        x = np.arange(len(metrics))
        width = 0.8 / max(len(surveys), 1)
        for i, s in enumerate(surveys):
            row = df[df["survey"] == s].iloc[0]
            vals = [float(row[m]) if pd.notna(row[m]) else 0.0 for m in metrics]
            ax.bar(x + i * width, vals, width=width, color=cols[i],
                   edgecolor="black", linewidth=0.4,
                   label=s.replace("_", " ") if (ax is axl and len(surveys) > 1)
                   else None)
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_xticks(x + width * (len(surveys) - 1) / 2)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)

    axl.set_ylabel("shifted $-$ reference")
    axr.set_ylabel("Jaccard index")
    axr.set_ylim(0, 1)
    # A legend for a single survey is not just redundant: the swatch is the
    # same colour and shape as the bars, so it reads as a stray data bar
    # floating off the axis. Name the survey in the title instead.
    if len(surveys) > 1:
        axl.legend(loc="upper left", fontsize=8, frameon=False)
        subject = "cross-survey transfer"
    else:
        subject = surveys[0].replace("_", " ").replace("euclid q1", "Euclid Q1")
    fig.suptitle(f"{subject} relative to the DECaLS reference", fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return Path(out)


def concept_fidelity(fidelity_csv: Path, out: Path) -> Path | None:
    p = Path(fidelity_csv)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    apply_paper_style()
    import matplotlib.pyplot as plt

    df = df.set_index("vs_symbolic")[["spearman", "pearson"]]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    y = np.arange(len(df))
    cols = series_palette(2)
    ax.barh(y - 0.2, df["spearman"], height=0.35, color=cols[0],
            edgecolor="black", linewidth=0.4, label="Spearman $\\rho$")
    ax.barh(y + 0.2, df["pearson"], height=0.35, color=cols[1],
            edgecolor="black", linewidth=0.4, label="Pearson $r$")
    ax.axvline(0.0, color="k", lw=0.6, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([{"linear": "dense linear CBM",
                         "xgb": "XGBoost",
                         "xgb_shap": "XGBoost (SHAP)"}.get(i, i) for i in df.index])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlation with intrinsic symbolic weights")
    ax.set_title("Concept-importance fidelity")
    ax.legend(loc="best", fontsize=8, frameon=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return Path(out)


def coverage_comparison(revision_dir: Path, out: Path) -> Path | None:
    """Marginal vs class-conditional calibration: coverage, and its price.

    Two panels, because the result is a trade rather than an improvement:
    conditional validity is bought with prediction-set size, and a figure
    showing only coverage would misrepresent it.

    Returns None when scripts/run_revision.py has not been executed, so
    `build_all` stays usable on a fresh clone.
    """
    import matplotlib.pyplot as plt

    src = Path(revision_dir) / "mondrian_per_class.csv"
    if not src.exists():
        return None
    df = pd.read_csv(src)
    needed = {"class", "marginal_coverage", "mondrian_coverage", "n",
              "marginal_mean_set_size", "mondrian_mean_set_size"}
    if not needed <= set(df.columns):
        return None

    order = [c for c in ("E", "S0", "Sa", "Sb", "Sc", "Sd", "Irr")
             if c in set(df["class"])]
    df = df.set_index("class").loc[order].reset_index()
    degenerate = (df["degenerate"].astype(bool).to_numpy()
                  if "degenerate" in df.columns else np.zeros(len(df), bool))

    apply_paper_style()
    colors = series_palette(2)
    x = np.arange(len(df))
    width = 0.38
    n_classes = len(df)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)

    ax0.bar(x - width / 2, df["marginal_coverage"], width,
            label="marginal", color=colors[0])
    ax0.bar(x + width / 2, df["mondrian_coverage"], width,
            label="class-conditional", color=colors[1])
    ax0.axhline(0.90, color="k", lw=1.0, ls="--", zorder=4)
    ax0.annotate("nominal 0.90", xy=(n_classes - 0.5, 0.90), xytext=(0, 4),
                 textcoords="offset points", ha="right", fontsize=8)
    ax0.set_ylabel("empirical coverage")
    ax0.set_ylim(0, 1.30)
    ax0.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Legend above the axes: every in-axes corner collides with a bar.
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               fontsize=9, frameon=False)

    # A class calibrated on too few points is included by construction; say so
    # rather than letting a bar at 1.0 read as success.
    for i, deg in enumerate(degenerate):
        if deg:
            ax0.annotate(r"$n_{\rm cal}<9$", xy=(i + width / 2, 1.0),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5)

    ax1.bar(x - width / 2, df["marginal_mean_set_size"], width, color=colors[0])
    ax1.bar(x + width / 2, df["mondrian_mean_set_size"], width, color=colors[1])
    ax1.axhline(1.0, color="k", lw=0.8, ls=":", zorder=4)
    ax1.annotate("singleton", xy=(n_classes - 0.5, 1.0), xytext=(0, 4),
                 textcoords="offset points", ha="right", fontsize=8)
    ax1.set_ylabel(r"mean $|C(x)|$  (of %d)" % n_classes)
    ax1.set_ylim(0, n_classes)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n$n$={int(n)}" for c, n in zip(df["class"], df["n"])])
    fig.align_ylabels([ax0, ax1])
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    return Path(out)


def build_all(results_root: Path, paper_root: Path) -> list[Path]:
    figs_dir = Path(paper_root) / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for maker, args in [
        (rule_pareto,           (results_root / "symbolic" / "rule_table.csv", figs_dir / "pareto.pdf",
                                 results_root / "symbolic" / "fit_cache")),
        (confusion_matrix,      (results_root / "symbolic" / "metrics.json",   figs_dir / "confusion.pdf")),
        (coverage_and_selective,(results_root / "uncertainty",                 figs_dir / "selective_risk.pdf")),
        (robustness_shift,      (results_root / "tables" / "robustness.csv",   figs_dir / "robustness_shift.pdf")),
        (concept_fidelity,      (results_root / "baselines" / "fidelity.csv",  figs_dir / "concept_fidelity.pdf")),
        (coverage_comparison,   (results_root / "revision",                    figs_dir / "coverage_comparison.pdf")),
    ]:
        p = maker(*args)
        if p is not None:
            produced.append(p)
    return produced
