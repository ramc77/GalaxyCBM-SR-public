"""Per-concept fidelity: AUC (classification) / RMSE (regression), vs trivial baseline.

Pure numpy/sklearn/matplotlib. No torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from galaxycbm.concepts.heads import HeadSpec, prob_columns


def _classification_auc(y_true: pd.Series, y_pred_probs: pd.DataFrame,
                        classes: tuple[str, ...]) -> float:
    from sklearn.metrics import roc_auc_score

    mask = y_true.notna()
    if mask.sum() < 2:
        return float("nan")
    y = y_true[mask].astype(str)
    p = y_pred_probs.loc[mask]
    y_hot = pd.get_dummies(y).reindex(columns=list(classes), fill_value=0).to_numpy()
    if y_hot.sum(axis=0).min() == 0:
        # A class is missing from the ground truth → macro-AUC undefined.
        return float("nan")
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_hot[:, 1], p.iloc[:, 1].to_numpy(float)))
        return float(roc_auc_score(y_hot, p.to_numpy(float), multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _regression_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.notna() & y_pred.notna()
    if not mask.any():
        return float("nan")
    err = y_true[mask].to_numpy(float) - y_pred[mask].to_numpy(float)
    return float(np.sqrt(np.mean(err * err)))


def per_concept_fidelity(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    heads: list[HeadSpec],
    *,
    y_train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """AUC vs 0.5 for classification; RMSE vs σ(train-target) for regression."""
    rows: list[dict[str, object]] = []
    for h in heads:
        if h.kind == "classification":
            cols = prob_columns(h)
            if any(c not in y_pred.columns for c in cols) or h.name not in y_true.columns:
                rows.append({
                    "concept": h.name, "kind": h.kind, "metric": "AUC",
                    "value": float("nan"), "n": 0,
                    "baseline_metric": "AUC", "baseline_value": 0.5,
                    "beats_baseline": False, "note": "missing cols",
                })
                continue
            auc = _classification_auc(y_true[h.name], y_pred[cols], h.classes or ())
            rows.append({
                "concept": h.name, "kind": h.kind, "metric": "AUC",
                "value": auc, "n": int(y_true[h.name].notna().sum()),
                "baseline_metric": "AUC", "baseline_value": 0.5,
                "beats_baseline": bool(np.isfinite(auc) and auc > 0.5),
                "note": "",
            })
        else:
            if h.name not in y_pred.columns or h.name not in y_true.columns:
                rows.append({
                    "concept": h.name, "kind": h.kind, "metric": "RMSE",
                    "value": float("nan"), "n": 0,
                    "baseline_metric": "RMSE", "baseline_value": float("nan"),
                    "beats_baseline": False, "note": "missing cols",
                })
                continue
            rmse = _regression_rmse(y_true[h.name], y_pred[h.name])
            base_src = (y_train[h.name] if (y_train is not None and h.name in y_train.columns)
                        else y_true[h.name])
            base = float(base_src.dropna().std(ddof=0))
            rows.append({
                "concept": h.name, "kind": h.kind, "metric": "RMSE",
                "value": rmse, "n": int(y_true[h.name].notna().sum()),
                "baseline_metric": "RMSE (σ)", "baseline_value": base,
                "beats_baseline": bool(np.isfinite(rmse) and np.isfinite(base) and rmse < base),
                "note": "",
            })
    return pd.DataFrame(rows)


def reliability_figure(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    heads: list[HeadSpec],
    path: str | Path,
) -> Path:
    """Reliability curves (binary heads) + parity scatter (regression heads)."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(heads)
    ncols = 3
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, h in zip(axes, heads):
        if h.kind == "classification" and h.classes and len(h.classes) == 2:
            col = f"{h.name}__{h.classes[1]}"
            if col not in y_pred.columns or h.name not in y_true.columns:
                ax.set_title(f"{h.name}: n/a"); ax.set_axis_off(); continue
            p = pd.Series(y_pred[col].to_numpy(float))
            y = (y_true[h.name].astype(str) == h.classes[1]).astype(float).reset_index(drop=True)
            bins = pd.cut(p, np.linspace(0, 1, 11))
            binned = pd.DataFrame({"p": p.values, "y": y.values, "b": bins}).groupby("b", observed=True)
            ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.4)
            ax.plot(binned["p"].mean(), binned["y"].mean(), "o-")
            ax.set_title(h.name); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("predicted prob"); ax.set_ylabel("empirical rate")
        elif h.kind == "regression":
            if h.name not in y_pred.columns or h.name not in y_true.columns:
                ax.set_title(f"{h.name}: n/a"); ax.set_axis_off(); continue
            yt = y_true[h.name].to_numpy(float); yp = y_pred[h.name].to_numpy(float)
            ax.scatter(yt, yp, s=6, alpha=0.4)
            lo = float(np.nanmin([yt, yp])); hi = float(np.nanmax([yt, yp]))
            ax.plot([lo, hi], [lo, hi], "--", color="k", alpha=0.4)
            ax.set_title(h.name); ax.set_xlabel("true"); ax.set_ylabel("pred")
        else:
            ax.set_title(f"{h.name} ({h.n_classes}-way)"); ax.set_axis_off()
    for ax in axes[len(heads):]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
