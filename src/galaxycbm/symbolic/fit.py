"""One-vs-rest PySR symbolic classifier + stratified k-fold expression selection.

For each Hubble class c we fit ONE PySRRegressor on the whole training pool
against the binary indicator y = 1[hubble_type == c]. PySR emits its Pareto
frontier of expressions (complexity vs training loss). We then score every
equation on stratified k-fold *holdouts* to pick the equation that generalises
— cheaper than k separate PySR fits and still a valid CV selection.

At inference, argmax over per-class scores gives the predicted Hubble type
(mirrors the SR star/galaxy/quasar-separation formulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ClassRule:
    hubble_class: str
    equation_str: str        # SymPy-format equation (bare identifiers)
    latex: str
    complexity: int
    pysr_score: float        # PySR's own score (bigger = better trade-off)
    cv_accuracy: float       # mean stratified k-fold holdout accuracy


@dataclass
class SymbolicFitResult:
    rules: list[ClassRule]
    per_class_pareto: dict[str, pd.DataFrame] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PySR wiring
# ---------------------------------------------------------------------------


def _pysr_kwargs(symbolic_cfg) -> dict:
    p = symbolic_cfg.pysr
    return {
        "niterations": int(p.niterations),
        "populations": int(p.populations),
        "population_size": int(p.population_size),
        "maxsize": int(p.maxsize),
        "parsimony": float(p.parsimony),
        "binary_operators": list(p.binary_operators),
        "unary_operators": list(p.unary_operators),
        "constraints": {op: int(c) for op, c in dict(p.constraints).items()},
        "model_selection": str(p.model_selection),
        "elementwise_loss": str(p.loss),
        "turbo": bool(p.turbo),
        "progress": False,
        "verbosity": 0,
        "deterministic": True,
        "parallelism": "serial",   # required by PySR when deterministic=True
        "temp_equation_file": True,
    }


def _fit_regressor(X: pd.DataFrame, y: np.ndarray, *, seed: int, kwargs: dict):
    from pysr import PySRRegressor

    kw = dict(kwargs)
    kw["random_state"] = int(seed)
    est = PySRRegressor(**kw)
    est.fit(X.values, y, variable_names=list(X.columns))
    return est


def _pareto_frame(est) -> pd.DataFrame:
    """Copy the fitted PySR Pareto frontier as a DataFrame."""
    return est.equations_.copy()


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------


def _lambdify(expr) -> tuple[list[str], object]:
    import sympy

    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    fn = sympy.lambdify(syms, expr, modules=["numpy"])
    return [str(s.name) for s in syms], fn


def evaluate_expression(expr, X: pd.DataFrame) -> np.ndarray:
    """Evaluate a SymPy expression on a DataFrame. Missing symbols → error."""
    used, fn = _lambdify(expr)
    if not used:
        return np.full(len(X), float(fn()))
    vals = [X[name].to_numpy(float) for name in used]
    out = fn(*vals)
    out = np.asarray(out, dtype=float)
    if out.ndim == 0:
        out = np.full(len(X), float(out))
    return out


# ---------------------------------------------------------------------------
# CV expression selection
# ---------------------------------------------------------------------------


def _cv_accuracy(expr, X: pd.DataFrame, y_bin: np.ndarray, cv_splits) -> float:
    accs: list[float] = []
    for _, va in cv_splits:
        try:
            s = evaluate_expression(expr, X.iloc[va].reset_index(drop=True))
            accs.append(float(np.mean((s >= 0.5).astype(int) == y_bin[va])))
        except Exception:
            accs.append(0.0)
    return float(np.mean(accs)) if accs else float("nan")


def _cache_key(X_train: pd.DataFrame, y_bin: np.ndarray, kwargs: dict, seed: int) -> str:
    """Stable hash over training-set + PySR knobs. Cached fit is reused iff
    every input to the PySR run matches — same features, same labels,
    same operators, same seed.
    """
    import hashlib
    import json

    h = hashlib.sha256()
    h.update(str(list(X_train.columns)).encode())
    h.update(np.ascontiguousarray(X_train.values, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y_bin, dtype=np.int8).tobytes())
    h.update(json.dumps(kwargs, sort_keys=True, default=str).encode())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]


def fit_symbolic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    symbolic_cfg,
    *,
    cache_dir: str | Path = "results/symbolic/fit_cache",
) -> SymbolicFitResult:
    """Train PySR per Hubble class, then CV-select the winning expression.

    Per-class resume: each class's PySR fit + winning rule is cached under
    ``cache_dir/<class>__<hash>.json``. Re-running with the same inputs
    skips the fit entirely. Ctrl-C mid-loop is safe — completed classes
    survive; only the interrupted class re-runs.
    """
    import json as _json

    from sklearn.model_selection import StratifiedKFold

    kwargs = _pysr_kwargs(symbolic_cfg)
    classes = sorted(y_train.dropna().astype(str).unique().tolist())
    n_splits = int(symbolic_cfg.training.cv.n_splits)
    seed = int(symbolic_cfg.seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    rules: list[ClassRule] = []
    per_class_pareto: dict[str, pd.DataFrame] = {}

    for c in classes:
        y_bin = (y_train.astype(str) == c).astype(int).to_numpy()
        key = _cache_key(X_train, y_bin, kwargs, seed)
        cache_json = cache_dir / f"{c.replace('/', '_')}__{key}.json"
        cache_pareto = cache_dir / f"{c.replace('/', '_')}__{key}__pareto.parquet"

        if cache_json.exists() and cache_pareto.exists():
            print(f"[symbolic] class {c!r}: cached fit ({cache_json.name})")
            payload = _json.loads(cache_json.read_text())
            rules.append(ClassRule(
                hubble_class=payload["hubble_class"],
                equation_str=payload["equation_str"],
                latex=payload["latex"],
                complexity=int(payload["complexity"]),
                pysr_score=float(payload["pysr_score"]),
                cv_accuracy=float(payload["cv_accuracy"]),
            ))
            per_class_pareto[c] = pd.read_parquet(cache_pareto)
            continue

        print(f"[symbolic] class {c!r}: fitting …")
        est = _fit_regressor(X_train, y_bin, seed=seed, kwargs=kwargs)
        pareto = _pareto_frame(est)
        cv_splits = list(skf.split(X_train.values, y_bin))
        cv_accs = [
            _cv_accuracy(row["sympy_format"], X_train, y_bin, cv_splits)
            for _, row in pareto.iterrows()
        ]
        pareto = pareto.assign(cv_accuracy=cv_accs)
        winner = pareto.sort_values(
            ["cv_accuracy", "complexity"], ascending=[False, True]
        ).iloc[0]

        import sympy as _sp
        latex_str = str(winner["latex"]) if "latex" in winner.index else _sp.latex(winner["sympy_format"])
        rule = ClassRule(
            hubble_class=c,
            equation_str=str(winner["equation"]),
            latex=latex_str,
            complexity=int(winner["complexity"]),
            pysr_score=float(winner["score"]),
            cv_accuracy=float(winner["cv_accuracy"]),
        )
        # Persist immediately so a mid-loop Ctrl-C keeps completed classes.
        cache_json.write_text(_json.dumps({
            "hubble_class": rule.hubble_class,
            "equation_str": rule.equation_str,
            "latex": rule.latex,
            "complexity": rule.complexity,
            "pysr_score": rule.pysr_score,
            "cv_accuracy": rule.cv_accuracy,
        }, indent=2) + "\n", encoding="utf-8")
        # Pareto carries live Python objects PySR attaches for in-process use
        # (sympy_format = SymPy expr, lambda_format = a compiled callable) —
        # neither is Arrow-serialisable. Keep only plain columns for the
        # cache; stringify sympy_format so a human can still read it.
        pareto_serialisable = pareto.copy()
        if "sympy_format" in pareto_serialisable.columns:
            pareto_serialisable["sympy_format"] = pareto_serialisable["sympy_format"].astype(str)
        drop_cols = [c for c in ("lambda_format",) if c in pareto_serialisable.columns]
        pareto_serialisable = pareto_serialisable.drop(columns=drop_cols)
        pareto_serialisable.to_parquet(cache_pareto, index=False)

        rules.append(rule)
        per_class_pareto[c] = pareto

    return SymbolicFitResult(
        rules=rules,
        per_class_pareto=per_class_pareto,
        classes=classes,
        feature_columns=list(X_train.columns),
    )


# ---------------------------------------------------------------------------
# Exports: LaTeX, plain text, callable Python
# ---------------------------------------------------------------------------


def export_latex(result: SymbolicFitResult, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% GalaxyCBM-SR — Stage-2 symbolic decision head (one-vs-rest)."]
    for r in result.rules:
        lines.append(
            f"\\[ \\text{{score}}_{{\\text{{{r.hubble_class}}}}} = {r.latex} \\]"
            f"  %% complexity={r.complexity}, CV acc={r.cv_accuracy:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_plain(result: SymbolicFitResult, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# GalaxyCBM-SR — Stage-2 rules (predict class = argmax over scores).",
             f"# classes = {result.classes}",
             f"# feature_columns = {result.feature_columns}",
             ""]
    for r in result.rules:
        lines.append(
            f"score[{r.hubble_class}] = {r.equation_str}    "
            f"# complexity={r.complexity}, CV_acc={r.cv_accuracy:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_callable(result: SymbolicFitResult, path: str | Path) -> Path:
    """Emit a self-contained Python module with score(X) and predict(X)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    exprs = {r.hubble_class: r.equation_str for r in result.rules}
    header = f'''"""Auto-generated symbolic classifier — do not edit by hand.

Regenerated by scripts/train_symbolic.py.
Consumers pass a pandas DataFrame with the columns listed in FEATURE_COLUMNS
(the sklearn-safe predicted-concept column names) and receive either a
per-class score DataFrame or an array of argmax labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sympy

CLASSES = {result.classes!r}
FEATURE_COLUMNS = {result.feature_columns!r}
EXPRESSIONS = {exprs!r}

_LAMBDAS: dict | None = None


def _compile() -> dict:
    global _LAMBDAS
    if _LAMBDAS is not None:
        return _LAMBDAS
    ns = {{name: sympy.Symbol(name) for name in FEATURE_COLUMNS}}
    _LAMBDAS = {{}}
    for cls, expr_str in EXPRESSIONS.items():
        expr = sympy.sympify(expr_str, locals=ns)
        used = sorted(expr.free_symbols, key=lambda s: s.name)
        fn = sympy.lambdify(used, expr, modules=["numpy"])
        _LAMBDAS[cls] = ([str(s.name) for s in used], fn)
    return _LAMBDAS


def score(X: pd.DataFrame) -> pd.DataFrame:
    lambdas = _compile()
    out = {{}}
    for cls, (used, fn) in lambdas.items():
        if not used:
            out[cls] = np.full(len(X), float(fn()))
        else:
            vals = [X[name].to_numpy(float) for name in used]
            arr = np.asarray(fn(*vals), dtype=float)
            if arr.ndim == 0:
                arr = np.full(len(X), float(arr))
            out[cls] = arr
    return pd.DataFrame(out, index=X.index)


def predict(X: pd.DataFrame) -> np.ndarray:
    s = score(X)
    return s.idxmax(axis=1).to_numpy()
'''
    path.write_text(header, encoding="utf-8")
    return path


def rules_dataframe(result: SymbolicFitResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "hubble_class": r.hubble_class,
        "expression": r.equation_str,
        "latex": r.latex,
        "complexity": r.complexity,
        "cv_accuracy": r.cv_accuracy,
        "pysr_score": r.pysr_score,
    } for r in result.rules])
