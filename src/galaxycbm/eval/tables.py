"""Table renderers for the paper: CSV pass-through + booktabs-style LaTeX.

We avoid pandas.to_latex (drags in styling flags that vary across pandas
releases) and emit `tabular` explicitly — stable, and the same code renders
every stage's table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(v: object, floatfmt: str) -> str:
    if pd.isna(v):
        return "--"
    if isinstance(v, float):
        return f"{v:{floatfmt}}"
    s = str(v)
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def dataframe_to_latex(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    floatfmt: str = ".3f",
) -> str:
    if df.empty:
        return "% (empty)"
    align = "l" + "r" * (df.shape[1] - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(_fmt(c, floatfmt) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(_fmt(v, floatfmt) for v in row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def write_table(
    df: pd.DataFrame,
    out_csv: str | Path,
    out_tex: str | Path,
    *,
    caption: str,
    label: str,
    floatfmt: str = ".3f",
) -> tuple[Path, Path]:
    out_csv = Path(out_csv); out_tex = Path(out_tex)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_tex.write_text(
        dataframe_to_latex(df, caption=caption, label=label, floatfmt=floatfmt),
        encoding="utf-8",
    )
    return out_csv, out_tex
