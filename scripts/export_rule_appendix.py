"""Typeset the Stage-2 rule set for the manuscript appendix.

The manuscript prints the deployed equations in full. PySR's own LaTeX renders
each feature name as a mangled sub/superscript pair and is unreadable on the
page, so this script re-derives the LaTeX from the *deployed* expression
strings -- the same strings `score_expressions` sympifies at inference -- after
substituting a short, declared symbol for every concept.

Nothing is simplified or rounded beyond SymPy's own canonical ordering, so the
printed equation and the evaluated equation are the same object.

Emits:
    paper/tables/rules_full.tex        the seven equations, aligned
    paper/tables/rule_symbols.tex      the symbol dictionary
    paper/tables/rule_tree_map.tex     concepts used, by Galaxy Zoo tree node
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import sympy

# ---------------------------------------------------------------------------
# Concept -> printed symbol. Every feature reachable by a rule must appear.
# ---------------------------------------------------------------------------
SYMBOLS: dict[str, str] = {
    "smooth_or_featured__smooth":       r"v_{\mathrm{sm}}",
    "smooth_or_featured__featured_or_disk": r"v_{\mathrm{ft}}",
    "smooth_or_featured__artifact":     r"v_{\mathrm{art}}",
    "disk_edge_on__no":                 r"v_{\overline{\mathrm{edge}}}",
    "bar__strong":                      r"v_{\mathrm{bar}}",
    "bar__weak":                        r"v_{\mathrm{bar\,w}}",
    "bar__no":                          r"v_{\overline{\mathrm{bar}}}",
    "has_spiral_arms__no":              r"v_{\overline{\mathrm{arm}}}",
    "bulge_size__dominant":             r"v_{\mathrm{B4}}",
    "bulge_size__large":                r"v_{\mathrm{B3}}",
    "bulge_size__moderate":             r"v_{\mathrm{B2}}",
    "bulge_size__small":                r"v_{\mathrm{B1}}",
    "bulge_size__none":                 r"v_{\mathrm{B0}}",
    "how_rounded__round":               r"v_{\mathrm{rnd}}",
    "how_rounded__in_between":          r"v_{\mathrm{inb}}",
    "how_rounded__cigar_shaped":        r"v_{\mathrm{cig}}",
    "edge_on_bulge__boxy":              r"v_{\mathrm{boxy}}",
    "edge_on_bulge__rounded":           r"v_{\mathrm{ebr}}",
    "edge_on_bulge__none":              r"v_{\mathrm{ebn}}",
    "spiral_winding__tight":            r"v_{\mathrm{W_t}}",
    "spiral_winding__medium":           r"v_{\mathrm{W_m}}",
    "spiral_winding__loose":            r"v_{\mathrm{W_l}}",
    "spiral_arm_count__1":              r"v_{\mathrm{N1}}",
    "spiral_arm_count__2":              r"v_{\mathrm{N2}}",
    "spiral_arm_count__3":              r"v_{\mathrm{N3}}",
    "spiral_arm_count__4":              r"v_{\mathrm{N4}}",
    "spiral_arm_count__more_than_4":    r"v_{\mathrm{N5+}}",
    "spiral_arm_count__cant_tell":      r"v_{\mathrm{N?}}",
    "merging__none":                    r"v_{\mathrm{M0}}",
    "merging__minor_disturbance":       r"v_{\mathrm{M1}}",
    "merging__major_disturbance":       r"v_{\mathrm{M2}}",
    "merging__merger":                  r"v_{\mathrm{M3}}",
    "concentration":                    r"C",
    "asymmetry":                        r"A",
    "smoothness":                       r"S",
    "gini":                             r"G",
    "m20":                              r"M_{20}",
    "sersic_n":                         r"n",
    "r_eff_pixels":                     r"r_{\mathrm{e}}",
}

# Human-readable gloss, and the Galaxy Zoo tree node each answer belongs to.
GLOSS: dict[str, tuple[str, str]] = {
    "smooth_or_featured__smooth":       ("smooth (no visible structure)", "T1 smooth-or-featured"),
    "smooth_or_featured__featured_or_disk": ("featured or disc", "T1 smooth-or-featured"),
    "smooth_or_featured__artifact":     ("artefact / star", "T1 smooth-or-featured"),
    "disk_edge_on__no":                 ("disc not edge-on", "T2 edge-on"),
    "bar__strong":                      ("strong bar", "T3 bar"),
    "has_spiral_arms__no":              ("no spiral arms visible", "T4 spiral arms"),
    "bulge_size__dominant":             ("dominant bulge", "T5 bulge size"),
    "bulge_size__large":                ("large bulge", "T5 bulge size"),
    "bulge_size__moderate":             ("moderate bulge", "T5 bulge size"),
    "bulge_size__none":                 ("no bulge", "T5 bulge size"),
    "how_rounded__round":               ("round", "T6 roundedness"),
    "how_rounded__cigar_shaped":        ("cigar-shaped", "T6 roundedness"),
    "spiral_winding__tight":            ("tightly wound arms", "T8 arm winding"),
    "spiral_winding__medium":           ("medium winding", "T8 arm winding"),
    "spiral_winding__loose":            ("loosely wound arms", "T8 arm winding"),
    "spiral_arm_count__more_than_4":    ("more than four arms", "T9 arm count"),
    "merging__none":                    ("no merger signature", "T10 merger"),
    "merging__minor_disturbance":       ("minor disturbance", "T10 merger"),
    "merging__major_disturbance":       ("major disturbance", "T10 merger"),
    "merging__merger":                  ("ongoing merger", "T10 merger"),
    "smoothness":                       (r"\textsc{statmorph} clumpiness $S$", "structural"),
}

# Hubble sequence order for the printed tables (the CSV is alphabetical).
HUBBLE_ORDER = ["E", "S0", "Sa", "Sb", "Sc", "Sd", "Irr"]

TREE_ORDER = [
    "T1 smooth-or-featured", "T2 edge-on", "T3 bar", "T4 spiral arms",
    "T5 bulge size", "T6 roundedness", "T8 arm winding", "T9 arm count",
    "T10 merger", "structural",
]


HEADER = "% Auto-generated by scripts/export_rule_appendix.py -- do not edit.\n"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _deploy_expr(equation_str: str) -> sympy.Expr:
    """Sympify exactly as `score_expressions` does, then make it printable.

    PySR emits `square(x)` / `cube(x)`, which SymPy keeps as undefined
    functions and typesets as \operatorname{square}. Rewriting them as powers
    changes the printed form, not the value.
    """
    expr = sympy.sympify(equation_str)
    for name, power in (("square", 2), ("cube", 3)):
        fn = sympy.Function(name)
        expr = expr.replace(fn, lambda arg, _p=power: arg ** _p)
    return expr


def _round_floats(expr: sympy.Expr, sig: int = 6) -> sympy.Expr:
    """Display coefficients at `sig` significant figures.

    SymPy folds constants when it canonicalises (e.g. e^{a+c} -> e^{c} e^{a}),
    which can expose a coefficient at full double precision. The exact values
    are in the machine-readable release; the paper prints six figures.
    """
    return expr.xreplace({f: sympy.Float(f, sig) for f in expr.atoms(sympy.Float)})


def render(rule_table: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = pd.read_csv(rule_table)
    tbl["_order"] = tbl["hubble_class"].map(
        {c: i for i, c in enumerate(HUBBLE_ORDER)}).fillna(99)
    tbl = tbl.sort_values("_order").reset_index(drop=True)

    subs = {sympy.Symbol(k): sympy.Symbol(v) for k, v in SYMBOLS.items()}

    eq_lines: list[str] = []
    used_overall: dict[str, set[str]] = {}

    for _, row in tbl.iterrows():
        cls = str(row["hubble_class"])
        expr = _deploy_expr(str(row["expression"]))
        raw_names = sorted(s.name for s in expr.free_symbols)
        unknown = [n for n in raw_names if n not in SYMBOLS]
        if unknown:
            raise KeyError(f"no printed symbol declared for {unknown}")
        used_overall[cls] = set(raw_names)
        printed = _round_floats(expr.subs(subs))
        pretty = sympy.latex(printed, mul_symbol="dot", fold_short_frac=False)
        eq_lines.append(
            f"% {cls}: complexity={int(row['complexity'])}, "
            f"CV acc={float(row['cv_accuracy']):.3f}\n"
            + rf"g_{{\mathrm{{{cls}}}}} &= {pretty} \label{{eq:rule_{cls}}}"
        )

    # Each generated file carries its own complete environment. TeX's \input
    # cannot be used *inside* an alignment -- the end-of-file token breaks the
    # \\ lookahead and the next \noalign (\bottomrule) is then misplaced -- so
    # main.tex inputs these at top level instead.
    (out_dir / "rules_full.tex").write_text(
        HEADER
        + "{\\small\n\\begin{align}\n"
        # No trailing row break: align would otherwise end with an empty row.
        + " \\\\[8pt]\n".join(eq_lines)
        + "\n\\end{align}}\n",
        encoding="utf-8",
    )

    # ---- symbol dictionary, restricted to symbols the rules actually use ---
    used_any = sorted(set().union(*used_overall.values()))
    rows = []
    for name in used_any:
        gloss, node = GLOSS.get(name, ("", "structural"))
        rows.append((TREE_ORDER.index(node), node, SYMBOLS[name], _tex_escape(name), gloss))
    rows.sort()
    dict_lines = [
        HEADER.rstrip("\n"),
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Symbol & Concept variable & Meaning \\",
        r"\midrule",
    ]
    last_node = None
    for _, node, sym, raw, gloss in rows:
        if node != last_node:
            dict_lines.append(rf"\multicolumn{{3}}{{l}}{{\emph{{{node}}}}} \\")
            last_node = node
        dict_lines.append(rf"$\,\,{sym}$ & \code{{{raw}}} & {gloss} \\")
    dict_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "rule_symbols.tex").write_text("\n".join(dict_lines) + "\n", encoding="utf-8")

    # ---- Galaxy Zoo tree node coverage per class --------------------------
    classes = list(tbl["hubble_class"])
    map_lines = [
        HEADER.rstrip("\n"),
        r"\begin{tabular}{l" + "c" * len(classes) + "}",
        r"\toprule",
        "Galaxy Zoo task & " + " & ".join(classes) + r" \\",
        r"\midrule",
    ]
    for node in TREE_ORDER:
        members = {n for n, (_, nd) in GLOSS.items() if nd == node}
        if not members:
            continue
        marks = []
        for cls in classes:
            marks.append(r"$\bullet$" if used_overall[cls] & members else "")
        if not any(marks):
            continue
        map_lines.append(f"{node} & " + " & ".join(marks) + r" \\")
    map_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "rule_tree_map.tex").write_text("\n".join(map_lines) + "\n", encoding="utf-8")

    print(f"[appendix] wrote {out_dir/'rules_full.tex'}")
    print(f"[appendix] wrote {out_dir/'rule_symbols.tex'} ({len(rows)} symbols)")
    print(f"[appendix] wrote {out_dir/'rule_tree_map.tex'}")


if __name__ == "__main__":
    render(Path("results/symbolic/rule_table.csv"), Path("paper/tables"))
