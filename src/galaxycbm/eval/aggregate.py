"""Aggregate every stage's artefacts into a single `results/metrics.json`.

Missing artefacts are recorded as `null` — Stage 1 is often skipped on
Intel Macs, so the aggregator must never crash when a file isn't there.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_ROOT = Path("results")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_records(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return pd.read_csv(path).to_dict(orient="records")


def _stage(name: str, path: str) -> dict[str, Any]:
    p = Path(path)
    return {"name": name, "run_json": _read_json(p / "run.json")}


def aggregate_metrics(root: Path = RESULTS_ROOT) -> dict[str, Any]:
    root = Path(root)
    return {
        "generated_from": str(root.resolve()),
        "git_sha": _git_sha(),
        "stages": {
            "scaffold":    _stage("scaffold",    str(root / "scaffold")),
            "concepts":    _stage("concepts",    str(root / "concepts")) | {
                "fidelity": _read_csv_records(root / "concepts" / "fidelity.csv"),
            },
            "symbolic":    _stage("symbolic",    str(root / "symbolic")) | {
                "metrics": _read_json(root / "symbolic" / "metrics.json"),
                "rules":   _read_csv_records(root / "symbolic" / "rule_table.csv"),
            },
            "uncertainty": _stage("uncertainty", str(root / "uncertainty")) | {
                "summary":            _read_json(root / "uncertainty" / "metrics.json"),
                "per_class_coverage": _read_csv_records(root / "uncertainty" / "per_class_coverage.csv"),
                "selective_curve":    _read_csv_records(root / "uncertainty" / "selective_curve.csv"),
            },
            "baselines":   _stage("baselines",   str(root / "baselines")) | {
                "rows":     _read_csv_records(root / "tables" / "comparison.csv"),
                "fidelity": _read_csv_records(root / "baselines" / "fidelity.csv"),
            },
            "robustness":  _stage("robustness",  str(root / "robustness")) | {
                "table": _read_csv_records(root / "tables" / "robustness.csv"),
            },
            "revision":    _stage("revision",    str(root / "revision")) | {
                "mondrian_summary":  _read_json(root / "revision" / "mondrian_summary.json"),
                "mondrian_per_class": _read_csv_records(root / "revision" / "mondrian_per_class.csv"),
                "compact":           _read_json(root / "revision" / "compact_comparison.json"),
                "compact_rules":     _read_csv_records(root / "revision" / "compact_rules.csv"),
            },
            "data.concepts": _stage("data.concepts", str(root / "data.concepts")),
            "data.labels":   _stage("data.labels",   str(root / "data.labels")),
        },
    }


def resolve_path(obj: Any, dotted: str) -> Any:
    """Follow a dotted path through nested dicts/lists.

    Integer segments index lists; anything else is a dict key. Returns
    `_MISSING` if the path can't be resolved (distinguishable from a
    legitimately-stored null).
    """
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return _MISSING
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        elif isinstance(cur, dict):
            if part not in cur:
                return _MISSING
            cur = cur[part]
        else:
            return _MISSING
    return cur


class _Missing:
    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _Missing()


def path_exists_and_is_populated(obj: Any, dotted: str) -> bool:
    v = resolve_path(obj, dotted)
    if v is _MISSING or v is None:
        return False
    if isinstance(v, (list, dict, str)) and len(v) == 0:
        return False
    return True
