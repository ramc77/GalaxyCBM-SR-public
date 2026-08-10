"""Check that every claim listed in paper/claims.yaml has a value in metrics.json.

An "orphan claim" is a paper-cited number that doesn't trace to results/.
The gate is: `check_claims` returns a report; the driver fails if any claim
is missing (unless marked `optional: true` in the manifest).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from galaxycbm.eval.aggregate import path_exists_and_is_populated


@dataclass
class ClaimReport:
    total: int
    present: int
    missing: list[str]
    missing_optional: list[str]

    @property
    def ok(self) -> bool:
        return len(self.missing) == 0


def load_claims_manifest(path: str | Path) -> list[dict[str, Any]]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "claims" not in doc:
        raise ValueError(f"{path}: expected a top-level `claims:` list.")
    return list(doc["claims"])


def check_claims(metrics: dict, claims: list[dict[str, Any]]) -> ClaimReport:
    missing: list[str] = []
    missing_optional: list[str] = []
    present = 0
    for entry in claims:
        dotted = str(entry["path"])
        optional = bool(entry.get("optional", False))
        if path_exists_and_is_populated(metrics, dotted):
            present += 1
        elif optional:
            missing_optional.append(dotted)
        else:
            missing.append(dotted)
    return ClaimReport(total=len(claims), present=present,
                       missing=missing, missing_optional=missing_optional)
