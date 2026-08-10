"""Structured run logging.

Every stage should call :func:`write_run_json` at the top of its script so a
`results/<stage>/run.json` records the git SHA, config hash, package
versions, and seed. Reviewers can then reproduce any headline number in the
paper from a single JSON.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from galaxycbm import __version__ as _pkg_version
from galaxycbm.utils.io import ensure_dir, write_json

# Third-party libs whose version we always want in the run log. Missing ones
# are recorded as null rather than raising — the log should never take out a
# real training run.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "zoobot",
    "statmorph",
    "astropy",
    "photutils",
    "scikit-image",
    "pysr",
    "juliacall",
    "mapie",
    "scikit-learn",
    "xgboost",   # only present when installed with --extra baselines
    "shap",      # only present when installed with --extra baselines
    "pandas",
    "numpy",
    "pyarrow",
    "matplotlib",
    "hydra-core",
    "omegaconf",
)


@dataclass
class RunRecord:
    stage: str
    seed: int | None
    git_sha: str | None
    git_dirty: bool
    python: str
    platform: str
    galaxycbm_version: str
    packages: dict[str, str | None]
    config_hash: str | None = None
    config: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_sha_and_dirty() -> tuple[str | None, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sha, False
    return sha, bool(status.strip())


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _hash_config(config: Any) -> str:
    payload = json.dumps(_to_plain(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_plain(obj: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, DictConfig | ListConfig):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


def build_run_record(
    stage: str,
    *,
    seed: int | None = None,
    config: Any = None,
    extra: dict[str, Any] | None = None,
) -> RunRecord:
    sha, dirty = _git_sha_and_dirty()
    plain_cfg = _to_plain(config) if config is not None else None
    return RunRecord(
        stage=stage,
        seed=seed,
        git_sha=sha,
        git_dirty=dirty,
        python=sys.version.split()[0],
        platform=platform.platform(),
        galaxycbm_version=_pkg_version,
        packages=_package_versions(),
        config_hash=_hash_config(plain_cfg) if plain_cfg is not None else None,
        config=plain_cfg if isinstance(plain_cfg, dict) else None,
        extra=extra or {},
    )


def write_run_json(
    stage: str,
    results_root: str | Path = "results",
    *,
    seed: int | None = None,
    config: Any = None,
    extra: dict[str, Any] | None = None,
    filename: str = "run.json",
) -> Path:
    """Write ``results/<stage>/<filename>`` atomically. Returns the path."""
    stage_dir = ensure_dir(Path(results_root) / stage)
    record = build_run_record(stage, seed=seed, config=config, extra=extra)
    return write_json(stage_dir / filename, record.to_dict())
