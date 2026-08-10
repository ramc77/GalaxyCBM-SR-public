"""Config loader.

We keep the loader thin: read a YAML file from ``configs/`` and return an
:class:`omegaconf.DictConfig`. Stages that need Hydra composition or CLI
overrides can still invoke Hydra directly — the P0 loader is what tests and
one-off scripts use, and it stays offline / dependency-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _config_path(name_or_path: str, config_dir: Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    candidate = config_dir / f"{p.stem or p.name}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Config '{name_or_path}' not found — looked at {candidate}. "
            f"Available: {sorted(x.name for x in config_dir.glob('*.yaml'))}"
        )
    return candidate


def load_config(
    name: str,
    *,
    config_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Load a config by name (``'data'``) or explicit path.

    Overrides are merged last, so callers can inject values without editing
    the YAML on disk.
    """
    cfg_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR
    path = _config_path(name, cfg_dir)
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config at {path} did not parse as a mapping.")
    return cfg
