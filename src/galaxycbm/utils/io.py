"""Safe IO helpers.

Every write goes through a tempfile + atomic rename so a killed process can
never leave a half-written parquet / json / png in the results tree.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """mkdir -p with return of the resolved Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    p = Path(path)
    _atomic_write(p, data)
    return p


def atomic_write_text(path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def _json_default(o: Any) -> Any:
    """Coerce common non-JSON types (OmegaConf, numpy, sets, Paths) to plain values."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(o, (DictConfig, ListConfig)):
            return OmegaConf.to_container(o, resolve=True)
    except ImportError:
        pass
    if hasattr(o, "tolist"):        # numpy scalars/arrays
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def write_json(
    path: str | os.PathLike[str],
    obj: Mapping[str, Any] | list[Any],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    return atomic_write_text(
        path,
        json.dumps(obj, indent=indent, sort_keys=sort_keys, default=_json_default) + "\n",
    )


def read_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping at {path!s}, got {type(loaded).__name__}.")
    return loaded
