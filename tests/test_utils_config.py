"""Config loader tests — reads real configs from disk and applies overrides."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig

from galaxycbm.utils.config import load_config


@pytest.mark.parametrize("name", ["data", "concepts", "model", "symbolic", "conformal"])
def test_load_each_config_by_name(name: str) -> None:
    cfg = load_config(name)
    assert isinstance(cfg, DictConfig)
    assert len(cfg) > 0


def test_overrides_are_merged_last(tmp_path):
    (tmp_path / "toy.yaml").write_text("a: 1\nb: {c: 2}\n")
    cfg = load_config("toy", config_dir=tmp_path, overrides={"b": {"c": 42}, "d": [1, 2]})
    assert cfg.a == 1
    assert cfg.b.c == 42
    assert list(cfg.d) == [1, 2]


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist", config_dir=tmp_path)
