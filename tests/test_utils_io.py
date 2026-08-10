"""Safe-IO tests."""

from __future__ import annotations

import json

import pytest

from galaxycbm.utils.io import (
    atomic_write_bytes,
    atomic_write_text,
    ensure_dir,
    read_yaml,
    write_json,
)


def test_ensure_dir_is_idempotent(tmp_path):
    p = tmp_path / "a" / "b" / "c"
    assert not p.exists()
    ensure_dir(p)
    ensure_dir(p)
    assert p.is_dir()


def test_atomic_writes(tmp_path):
    text_path = atomic_write_text(tmp_path / "hello.txt", "hi")
    bytes_path = atomic_write_bytes(tmp_path / "hello.bin", b"\x00\x01")
    assert text_path.read_text() == "hi"
    assert bytes_path.read_bytes() == b"\x00\x01"


def test_write_json_sorts_keys_and_ends_with_newline(tmp_path):
    p = write_json(tmp_path / "run.json", {"b": 2, "a": 1})
    raw = p.read_text()
    assert raw.endswith("\n")
    assert list(json.loads(raw).keys()) == ["a", "b"]


def test_read_yaml_roundtrip(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("x: 1\ny: [a, b]\n")
    got = read_yaml(yaml_path)
    assert got == {"x": 1, "y": ["a", "b"]}


def test_read_yaml_rejects_scalar_top_level(tmp_path):
    (tmp_path / "scalar.yaml").write_text("42\n")
    with pytest.raises(TypeError):
        read_yaml(tmp_path / "scalar.yaml")
