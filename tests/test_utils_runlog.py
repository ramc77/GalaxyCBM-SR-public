"""Run-log tests: writes results/<stage>/run.json with SHA, versions, seed."""

from __future__ import annotations

import json

from galaxycbm.utils.runlog import build_run_record, write_run_json


def test_build_run_record_captures_expected_fields():
    rec = build_run_record("smoke", seed=42, config={"alpha": 0.1})
    d = rec.to_dict()
    for key in (
        "stage",
        "seed",
        "git_sha",
        "git_dirty",
        "python",
        "platform",
        "galaxycbm_version",
        "packages",
        "config_hash",
        "config",
    ):
        assert key in d
    assert d["stage"] == "smoke"
    assert d["seed"] == 42
    assert d["config"] == {"alpha": 0.1}
    assert isinstance(d["packages"], dict)
    assert isinstance(d["config_hash"], str) and len(d["config_hash"]) == 64


def test_config_hash_is_deterministic_across_key_order():
    a = build_run_record("s", config={"x": 1, "y": 2}).config_hash
    b = build_run_record("s", config={"y": 2, "x": 1}).config_hash
    assert a == b


def test_write_run_json_creates_stage_dir_and_writes_atomically(tmp_path):
    path = write_run_json(
        "stage1",
        results_root=tmp_path,
        seed=7,
        config={"backbone": "convnext_nano"},
        extra={"note": "smoke"},
    )
    assert path == tmp_path / "stage1" / "run.json"
    payload = json.loads(path.read_text())
    assert payload["stage"] == "stage1"
    assert payload["seed"] == 7
    assert payload["extra"] == {"note": "smoke"}
    # File must not be zero-length (atomic write completed).
    assert path.stat().st_size > 0
