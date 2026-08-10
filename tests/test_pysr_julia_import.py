"""PySR + Julia smoke test.

Marked ``julia`` and skipped by default because the first import triggers a
~300 MB Julia download via juliapkg. Run explicitly with:
    uv run pytest -m julia
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.julia
def test_pysr_imports_cleanly_and_reports_a_version():
    pysr = importlib.import_module("pysr")
    assert hasattr(pysr, "__version__")
    assert isinstance(pysr.__version__, str)
    # Touch the sklearn estimator — this is what Stage 2 will use.
    from pysr import PySRRegressor  # noqa: F401


@pytest.mark.julia
def test_juliacall_imports_a_working_julia():
    juliacall = importlib.import_module("juliacall")
    Main = juliacall.Main
    assert int(Main.seval("1 + 1")) == 2
