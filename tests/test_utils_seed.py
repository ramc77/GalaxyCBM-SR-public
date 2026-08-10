"""Seed determinism tests."""

from __future__ import annotations

import random

import numpy as np

from galaxycbm.utils.seed import seed_everything


def test_seed_everything_returns_the_seed():
    assert seed_everything(123) == 123


def test_seed_everything_makes_python_random_reproducible():
    seed_everything(7)
    a = [random.random() for _ in range(5)]
    seed_everything(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(7)
    a = np.random.rand(10)
    seed_everything(7)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)
