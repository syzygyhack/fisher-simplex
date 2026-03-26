"""Shared test fixtures for fisher-simplex."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


def uniform_simplex(n: int) -> np.ndarray:
    """Return the uniform composition (1/n, ..., 1/n) on Delta^{n-1}."""
    return np.full(n, 1.0 / n)


def vertex_simplex(n: int, k: int = 0) -> np.ndarray:
    """Return vertex e_k on Delta^{n-1}."""
    v = np.zeros(n)
    v[k] = 1.0
    return v


def random_simplex(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Return (m, n) Dirichlet(1, ..., 1) samples on Delta^{n-1}."""
    return rng.dirichlet(np.ones(n), size=m)
