"""Tests for fisher_simplex.generators module — S4 and extras."""

from __future__ import annotations

import numpy as np
import pytest

from fisher_simplex.generators import (
    broken_stick,
    dirichlet_community,
    geometric_series,
    lognormal_community,
    market_shares,
    portfolio_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_GENERATORS = [
    ("dirichlet_community", dirichlet_community, {}),
    ("broken_stick", broken_stick, {}),
    ("lognormal_community", lognormal_community, {}),
    ("geometric_series", geometric_series, {}),
    ("market_shares", market_shares, {}),
    ("portfolio_weights", portfolio_weights, {}),
]

SHAPE_COMBOS = [(3, 100), (5, 50), (10, 200), (20, 10)]
_GEN_IDS = [g[0] for g in ALL_GENERATORS]


def _assert_valid_simplex(arr: np.ndarray, m: int, n: int) -> None:
    """Assert array is shape (m, n), nonneg, rows sum to 1."""
    assert arr.shape == (m, n), f"Expected shape ({m}, {n}), got {arr.shape}"
    assert np.all(arr >= 0), "Found negative entries"
    row_sums = arr.sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# S4: All generators return valid simplex arrays
# ---------------------------------------------------------------------------


class TestS4ValidSimplex:
    """S4: Every generator returns valid simplex arrays for multiple (n, m)."""

    @pytest.mark.parametrize("n, m", SHAPE_COMBOS)
    @pytest.mark.parametrize("name, gen, kwargs", ALL_GENERATORS, ids=_GEN_IDS)
    def test_valid_simplex(
        self,
        name: str,
        gen,
        kwargs: dict,
        n: int,
        m: int,
        rng: np.random.Generator,
    ) -> None:
        result = gen(n, m, rng=rng, **kwargs)
        _assert_valid_simplex(result, m, n)


# ---------------------------------------------------------------------------
# Reproducibility: same seed → same output
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same rng seed produces identical output for every generator."""

    @pytest.mark.parametrize("name, gen, kwargs", ALL_GENERATORS, ids=_GEN_IDS)
    def test_reproducible(self, name: str, gen, kwargs: dict) -> None:
        seed = 12345
        a = gen(5, 20, rng=np.random.default_rng(seed), **kwargs)
        b = gen(5, 20, rng=np.random.default_rng(seed), **kwargs)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Edge cases: n=2, m=1
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Minimum valid dimensions produce correct output."""

    @pytest.mark.parametrize("name, gen, kwargs", ALL_GENERATORS, ids=_GEN_IDS)
    def test_n2_m1(
        self,
        name: str,
        gen,
        kwargs: dict,
        rng: np.random.Generator,
    ) -> None:
        result = gen(2, 1, rng=rng, **kwargs)
        _assert_valid_simplex(result, 1, 2)


# ---------------------------------------------------------------------------
# No generator raises on standard inputs
# ---------------------------------------------------------------------------


class TestNoRaise:
    """All generators run without error on standard inputs."""

    @pytest.mark.parametrize("name, gen, kwargs", ALL_GENERATORS, ids=_GEN_IDS)
    def test_no_raise(
        self,
        name: str,
        gen,
        kwargs: dict,
        rng: np.random.Generator,
    ) -> None:
        gen(5, 50, rng=rng, **kwargs)


# ---------------------------------------------------------------------------
# market_shares — all concentration values
# ---------------------------------------------------------------------------


class TestMarketShares:
    """Test market_shares with all concentration settings."""

    @pytest.mark.parametrize("concentration", ["low", "medium", "high", "mixed"])
    def test_concentration(self, concentration: str, rng: np.random.Generator) -> None:
        result = market_shares(5, 50, concentration=concentration, rng=rng)
        _assert_valid_simplex(result, 50, 5)


# ---------------------------------------------------------------------------
# portfolio_weights — all style values
# ---------------------------------------------------------------------------


class TestPortfolioWeights:
    """Test portfolio_weights with all style settings."""

    @pytest.mark.parametrize("style", ["equal", "concentrated", "diversified", "mixed"])
    def test_style(self, style: str, rng: np.random.Generator) -> None:
        result = portfolio_weights(5, 50, style=style, rng=rng)
        _assert_valid_simplex(result, 50, 5)
