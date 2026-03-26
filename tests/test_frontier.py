"""Tests for fisher_simplex.frontier module."""

from __future__ import annotations

import numpy as np

from fisher_simplex.core import h3, q_delta
from fisher_simplex.frontier import (
    frontier8_batch,
    frontier8_coordinates,
    frontier8_residual,
)
from tests.conftest import random_simplex

# ---------------------------------------------------------------------------
# frontier8_coordinates — shape and consistency
# ---------------------------------------------------------------------------


class TestFrontier8Coordinates:
    """Shape, forced-pair consistency, and orthogonality tests."""

    def test_shape_single(self) -> None:
        """Single input returns shape (4,)."""
        s = np.array([0.2, 0.3, 0.1, 0.15, 0.25])
        result = frontier8_coordinates(s)
        assert result.shape == (4,)

    def test_shape_batch(self, rng: np.random.Generator) -> None:
        """Batch input returns shape (M, 4)."""
        samples = random_simplex(5, 20, rng)
        result = frontier8_coordinates(samples)
        assert result.shape == (20, 4)

    def test_first_two_columns_match_forced_pair(
        self, rng: np.random.Generator
    ) -> None:
        """First two columns of output match q_delta and h3 exactly."""
        samples = random_simplex(5, 50, rng)
        coords = frontier8_coordinates(samples)
        np.testing.assert_allclose(coords[:, 0], q_delta(samples), atol=1e-14)
        np.testing.assert_allclose(coords[:, 1], h3(samples), atol=1e-14)

    def test_orthogonality_e8_vs_q_delta(self) -> None:
        """E[E8_k * Q_delta] ~ 0 under MC Dirichlet(1), 10000 samples."""
        rng = np.random.default_rng(seed=123)
        samples = rng.dirichlet(np.ones(5), size=10000)
        coords = frontier8_coordinates(samples)
        qd = coords[:, 0]
        e8_1 = coords[:, 2]
        e8_2 = coords[:, 3]

        np.testing.assert_allclose(np.mean(e8_1 * qd), 0.0, atol=0.05)
        np.testing.assert_allclose(np.mean(e8_2 * qd), 0.0, atol=0.05)

    def test_orthogonality_e8_vs_h3(self) -> None:
        """E[E8_k * H_3] ~ 0 under MC Dirichlet(1), 10000 samples."""
        rng = np.random.default_rng(seed=456)
        samples = rng.dirichlet(np.ones(5), size=10000)
        coords = frontier8_coordinates(samples)
        h = coords[:, 1]
        e8_1 = coords[:, 2]
        e8_2 = coords[:, 3]

        np.testing.assert_allclose(np.mean(e8_1 * h), 0.0, atol=0.05)
        np.testing.assert_allclose(np.mean(e8_2 * h), 0.0, atol=0.05)


# ---------------------------------------------------------------------------
# frontier8_batch
# ---------------------------------------------------------------------------


class TestFrontier8Batch:
    """Tests for the batch convenience wrapper."""

    def test_single_becomes_2d(self) -> None:
        """Single input returns shape (1, 4)."""
        s = np.array([0.2, 0.3, 0.1, 0.15, 0.25])
        result = frontier8_batch(s)
        assert result.shape == (1, 4)

    def test_matches_coordinates(self, rng: np.random.Generator) -> None:
        """frontier8_batch matches frontier8_coordinates for batch input."""
        samples = random_simplex(5, 30, rng)
        coords = frontier8_coordinates(samples)
        batch = frontier8_batch(samples)
        np.testing.assert_allclose(batch, coords, atol=1e-14)


# ---------------------------------------------------------------------------
# frontier8_residual — R3 and diagnostics
# ---------------------------------------------------------------------------


class TestFrontier8Residual:
    """Tests for frontier8_residual diagnostics."""

    def test_r3_degree8_improvement(self) -> None:
        """R3: R^2_frontier > R^2_forced for a degree-8 target.

        Under Dirichlet(1) at N=5, raw p_4 is >99% explained by the
        forced model (power sums are highly correlated near the center).
        To isolate the genuine degree-8 signal, we extract the part of
        p_4 orthogonal to the forced block — a degree-4 polynomial that
        is NOT in span{Q_delta, H_3} by construction.
        """
        rng = np.random.default_rng(seed=789)
        N = 5
        samples = rng.dirichlet(np.ones(N), size=500)

        # Extract E8 component of p_4: residual after removing
        # forced-block projection. This is a degree-4 polynomial
        # (p_4 minus a linear combination of {1, Q_delta, H_3}).
        p4 = np.sum(samples**4, axis=-1)
        qd = q_delta(samples)
        h = h3(samples)
        X_forced = np.column_stack([np.ones(len(samples)), qd, h])
        beta = np.linalg.lstsq(X_forced, p4, rcond=None)[0]
        target = p4 - X_forced @ beta

        result = frontier8_residual(samples, target)

        assert result["r_squared_frontier"] > result["r_squared_forced"] + 0.01
        assert result["needs_frontier"] is True

    def test_forced_sufficient_target(self) -> None:
        """No frontier needed when target is in span{1, Q_delta, H_3}."""
        rng = np.random.default_rng(seed=101)
        samples = rng.dirichlet(np.ones(5), size=500)
        target = q_delta(samples)

        result = frontier8_residual(samples, target)

        assert result["r_squared_forced"] > 0.99
        assert result["frontier_improvement"] < 0.01
        assert result["needs_frontier"] is False

    def test_returns_expected_keys(self, rng: np.random.Generator) -> None:
        """Return dict has exactly the specified keys."""
        samples = random_simplex(5, 50, rng)
        target = np.sum(samples**4, axis=-1)
        result = frontier8_residual(samples, target)

        assert set(result.keys()) == {
            "r_squared_forced",
            "r_squared_frontier",
            "frontier_improvement",
            "needs_frontier",
        }

    def test_quadratic_model(self) -> None:
        """model='quadratic' runs without error and returns valid results."""
        rng = np.random.default_rng(seed=202)
        samples = rng.dirichlet(np.ones(5), size=500)
        target = np.sum(samples**4, axis=-1)

        result = frontier8_residual(samples, target, model="quadratic")

        assert 0.0 <= result["r_squared_forced"] <= 1.0
        assert 0.0 <= result["r_squared_frontier"] <= 1.0
