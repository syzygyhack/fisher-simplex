"""Tests for fisher_simplex.core module — E-series and extras."""

from __future__ import annotations

import numpy as np
import pytest

from fisher_simplex.core import (
    binary_fisher_angle,
    binary_overlap,
    effective_number_herfindahl,
    effective_number_shannon,
    fisher_lift,
    fisher_project,
    forced_coordinates,
    forced_pair,
    gini_simpson,
    h3,
    herfindahl,
    overlap_divergence,
    phi,
    psi,
    psi_overlap,
    q_delta,
    qh_ratio,
    shannon_entropy,
    simpson_index,
)
from tests.conftest import random_simplex, uniform_simplex, vertex_simplex

# ---------------------------------------------------------------------------
# E-series tests (spec section 4.1)
# ---------------------------------------------------------------------------


class TestESeries:
    """E1-E8 foundational invariant tests."""

    def test_e1_binary_overlap_identity(self, rng: np.random.Generator) -> None:
        """E1: Phi_2 = Psi_2 = 4*s1*s2 for 1000 random binary compositions."""
        samples = random_simplex(2, 1000, rng)
        phi_vals = phi(samples)
        psi_vals = psi_overlap(samples)
        expected = 4.0 * samples[:, 0] * samples[:, 1]
        np.testing.assert_allclose(phi_vals, expected, atol=1e-14)
        np.testing.assert_allclose(psi_vals, expected, atol=1e-14)

    def test_e2_ternary_divergence(self, rng: np.random.Generator) -> None:
        """E2: Phi_3 != Psi_3 for >95% of 1000 Dirichlet(1,1,1) samples."""
        samples = random_simplex(3, 1000, rng)
        d = overlap_divergence(samples)
        frac_nonzero = np.mean(np.abs(d) > 1e-10)
        assert frac_nonzero > 0.95

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_e3_q_delta_uniform_zero(self, n: int) -> None:
        """E3: Q_delta(uniform) = 0 for various N."""
        s = uniform_simplex(n)
        np.testing.assert_allclose(q_delta(s), 0.0, atol=1e-14)

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_e4_h3_uniform(self, n: int) -> None:
        """E4: H_3(uniform) = 2/(N^2*(N+2)) for various N."""
        s = uniform_simplex(n)
        expected = 2.0 / (n**2 * (n + 2))
        np.testing.assert_allclose(h3(s), expected, rtol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_e5_q_delta_vertex(self, n: int) -> None:
        """E5: Q_delta(vertex) = 1 - 1/N for various N."""
        s = vertex_simplex(n)
        np.testing.assert_allclose(q_delta(s), 1.0 - 1.0 / n, atol=1e-14)

    def test_e6_fisher_roundtrip(self, rng: np.random.Generator) -> None:
        """E6: s -> sqrt(s) -> (sqrt(s))^2 recovers s."""
        samples = random_simplex(5, 100, rng)
        recovered = fisher_project(fisher_lift(samples))
        np.testing.assert_allclose(recovered, samples, atol=1e-14)

    def test_e7_fisher_lift_unit_norm(self, rng: np.random.Generator) -> None:
        """E7: ||fisher_lift(s)||^2 = 1."""
        samples = random_simplex(5, 100, rng)
        lifted = fisher_lift(samples)
        norms_sq = np.sum(lifted**2, axis=-1)
        np.testing.assert_allclose(norms_sq, 1.0, atol=1e-14)

    def test_e8_fisher_lift_nonneg(self, rng: np.random.Generator) -> None:
        """E8: fisher_lift(s) >= 0 elementwise."""
        samples = random_simplex(5, 100, rng)
        lifted = fisher_lift(samples)
        assert np.all(lifted >= 0)


# ---------------------------------------------------------------------------
# Overlap families — additional tests
# ---------------------------------------------------------------------------


class TestOverlap:
    """Tests for phi, psi_overlap, overlap_divergence."""

    def test_psi_alias(self) -> None:
        """psi is an alias for psi_overlap."""
        assert psi is psi_overlap

    def test_psi_zero_entry(self) -> None:
        """psi_overlap returns 0 when any entry is 0."""
        s = vertex_simplex(3, 0)
        assert psi_overlap(s) == 0.0

    def test_phi_uniform(self) -> None:
        """phi(uniform) = N/(N-1) * (1 - N*(1/N)^2) = 1."""
        for n in (3, 5, 10):
            s = uniform_simplex(n)
            np.testing.assert_allclose(phi(s), 1.0, atol=1e-14)

    def test_single_composition(self) -> None:
        """Functions work on 1-d input."""
        s = np.array([0.5, 0.3, 0.2])
        result = phi(s)
        assert np.isscalar(result) or result.ndim == 0


# ---------------------------------------------------------------------------
# Forced-pair invariants
# ---------------------------------------------------------------------------


class TestForcedPair:
    """Tests for q_delta, h3, forced_pair, forced_coordinates."""

    def test_binary_q_delta_identity(self) -> None:
        """Paper §4.2: Q_Δ(x, 1-x) = 2*(x - 1/2)^2 for 100 values."""
        xs = np.linspace(0.0, 1.0, 100)
        for x in xs:
            s = np.array([x, 1.0 - x])
            result = q_delta(s)
            expected = 2.0 * (x - 0.5) ** 2
            np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_forced_pair_returns_tuple(self) -> None:
        s = uniform_simplex(3)
        q, h = forced_pair(s)
        np.testing.assert_allclose(q, q_delta(s))
        np.testing.assert_allclose(h, h3(s))

    def test_forced_coordinates_1d(self) -> None:
        s = uniform_simplex(4)
        fc = forced_coordinates(s)
        assert fc.shape == (2,)
        np.testing.assert_allclose(fc[0], q_delta(s))
        np.testing.assert_allclose(fc[1], h3(s))

    def test_forced_coordinates_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        fc = forced_coordinates(samples)
        assert fc.shape == (10, 2)


# ---------------------------------------------------------------------------
# Heuristic diagnostic
# ---------------------------------------------------------------------------


class TestQHRatio:
    """Tests for qh_ratio."""

    def test_on_zero_nan(self) -> None:
        result = qh_ratio(np.array([0.5, 0.3, 0.2]))
        assert np.isfinite(result)

    def test_on_zero_modes(self) -> None:
        """Test that on_zero parameter is accepted for all modes."""
        s = np.array([0.5, 0.3, 0.2])
        for mode in ("nan", "inf", "clip"):
            result = qh_ratio(s, on_zero=mode)
            assert np.isfinite(result)  # this input has nonzero H3


# ---------------------------------------------------------------------------
# Bridge statistics
# ---------------------------------------------------------------------------


class TestBridgeStatistics:
    """Tests for herfindahl, simpson_index, shannon_entropy, etc."""

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_herfindahl_uniform(self, n: int) -> None:
        """herfindahl(uniform) = 1/N."""
        s = uniform_simplex(n)
        np.testing.assert_allclose(herfindahl(s), 1.0 / n, atol=1e-14)

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_shannon_entropy_uniform(self, n: int) -> None:
        """shannon_entropy(uniform) = log(N)."""
        s = uniform_simplex(n)
        np.testing.assert_allclose(shannon_entropy(s), np.log(n), atol=1e-14)

    def test_simpson_index_uniform(self) -> None:
        s = uniform_simplex(4)
        np.testing.assert_allclose(simpson_index(s), 1.0 - 1.0 / 4)

    def test_gini_simpson_alias(self) -> None:
        assert gini_simpson is simpson_index

    def test_effective_number_herfindahl_uniform(self) -> None:
        s = uniform_simplex(5)
        np.testing.assert_allclose(effective_number_herfindahl(s), 5.0, atol=1e-14)

    def test_effective_number_shannon_uniform(self) -> None:
        s = uniform_simplex(5)
        np.testing.assert_allclose(effective_number_shannon(s), 5.0, atol=1e-12)

    def test_shannon_entropy_base2(self) -> None:
        s = uniform_simplex(4)
        np.testing.assert_allclose(
            shannon_entropy(s, base=2), np.log(4) / np.log(2), atol=1e-14
        )

    def test_shannon_entropy_zero_entry(self) -> None:
        """0 * log(0) = 0 convention."""
        s = vertex_simplex(3, 0)
        result = shannon_entropy(s)
        assert np.isfinite(result)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_herfindahl_vertex(self) -> None:
        s = vertex_simplex(3, 0)
        np.testing.assert_allclose(herfindahl(s), 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------


class TestBinaryHelpers:
    """Tests for binary_overlap and binary_fisher_angle."""

    def test_binary_overlap_half(self) -> None:
        np.testing.assert_allclose(binary_overlap(0.5), 1.0, atol=1e-14)

    def test_binary_overlap_zero(self) -> None:
        np.testing.assert_allclose(binary_overlap(0.0), 0.0, atol=1e-14)

    def test_binary_overlap_one(self) -> None:
        np.testing.assert_allclose(binary_overlap(1.0), 0.0, atol=1e-14)

    def test_binary_fisher_angle_one(self) -> None:
        """arccos(sqrt(1)) = arccos(1) = 0."""
        np.testing.assert_allclose(binary_fisher_angle(1.0), 0.0, atol=1e-14)

    def test_binary_fisher_angle_half(self) -> None:
        expected = 2.0 * np.arccos(np.sqrt(0.5))
        np.testing.assert_allclose(binary_fisher_angle(0.5), expected, atol=1e-14)

    def test_binary_overlap_array(self) -> None:
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = binary_overlap(x)
        expected = 4.0 * x * (1.0 - x)
        np.testing.assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Batch shape tests
# ---------------------------------------------------------------------------


class TestBatchShapes:
    """All functions handle (M, N) input correctly."""

    def test_phi_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = phi(samples)
        assert result.shape == (10,)

    def test_psi_overlap_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = psi_overlap(samples)
        assert result.shape == (10,)

    def test_q_delta_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = q_delta(samples)
        assert result.shape == (10,)

    def test_h3_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = h3(samples)
        assert result.shape == (10,)

    def test_herfindahl_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = herfindahl(samples)
        assert result.shape == (10,)

    def test_shannon_entropy_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = shannon_entropy(samples)
        assert result.shape == (10,)

    def test_fisher_lift_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = fisher_lift(samples)
        assert result.shape == (10, 4)

    def test_fisher_project_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        lifted = fisher_lift(samples)
        result = fisher_project(lifted)
        assert result.shape == (10, 4)

    def test_qh_ratio_batch(self, rng: np.random.Generator) -> None:
        samples = random_simplex(4, 10, rng)
        result = qh_ratio(samples)
        assert result.shape == (10,)


# ---------------------------------------------------------------------------
# fisher_project contract validation
# ---------------------------------------------------------------------------


class TestFisherProjectContract:
    """fisher_project must enforce nonneg and warn on non-unit-norm."""

    def test_rejects_negative_amplitudes(self) -> None:
        """Negative elements raise ValueError."""
        psi = np.array([-0.5, 0.5, 0.707])
        with pytest.raises(ValueError, match="nonnegative"):
            fisher_project(psi)

    def test_warns_non_unit_norm(self) -> None:
        """Non-unit-norm input emits UserWarning."""
        psi = np.array([0.5, 0.5, 0.5])  # norm ~ 0.866
        with pytest.warns(UserWarning, match="non-unit-norm"):
            fisher_project(psi)

    def test_accepts_valid_amplitudes(self) -> None:
        """Valid nonneg unit-norm input returns psi^2 without warning."""
        import warnings

        psi = np.array([0.5, 0.5, np.sqrt(0.5)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = fisher_project(psi)
        np.testing.assert_allclose(result, psi**2, atol=1e-14)

    def test_roundtrip_no_warning(self, rng: np.random.Generator) -> None:
        """fisher_project(fisher_lift(s)) does not warn."""
        import warnings

        samples = random_simplex(5, 20, rng)
        lifted = fisher_lift(samples)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = fisher_project(lifted)
        np.testing.assert_allclose(result, samples, atol=1e-14)
