"""Tests for fisher_simplex.analysis module — R-series, S-series, and extras."""

from __future__ import annotations

import numpy as np
import pytest

from fisher_simplex.analysis import (
    batch_diagnostic,
    community_type_discriminant,
    comparative_index_report,
    concentration_profile_report,
    distributional_shift,
    divergence_analysis,
    forced_block_regression,
    full_diagnostic,
    pairwise_ranking_disagreement,
    sufficient_statistic_efficiency,
)
from fisher_simplex.generators import (
    dirichlet_community,
    geometric_series,
    lognormal_community,
)
from tests.conftest import random_simplex

# ---------------------------------------------------------------------------
# R-series tests (spec section 6.3)
# ---------------------------------------------------------------------------


class TestRSeries:
    """R1-R2: forced-block regression tests."""

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_r1_p4_in_forced_block(self, n: int) -> None:
        """R1: p4 = sum(s_i^4) lies in the forced block.

        Generate 500 Dirichlet(1) samples, compute target = sum(s^4),
        fit against (Q_delta, H_3) quadratically, assert R^2 > 0.999.
        """
        rng = np.random.default_rng(seed=42)
        X = dirichlet_community(n, 500, alpha=1.0, rng=rng)
        target = np.sum(X**4, axis=-1)

        result = sufficient_statistic_efficiency(X, target)
        assert result["r_squared_quadratic"] > 0.999, (
            f"R^2_quad = {result['r_squared_quadratic']:.6f} for N={n}"
        )
        assert result["in_forced_block"] is True

    def test_r2_p5_beyond_forced_block(self) -> None:
        """R2: p5 is NOT fully in the forced block at N=30.

        R^2(p5 | Q,H quad) < R^2(p4 | Q,H quad).
        """
        rng = np.random.default_rng(seed=42)
        n = 30
        X = dirichlet_community(n, 500, alpha=1.0, rng=rng)

        target_p4 = np.sum(X**4, axis=-1)
        target_p5 = np.sum(X**5, axis=-1)

        r2_p4 = sufficient_statistic_efficiency(X, target_p4)["r_squared_quadratic"]
        r2_p5 = sufficient_statistic_efficiency(X, target_p5)["r_squared_quadratic"]

        assert r2_p5 < r2_p4, f"R^2(p5)={r2_p5:.6f} should be < R^2(p4)={r2_p4:.6f}"


# ---------------------------------------------------------------------------
# S-series tests (spec section 6.4)
# ---------------------------------------------------------------------------


class TestSSeries:
    """S1-S3: statistical property tests."""

    def test_s1_divergence_growth(self) -> None:
        """S1: mean |Phi - Psi| grows with N under Dirichlet(1).

        mean |D| at N=3 < N=5 < N=10 < N=20.
        """
        rng = np.random.default_rng(seed=42)
        m = 500
        ns = [3, 5, 10, 20]
        mean_divs = []

        for n in ns:
            X = dirichlet_community(n, m, alpha=1.0, rng=rng)
            # Use mean of absolute divergence
            d = np.abs(batch_diagnostic(X)["divergence"])
            mean_divs.append(float(np.mean(d)))

        for i in range(len(ns) - 1):
            assert mean_divs[i] < mean_divs[i + 1], (
                f"mean|D| at N={ns[i]} ({mean_divs[i]:.6f}) "
                f"should be < N={ns[i + 1]} ({mean_divs[i + 1]:.6f})"
            )

    def test_s2_generator_separation(self) -> None:
        """S2: geometric_series and lognormal_community produce distinct clouds.

        Mean (Q_delta, H_3) should differ significantly between generators.
        """
        rng = np.random.default_rng(seed=42)
        n, m = 10, 200

        X_geo = geometric_series(n, m, rng=rng)
        X_log = lognormal_community(n, m, rng=rng)

        diag_geo = batch_diagnostic(X_geo)
        diag_log = batch_diagnostic(X_log)

        # Means of Q_delta should differ
        q_mean_geo = np.mean(diag_geo["q_delta"])
        q_mean_log = np.mean(diag_log["q_delta"])
        assert abs(q_mean_geo - q_mean_log) > 0.01, (
            f"Q_delta means too similar: geo={q_mean_geo:.4f}, log={q_mean_log:.4f}"
        )

        # Means of H_3 should also differ
        h_mean_geo = np.mean(diag_geo["h3"])
        h_mean_log = np.mean(diag_log["h3"])
        # At least one of Q or H means should differ
        q_diff = abs(q_mean_geo - q_mean_log)
        h_diff = abs(h_mean_geo - h_mean_log)
        assert q_diff > 0.01 or h_diff > 0.001

    def test_s3_divergence_zero_for_n2(self) -> None:
        """S3: divergence_analysis returns mean_divergence ~ 0 for N=2 data.

        For N=2, Phi and Psi coincide identically.
        """
        rng = np.random.default_rng(seed=42)
        X = dirichlet_community(2, 200, alpha=1.0, rng=rng)
        result = divergence_analysis(X)
        assert abs(result["mean_divergence"]) < 1e-10, (
            f"mean_divergence = {result['mean_divergence']:.2e} for N=2"
        )


# ---------------------------------------------------------------------------
# Batch diagnostic tests
# ---------------------------------------------------------------------------


class TestBatchDiagnostic:
    """Tests for batch_diagnostic and full_diagnostic."""

    def test_batch_diagnostic_keys(self, rng: np.random.Generator) -> None:
        """batch_diagnostic returns all expected keys."""
        X = random_simplex(5, 20, rng)
        result = batch_diagnostic(X)
        expected_keys = {
            "n_components",
            "n_compositions",
            "phi",
            "psi",
            "divergence",
            "q_delta",
            "h3",
            "herfindahl",
            "simpson",
            "shannon",
            "fisher_coords",
        }
        assert set(result.keys()) == expected_keys

    def test_batch_diagnostic_shapes(self, rng: np.random.Generator) -> None:
        """batch_diagnostic arrays have correct shapes."""
        m, n = 20, 5
        X = random_simplex(n, m, rng)
        result = batch_diagnostic(X)

        assert result["n_components"] == n
        assert result["n_compositions"] == m

        for key in [
            "phi",
            "psi",
            "divergence",
            "q_delta",
            "h3",
            "herfindahl",
            "simpson",
            "shannon",
        ]:
            assert result[key].shape == (m,), f"{key} has wrong shape"

        assert result["fisher_coords"].shape == (m, n)

    def test_batch_diagnostic_with_heuristics(self, rng: np.random.Generator) -> None:
        """include_heuristics=True adds qh_ratio."""
        X = random_simplex(5, 20, rng)
        result = batch_diagnostic(X, include_heuristics=True)
        assert "qh_ratio" in result
        assert result["qh_ratio"].shape == (20,)

    def test_full_diagnostic_scalar_values(self, rng: np.random.Generator) -> None:
        """full_diagnostic returns scalar values for a single composition."""
        s = random_simplex(5, 1, rng)[0]
        result = full_diagnostic(s)

        assert result["n_components"] == 5
        assert result["n_compositions"] == 1

        for key in [
            "phi",
            "psi",
            "divergence",
            "q_delta",
            "h3",
            "herfindahl",
            "simpson",
            "shannon",
        ]:
            assert isinstance(result[key], float), f"{key} is not float"

        assert result["fisher_coords"].shape == (5,)


# ---------------------------------------------------------------------------
# Distributional shift tests
# ---------------------------------------------------------------------------


class TestDistributionalShift:
    """Tests for distributional_shift."""

    def test_distributional_shift_keys(self, rng: np.random.Generator) -> None:
        """distributional_shift returns correct dict keys."""
        X_ref = random_simplex(5, 15, rng)
        X_test = random_simplex(5, 15, rng)
        result = distributional_shift(X_ref, X_test)

        expected_keys = {
            "mean_distance",
            "cloud_distance",
            "ref_dispersion",
            "test_dispersion",
        }
        assert set(result.keys()) == expected_keys

    def test_distributional_shift_nonnegative(self, rng: np.random.Generator) -> None:
        """All distributional_shift values are nonnegative."""
        X_ref = random_simplex(5, 15, rng)
        X_test = random_simplex(5, 15, rng)
        result = distributional_shift(X_ref, X_test)

        for key, val in result.items():
            assert val >= 0, f"{key} = {val} is negative"

    def test_distributional_shift_self_small(self, rng: np.random.Generator) -> None:
        """Shift from a sample to itself has near-zero distances.

        Tolerance is 1e-7 because arccos amplifies ULP-level errors
        near BC=1, yielding d_F ~ O(sqrt(machine_eps)).
        """
        X = random_simplex(5, 20, rng)
        result = distributional_shift(X, X)
        assert result["cloud_distance"] < 1e-7
        assert result["mean_distance"] < 1e-7

    def test_invalid_summary_raises(self, rng: np.random.Generator) -> None:
        """Unknown summary parameter raises ValueError."""
        X_ref = random_simplex(5, 10, rng)
        X_test = random_simplex(5, 10, rng)
        with pytest.raises(ValueError, match="Unknown summary"):
            distributional_shift(X_ref, X_test, summary="median")


# ---------------------------------------------------------------------------
# Pairwise ranking disagreement tests
# ---------------------------------------------------------------------------


class TestPairwiseRankingDisagreement:
    """Tests for pairwise_ranking_disagreement."""

    def test_perfect_agreement(self) -> None:
        """Identical orderings have zero disagreement."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = pairwise_ranking_disagreement(a, a)
        assert result["disagreement_rate"] == 0.0
        assert result["n_disagreements"] == 0
        assert result["n_pairs"] == 6

    def test_reversed_ordering(self) -> None:
        """Reversed orderings have maximum disagreement."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        result = pairwise_ranking_disagreement(a, b)
        assert result["disagreement_rate"] == 1.0
        assert result["n_disagreements"] == 6
        assert result["concordance_tau"] < 0

    def test_partial_disagreement(self) -> None:
        """Partially different orderings have intermediate disagreement."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 3.0, 2.0])
        result = pairwise_ranking_disagreement(a, b)
        assert 0 < result["disagreement_rate"] < 1.0
        assert result["n_pairs"] == 3


# ---------------------------------------------------------------------------
# Sufficient statistic efficiency tests
# ---------------------------------------------------------------------------


class TestSufficientStatisticEfficiency:
    """Tests for sufficient_statistic_efficiency."""

    def test_returns_correct_keys(self, rng: np.random.Generator) -> None:
        """Returns all expected keys."""
        X = random_simplex(5, 50, rng)
        target = np.sum(X**2, axis=-1)
        result = sufficient_statistic_efficiency(X, target)

        expected_keys = {
            "r_squared_linear",
            "r_squared_quadratic",
            "in_forced_block",
            "residual_std",
        }
        assert set(result.keys()) == expected_keys

    def test_hhi_in_forced_block(self, rng: np.random.Generator) -> None:
        """HHI (sum s^2) should be perfectly predicted by Q_delta.

        HHI = Q_delta + 1/N, so linear R^2 should be ~1.
        """
        X = random_simplex(5, 200, rng)
        target = np.sum(X**2, axis=-1)
        result = sufficient_statistic_efficiency(X, target)
        assert result["r_squared_linear"] > 0.999

    def test_invalid_model_raises(self, rng: np.random.Generator) -> None:
        """Unknown model parameter raises ValueError."""
        X = random_simplex(5, 50, rng)
        target = np.sum(X**2, axis=-1)
        with pytest.raises(ValueError, match="Unknown model"):
            sufficient_statistic_efficiency(X, target, model="cubic")


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestReports:
    """Tests for text report generation."""

    def test_comparative_index_report_is_string(self, rng: np.random.Generator) -> None:
        """comparative_index_report returns a non-empty string."""
        X = random_simplex(5, 20, rng)
        report = comparative_index_report(X)
        assert isinstance(report, str)
        assert len(report) > 100
        assert "Comparative Index Report" in report

    def test_concentration_profile_report_is_string(
        self,
        rng: np.random.Generator,
    ) -> None:
        """concentration_profile_report returns a non-empty string."""
        X = random_simplex(5, 20, rng)
        report = concentration_profile_report(X)
        assert isinstance(report, str)
        assert len(report) > 100
        assert "Concentration Profile Report" in report


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestCommunityTypeDiscriminant:
    """Tests for community_type_discriminant."""

    def test_returns_correct_keys(self, rng: np.random.Generator) -> None:
        """Returns labels, scores, method, calibrator."""
        X = random_simplex(5, 20, rng)
        result = community_type_discriminant(X)
        assert set(result.keys()) == {"labels", "scores", "method", "calibrator"}
        assert result["method"] == "pair"
        assert result["calibrator"] is None
        assert result["labels"].shape == (20,)
        assert result["scores"].shape == (20, 2)

    def test_synthetic_v0_calibrator(self, rng: np.random.Generator) -> None:
        """synthetic_v0 calibrator works without error."""
        X = random_simplex(5, 20, rng)
        result = community_type_discriminant(X, calibrator="synthetic_v0")
        assert result["calibrator"] == "synthetic_v0"

    def test_unknown_calibrator_raises(self, rng: np.random.Generator) -> None:
        """Unknown calibrator raises ValueError."""
        X = random_simplex(5, 10, rng)
        with pytest.raises(ValueError, match="Unknown calibrator"):
            community_type_discriminant(X, calibrator="nonsense")

    def test_dispersed_label_reachable(self) -> None:
        """Near-uniform compositions should receive 'dispersed' label."""
        n = 5
        # Near-uniform compositions have very low Q_delta
        X = np.full((10, n), 1.0 / n)
        # Add tiny noise to avoid exact uniformity triggering issues
        rng = np.random.default_rng(42)
        X = X + rng.normal(0, 1e-6, X.shape)
        X = X / X.sum(axis=-1, keepdims=True)
        result = community_type_discriminant(X)
        assert "dispersed" in result["labels"]


# ---------------------------------------------------------------------------
# Forced-block regression tests
# ---------------------------------------------------------------------------


class TestForcedBlockRegression:
    """Tests for forced_block_regression."""

    def test_returns_correct_keys(self, rng: np.random.Generator) -> None:
        """forced_block_regression returns expected dict keys."""
        X = random_simplex(5, 50, rng)
        target = np.sum(X**2, axis=-1)
        result = forced_block_regression(X, target)
        expected_keys = {
            "coefficients",
            "predictions",
            "residuals",
            "r_squared",
        }
        assert set(result.keys()) == expected_keys

    def test_shapes(self, rng: np.random.Generator) -> None:
        """Predictions and residuals have shape (M,)."""
        m = 50
        X = random_simplex(5, m, rng)
        target = np.sum(X**2, axis=-1)
        result = forced_block_regression(X, target)
        assert result["predictions"].shape == (m,)
        assert result["residuals"].shape == (m,)
        assert isinstance(result["r_squared"], float)

    def test_hhi_high_r_squared(self, rng: np.random.Generator) -> None:
        """HHI = sum(s^2) is well-predicted by (Q, H) quadratic fit."""
        X = random_simplex(5, 200, rng)
        target = np.sum(X**2, axis=-1)
        result = forced_block_regression(X, target)
        assert result["r_squared"] > 0.999

    def test_unknown_basis_raises(self, rng: np.random.Generator) -> None:
        """Unknown basis raises ValueError."""
        X = random_simplex(5, 20, rng)
        target = np.sum(X**2, axis=-1)
        with pytest.raises(ValueError, match="Unknown basis"):
            forced_block_regression(X, target, basis="triple")


# ---------------------------------------------------------------------------
# Divergence analysis tests
# ---------------------------------------------------------------------------


class TestDivergenceAnalysis:
    """Tests for divergence_analysis."""

    def test_returns_correct_keys(self, rng: np.random.Generator) -> None:
        """divergence_analysis returns all expected keys."""
        X = random_simplex(5, 30, rng)
        result = divergence_analysis(X)
        expected_keys = {
            "n_components",
            "mean_divergence",
            "max_divergence",
            "divergence_std",
            "fraction_consequential",
            "ranking_disagreement",
            "recommendation",
        }
        assert set(result.keys()) == expected_keys

    def test_recommendation_is_heuristic(self, rng: np.random.Generator) -> None:
        """Recommendation string is labeled as empirical heuristic."""
        X = random_simplex(5, 30, rng)
        result = divergence_analysis(X)
        assert "heuristic" in result["recommendation"].lower()
