"""Tests for fisher_simplex.interp module."""

from __future__ import annotations

import numpy as np
import pytest

from fisher_simplex.geometry import fisher_geodesic
from fisher_simplex.interp import (
    chart_stability,
    discover_charts,
    extract_shared_modes,
    mean_overlap_matrix,
    overlap_matrix,
    project_to_modes,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# overlap_matrix
# ---------------------------------------------------------------------------


class TestOverlapMatrix:
    """Tests for overlap_matrix."""

    def test_diagonal_is_one(self, rng):
        X = rng.dirichlet(np.ones(5), size=10)
        ov = overlap_matrix(X)
        np.testing.assert_allclose(np.diag(ov), 1.0, atol=1e-10)

    def test_symmetric(self, rng):
        X = rng.dirichlet(np.ones(5), size=10)
        ov = overlap_matrix(X)
        np.testing.assert_allclose(ov, ov.T, atol=1e-15)

    def test_values_in_01(self, rng):
        X = rng.dirichlet(np.ones(5), size=10)
        ov = overlap_matrix(X)
        assert np.all(ov >= -1e-15)
        assert np.all(ov <= 1.0 + 1e-15)

    def test_identical_distributions_overlap_one(self):
        s = np.array([0.5, 0.3, 0.2])
        X = np.tile(s, (5, 1))
        ov = overlap_matrix(X)
        np.testing.assert_allclose(ov, 1.0, atol=1e-10)

    def test_vertex_distributions_overlap_zero(self):
        X = np.eye(3)
        ov = overlap_matrix(X)
        np.testing.assert_allclose(ov[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(ov[0, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(ov[1, 2], 0.0, atol=1e-10)

    def test_shape(self, rng):
        X = rng.dirichlet(np.ones(5), size=7)
        ov = overlap_matrix(X)
        assert ov.shape == (7, 7)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            overlap_matrix(np.array([0.5, 0.3, 0.2]))


# ---------------------------------------------------------------------------
# mean_overlap_matrix
# ---------------------------------------------------------------------------


class TestMeanOverlapMatrix:
    """Tests for mean_overlap_matrix."""

    def test_single_condition_matches_overlap_matrix(self, rng):
        X = rng.dirichlet(np.ones(5), size=8)
        X_3d = X[:, np.newaxis, :]  # (8, 1, 5)
        mean_ov = mean_overlap_matrix(X_3d)
        ov = overlap_matrix(X)
        np.testing.assert_allclose(mean_ov, ov, atol=1e-10)

    def test_symmetric(self, rng):
        n_e, n_c, N = 6, 4, 5
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)
        mean_ov = mean_overlap_matrix(X_3d)
        np.testing.assert_allclose(mean_ov, mean_ov.T, atol=1e-15)

    def test_diagonal_close_to_one(self, rng):
        n_e, n_c, N = 6, 4, 5
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)
        mean_ov = mean_overlap_matrix(X_3d)
        # Diagonal is average of self-overlaps (all 1.0)
        np.testing.assert_allclose(np.diag(mean_ov), 1.0, atol=1e-10)

    def test_shape(self, rng):
        n_e, n_c, N = 6, 4, 5
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)
        mean_ov = mean_overlap_matrix(X_3d)
        assert mean_ov.shape == (n_e, n_e)

    def test_rejects_2d_input(self, rng):
        with pytest.raises(ValueError):
            mean_overlap_matrix(rng.dirichlet(np.ones(5), size=10))


# ---------------------------------------------------------------------------
# discover_charts
# ---------------------------------------------------------------------------


class TestDiscoverCharts:
    """Tests for discover_charts."""

    def test_two_blocks(self):
        ov = np.block([
            [np.ones((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.ones((3, 3))],
        ])
        charts = discover_charts(ov, tau=0.5)
        assert charts["n_charts"] == 2
        assert len(charts["labels"]) == 6
        for c_id, members in charts["charts"].items():
            assert len(members) == 3

    def test_single_block(self):
        ov = np.ones((5, 5))
        charts = discover_charts(ov, tau=0.5)
        assert charts["n_charts"] == 1
        assert charts["sizes"][0] == 5

    def test_all_disconnected(self):
        ov = np.eye(5)
        charts = discover_charts(ov, tau=0.5)
        assert charts["n_charts"] == 5

    def test_returns_expected_keys(self):
        ov = np.ones((3, 3))
        charts = discover_charts(ov, tau=0.5)
        assert set(charts.keys()) == {"n_charts", "labels", "charts", "sizes"}

    def test_labels_cover_all_entities(self, rng):
        X = rng.dirichlet(np.ones(5), size=10)
        ov = overlap_matrix(X)
        charts = discover_charts(ov, tau=0.5)
        assert len(charts["labels"]) == 10
        # Every entity appears in exactly one chart
        all_members = []
        for members in charts["charts"].values():
            all_members.extend(members)
        assert sorted(all_members) == list(range(10))

    def test_well_separated_groups(self, rng):
        """Two groups of distributions that are far apart should form 2 charts."""
        group_a = rng.dirichlet([50, 1, 1, 1, 1], size=4)
        group_b = rng.dirichlet([1, 1, 1, 1, 50], size=4)
        X = np.vstack([group_a, group_b])
        ov = overlap_matrix(X)
        charts = discover_charts(ov, tau=0.5)
        assert charts["n_charts"] == 2


# ---------------------------------------------------------------------------
# chart_stability
# ---------------------------------------------------------------------------


class TestChartStability:
    """Tests for chart_stability."""

    def test_stable_groups_high_co_occurrence(self, rng):
        n_entities, n_conditions, N = 6, 8, 5
        X_3d = np.zeros((n_entities, n_conditions, N))
        for c in range(n_conditions):
            X_3d[:3, c] = rng.dirichlet([50, 1, 1, 1, 1], size=3)
            X_3d[3:, c] = rng.dirichlet([1, 1, 1, 1, 50], size=3)

        result = chart_stability(X_3d, tau=0.5)
        # Within-group co-occurrence should be high
        assert result["co_occurrence"][0, 1] > 0.7
        assert result["co_occurrence"][3, 4] > 0.7
        # Cross-group co-occurrence should be low
        assert result["co_occurrence"][0, 3] < 0.3

    def test_returns_expected_keys(self, rng):
        n_e, n_c, N = 4, 3, 5
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)

        result = chart_stability(X_3d, tau=0.5)
        expected = {
            "co_occurrence", "stable_labels",
            "n_stable_charts", "per_condition_labels",
        }
        assert set(result.keys()) == expected

    def test_co_occurrence_shape(self, rng):
        n_e, n_c, N = 5, 4, 3
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)
        result = chart_stability(X_3d, tau=0.5)
        assert result["co_occurrence"].shape == (n_e, n_e)
        assert result["per_condition_labels"].shape == (n_c, n_e)


# ---------------------------------------------------------------------------
# extract_shared_modes
# ---------------------------------------------------------------------------


class TestExtractSharedModes:
    """Tests for extract_shared_modes."""

    def test_shared_trajectory_high_r2(self, rng):
        """All entities follow the same geodesic with small noise → high R²."""
        n_entities, n_conditions, N = 5, 15, 5
        p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        q = np.array([0.05, 0.1, 0.15, 0.2, 0.5])

        X_3d = np.zeros((n_entities, n_conditions, N))
        for h in range(n_entities):
            for c in range(n_conditions):
                t = c / (n_conditions - 1)
                base = fisher_geodesic(p, q, t)
                noise = rng.normal(0, 0.003, N)
                perturbed = base + noise
                perturbed = np.maximum(perturbed, 1e-10)
                X_3d[h, c] = perturbed / perturbed.sum()

        result = extract_shared_modes(X_3d, n_modes=3)
        assert result["mean_r2"] > 0.5

    def test_returns_expected_keys(self, rng):
        n_e, n_c, N = 4, 8, 5
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)

        result = extract_shared_modes(X_3d, n_modes=2)
        expected_keys = {
            "centroid", "tangent_mean", "tangent_basis",
            "shared_modes", "shared_singular_values",
            "head_effects", "per_entity_r2", "mean_r2",
            "dim_90", "dim_95", "tangent_singular_values",
        }
        assert set(result.keys()) == expected_keys

    def test_centroid_is_valid_simplex(self, rng):
        X_3d = np.zeros((3, 5, 4))
        for c in range(5):
            X_3d[:, c] = rng.dirichlet(np.ones(4), size=3)

        result = extract_shared_modes(X_3d)
        centroid = result["centroid"]
        assert np.all(centroid >= 0)
        np.testing.assert_allclose(centroid.sum(), 1.0, atol=1e-6)

    def test_r2_in_valid_range(self, rng):
        X_3d = np.zeros((3, 8, 4))
        for c in range(8):
            X_3d[:, c] = rng.dirichlet(np.ones(4), size=3)

        result = extract_shared_modes(X_3d)
        # R² can be slightly negative due to numerical issues but not wildly
        assert np.all(result["per_entity_r2"] >= -0.1)
        assert np.all(result["per_entity_r2"] <= 1.01)

    def test_shapes(self, rng):
        n_e, n_c, N = 4, 8, 5
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)

        result = extract_shared_modes(X_3d, n_modes=2, tangent_dim=4)
        assert result["centroid"].shape == (N,)
        assert result["tangent_mean"].shape == (N,)
        assert result["tangent_basis"].shape[0] == 4
        assert result["tangent_basis"].shape[1] == N
        assert result["shared_modes"].shape[0] == 2
        assert result["shared_modes"].shape[1] == 4
        assert result["head_effects"].shape == (n_e, 4)
        assert result["per_entity_r2"].shape == (n_e,)

    def test_custom_centroid(self, rng):
        n_e, n_c, N = 3, 6, 5
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)

        custom_centroid = np.full(N, 1.0 / N)
        result = extract_shared_modes(X_3d, centroid=custom_centroid)
        np.testing.assert_allclose(result["centroid"], custom_centroid)

    def test_dim_estimates_positive(self, rng):
        X_3d = np.zeros((4, 10, 5))
        for c in range(10):
            X_3d[:, c] = rng.dirichlet(np.ones(5), size=4)

        result = extract_shared_modes(X_3d)
        assert result["dim_90"] >= 1
        assert result["dim_95"] >= result["dim_90"]

    def test_rejects_2d_input(self, rng):
        with pytest.raises(ValueError):
            extract_shared_modes(rng.dirichlet(np.ones(5), size=10))


# ---------------------------------------------------------------------------
# project_to_modes
# ---------------------------------------------------------------------------


class TestProjectToModes:
    """Tests for project_to_modes."""

    def _make_modes(self, rng, n_e=4, n_c=8, N=5, n_modes=2):
        """Helper to build modes for testing."""
        X_3d = np.zeros((n_e, n_c, N))
        for c in range(n_c):
            X_3d[:, c] = rng.dirichlet(np.ones(N), size=n_e)
        return extract_shared_modes(X_3d, n_modes=n_modes), X_3d

    def test_output_is_valid_simplex(self, rng):
        modes, X_3d = self._make_modes(rng)
        s = X_3d[0, 0]
        result = project_to_modes(
            s,
            centroid=modes["centroid"],
            tangent_basis=modes["tangent_basis"],
            shared_modes=modes["shared_modes"],
            tangent_mean=modes["tangent_mean"],
        )
        assert result.shape == s.shape
        assert np.all(result >= 0)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_batch_output_valid(self, rng):
        modes, X_3d = self._make_modes(rng)
        S = X_3d[:, 0, :]  # (n_e, N) batch
        result = project_to_modes(
            S,
            centroid=modes["centroid"],
            tangent_basis=modes["tangent_basis"],
            shared_modes=modes["shared_modes"],
            tangent_mean=modes["tangent_mean"],
        )
        assert result.shape == S.shape
        assert np.all(result >= 0)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)

    def test_centroid_no_centering_returns_centroid(self, rng):
        """Without tangent_mean, projecting centroid should give back centroid."""
        modes, _ = self._make_modes(rng)
        centroid = modes["centroid"]
        result = project_to_modes(
            centroid,
            centroid=centroid,
            tangent_basis=modes["tangent_basis"],
            shared_modes=modes["shared_modes"],
            # tangent_mean=None → no centering
        )
        # Logmap of centroid at centroid is ~0, so reconstruction ≈ centroid
        np.testing.assert_allclose(result, centroid, atol=0.05)

    def test_n_modes_parameter(self, rng):
        modes, X_3d = self._make_modes(rng, n_modes=3)
        s = X_3d[0, 0]
        # Different n_modes should give different results
        r1 = project_to_modes(
            s,
            centroid=modes["centroid"],
            tangent_basis=modes["tangent_basis"],
            shared_modes=modes["shared_modes"],
            tangent_mean=modes["tangent_mean"],
            n_modes=1,
        )
        r3 = project_to_modes(
            s,
            centroid=modes["centroid"],
            tangent_basis=modes["tangent_basis"],
            shared_modes=modes["shared_modes"],
            tangent_mean=modes["tangent_mean"],
            n_modes=3,
        )
        # Both valid simplices
        assert np.all(r1 >= 0)
        assert np.all(r3 >= 0)
        np.testing.assert_allclose(r1.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(r3.sum(), 1.0, atol=1e-6)

    def test_with_head_effect(self, rng):
        modes, X_3d = self._make_modes(rng)
        s = X_3d[0, 0]
        head_effect = modes["head_effects"][0]
        result = project_to_modes(
            s,
            centroid=modes["centroid"],
            tangent_basis=modes["tangent_basis"],
            shared_modes=modes["shared_modes"],
            tangent_mean=modes["tangent_mean"],
            head_effect=head_effect,
        )
        assert result.shape == s.shape
        assert np.all(result >= 0)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)
