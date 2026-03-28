"""Tests for fisher_simplex.geometry module — G-series and extras."""

from __future__ import annotations

import numpy as np
import pytest

from fisher_simplex.geometry import (
    OnlineFisherMean,
    WindowedFisherStats,
    bhattacharyya_coefficient,
    cross_fisher_distances,
    fisher_barycenter,
    fisher_cosine,
    fisher_distance,
    fisher_expmap,
    fisher_geodesic,
    fisher_kernel,
    fisher_logmap,
    fisher_mean,
    fisher_pca,
    fisher_rbf_kernel,
    geodesic_interpolate,
    hellinger_distance,
    kernel_matrix,
    pairwise_fisher_distances,
    pairwise_hellinger_distances,
    perturb_simplex,
    polynomial_fisher_kernel,
    sample_near,
    tangent_map,
)
from tests.conftest import random_simplex

# ---------------------------------------------------------------------------
# G-series tests (spec section 6.2)
# ---------------------------------------------------------------------------


class TestGSeries:
    """G1-G11 geometry tests per spec."""

    def test_g1_self_distance_zero(self, rng: np.random.Generator) -> None:
        """G1: fisher_distance(s, s) = 0 for 100 random compositions.

        Tolerance is 1e-7 because arccos amplifies ULP-level errors near 1.0:
        2*arccos(1 - eps) ≈ 2*sqrt(2*eps), and eps ~ N * machine_epsilon.
        """
        samples = random_simplex(5, 100, rng)
        for s in samples:
            d = fisher_distance(s, s)
            np.testing.assert_allclose(d, 0.0, atol=1e-7)

    def test_g2_symmetry(self, rng: np.random.Generator) -> None:
        """G2: fisher_distance(a, b) = fisher_distance(b, a) for 100 pairs."""
        a = random_simplex(5, 100, rng)
        b = random_simplex(5, 100, rng)
        for i in range(100):
            d_ab = fisher_distance(a[i], b[i])
            d_ba = fisher_distance(b[i], a[i])
            np.testing.assert_allclose(d_ab, d_ba, atol=1e-14)

    def test_g3_triangle_inequality(self, rng: np.random.Generator) -> None:
        """G3: d(a,c) <= d(a,b) + d(b,c) + 1e-10 for 500 triples (N=5)."""
        a = random_simplex(5, 500, rng)
        b = random_simplex(5, 500, rng)
        c = random_simplex(5, 500, rng)
        for i in range(500):
            d_ac = fisher_distance(a[i], c[i])
            d_ab = fisher_distance(a[i], b[i])
            d_bc = fisher_distance(b[i], c[i])
            assert d_ac <= d_ab + d_bc + 1e-10

    def test_g4_geodesic_endpoints(self, rng: np.random.Generator) -> None:
        """G4: geodesic(a, b, 0) = a, geodesic(a, b, 1) = b."""
        a = random_simplex(5, 20, rng)
        b = random_simplex(5, 20, rng)
        for i in range(20):
            start = fisher_geodesic(a[i], b[i], 0.0)
            end = fisher_geodesic(a[i], b[i], 1.0)
            np.testing.assert_allclose(start, a[i], atol=1e-14)
            np.testing.assert_allclose(end, b[i], atol=1e-14)

    def test_g5_geodesic_valid_simplex(self, rng: np.random.Generator) -> None:
        """G5: geodesic sums to 1 and is nonneg for t in {0,.25,.5,.75,1}."""
        a = random_simplex(5, 20, rng)
        b = random_simplex(5, 20, rng)
        for i in range(20):
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                pt = fisher_geodesic(a[i], b[i], t)
                np.testing.assert_allclose(pt.sum(), 1.0, atol=1e-12)
                assert np.all(pt >= -1e-15)

    def test_g6_fisher_mean_valid(self, rng: np.random.Generator) -> None:
        """G6: fisher_mean(X) sums to 1 and is nonneg."""
        X = random_simplex(5, 50, rng)
        m = fisher_mean(X)
        np.testing.assert_allclose(m.sum(), 1.0, atol=1e-12)
        assert np.all(m >= -1e-15)

    def test_g7_zero_containing(self) -> None:
        """G7: functions work with zero-containing compositions."""
        a = np.array([0.5, 0.5, 0.0])
        b = np.array([0.0, 0.5, 0.5])
        c = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        # Distance works
        d = fisher_distance(a, b)
        assert np.isfinite(d) and d >= 0

        # Mean works
        X = np.array([a, b, c])
        m = fisher_mean(X)
        np.testing.assert_allclose(m.sum(), 1.0, atol=1e-12)
        assert np.all(m >= -1e-15)

        # Geodesic works
        pt = fisher_geodesic(a, b, 0.5)
        np.testing.assert_allclose(pt.sum(), 1.0, atol=1e-12)
        assert np.all(pt >= -1e-15)

    def test_g8_fisher_hellinger_relation(self, rng: np.random.Generator) -> None:
        """G8: d_F(a,b) = 2*arccos(1 - d_H(a,b)^2) within atol 1e-12."""
        a = random_simplex(5, 100, rng)
        b = random_simplex(5, 100, rng)
        for i in range(100):
            d_f = fisher_distance(a[i], b[i])
            d_h = hellinger_distance(a[i], b[i])
            expected = 2.0 * np.arccos(np.clip(1.0 - d_h**2, -1.0, 1.0))
            np.testing.assert_allclose(d_f, expected, atol=1e-12)

    def test_g9_kernel_distance_relation(self, rng: np.random.Generator) -> None:
        """G9: fisher_kernel(a,b) = cos(d_F(a,b)/2) within atol 1e-12."""
        a = random_simplex(5, 100, rng)
        b = random_simplex(5, 100, rng)
        for i in range(100):
            k = fisher_kernel(a[i], b[i])
            d = fisher_distance(a[i], b[i])
            expected = np.cos(d / 2.0)
            np.testing.assert_allclose(k, expected, atol=1e-12)

    def test_g10_pairwise_properties(self, rng: np.random.Generator) -> None:
        """G10: pairwise_fisher_distances has shape (M,M), symmetric, zero diag."""
        X = random_simplex(5, 15, rng)
        D = pairwise_fisher_distances(X)
        m = X.shape[0]
        assert D.shape == (m, m)
        # Symmetric
        np.testing.assert_allclose(D, D.T, atol=1e-14)
        # Zero diagonal
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-14)

    def test_g11_kernel_matrix_psd(self, rng: np.random.Generator) -> None:
        """G11: kernel_matrix is (M,M), symmetric, positive semidefinite."""
        X = random_simplex(5, 15, rng)
        K = kernel_matrix(X)
        m = X.shape[0]
        assert K.shape == (m, m)
        # Symmetric
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        # Positive semidefinite: eigenvalues >= -1e-10
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10)


# ---------------------------------------------------------------------------
# Tangent tools
# ---------------------------------------------------------------------------


class TestTangentTools:
    """Tests for logmap, expmap, tangent_map."""

    def test_logmap_expmap_roundtrip(self, rng: np.random.Generator) -> None:
        """expmap(logmap(X)) ≈ X for random simplex points."""
        X = random_simplex(4, 20, rng)
        base = fisher_mean(X)
        V = fisher_logmap(X, base=base)
        recovered = fisher_expmap(V, base=base)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_logmap_single_input(self, rng: np.random.Generator) -> None:
        """logmap works on a single (N,) input."""
        X = random_simplex(4, 10, rng)
        base = fisher_mean(X)
        v = fisher_logmap(X[0], base=base)
        assert v.ndim == 1
        assert v.shape == (4,)

    def test_expmap_single_input(self) -> None:
        """expmap works on a single (N,) input."""
        base = np.array([0.25, 0.25, 0.25, 0.25])
        v = np.zeros(4)
        result = fisher_expmap(v, base=base)
        assert result.ndim == 1
        np.testing.assert_allclose(result, base, atol=1e-14)

    def test_tangent_map_returns_tuple(self, rng: np.random.Generator) -> None:
        """tangent_map returns (V, base)."""
        X = random_simplex(4, 10, rng)
        V, base_point = tangent_map(X)
        assert V.shape == (10, 4)
        assert base_point.shape == (4,)

    def test_logmap_zero_base_warns(self) -> None:
        """logmap warns when base has zero components."""
        X = np.array([[0.5, 0.5, 0.0], [0.3, 0.4, 0.3]])
        base = np.array([0.5, 0.5, 0.0])
        with pytest.warns(UserWarning, match="zero components"):
            fisher_logmap(X, base=base)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


class TestFisherPCA:
    """Tests for fisher_pca."""

    def test_pca_returns_correct_keys(self, rng: np.random.Generator) -> None:
        """fisher_pca returns dict with correct keys."""
        X = random_simplex(5, 20, rng)
        result = fisher_pca(X)
        expected_keys = {
            "base",
            "tangent_coords",
            "components",
            "singular_values",
            "explained_variance",
            "explained_variance_ratio",
            "scores",
        }
        assert set(result.keys()) == expected_keys

    def test_pca_shapes_default(self, rng: np.random.Generator) -> None:
        """fisher_pca shapes with default n_components."""
        X = random_simplex(5, 20, rng)
        result = fisher_pca(X)
        m, n = 20, 5
        k = min(m, n)
        assert result["base"].shape == (n,)
        assert result["tangent_coords"].shape == (m, n)
        assert result["components"].shape == (k, n)
        assert result["singular_values"].shape == (k,)
        assert result["explained_variance"].shape == (k,)
        assert result["explained_variance_ratio"].shape == (k,)
        assert result["scores"].shape == (m, k)

    def test_pca_n_components(self, rng: np.random.Generator) -> None:
        """fisher_pca scores shape is (M, K) where K = n_components."""
        X = random_simplex(5, 20, rng)
        result = fisher_pca(X, n_components=3)
        assert result["scores"].shape == (20, 3)
        assert result["components"].shape == (3, 5)

    def test_pca_variance_ratio_sums(self, rng: np.random.Generator) -> None:
        """explained_variance_ratio sums to <= 1."""
        X = random_simplex(5, 50, rng)
        result = fisher_pca(X)
        total = result["explained_variance_ratio"].sum()
        assert total <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Utility geometry
# ---------------------------------------------------------------------------


class TestUtilityGeometry:
    """Tests for sample_near and perturb_simplex."""

    def test_sample_near_valid_simplex(self) -> None:
        """sample_near returns valid simplex point."""
        s = np.array([0.2, 0.3, 0.5])
        rng = np.random.default_rng(42)
        result = sample_near(s, scale=0.1, rng=rng)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)
        assert np.all(result >= 0)

    def test_perturb_simplex_fisher(self) -> None:
        """perturb_simplex with mode='fisher' returns valid simplex."""
        s = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        result = perturb_simplex(s, eps=0.05, mode="fisher", rng=rng)
        assert result.shape == (4,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)
        assert np.all(result >= 0)

    def test_perturb_simplex_euclidean(self) -> None:
        """perturb_simplex with mode='euclidean' returns valid simplex."""
        s = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        result = perturb_simplex(s, eps=0.01, mode="euclidean", rng=rng)
        assert result.shape == (4,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)
        assert np.all(result >= 0)

    def test_perturb_simplex_euclidean_large_eps(self) -> None:
        """perturb_simplex with huge eps still returns valid simplex."""
        s = np.array([0.5, 0.5])
        rng = np.random.default_rng(5)
        result = perturb_simplex(s, eps=2.0, mode="euclidean", rng=rng)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# Kernels — extras
# ---------------------------------------------------------------------------


class TestKernels:
    """Additional kernel tests."""

    def test_fisher_cosine_is_fisher_kernel(self) -> None:
        """fisher_cosine is fisher_kernel."""
        assert fisher_cosine is fisher_kernel

    def test_hellinger_rbf_kernel(self, rng: np.random.Generator) -> None:
        """kernel_matrix with kind='hellinger_rbf' and sigma parameter."""
        X = random_simplex(4, 10, rng)
        K = kernel_matrix(X, kind="hellinger_rbf", sigma=1.0)
        assert K.shape == (10, 10)
        # Symmetric
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        # Diagonal is 1 (distance to self is 0, exp(0)=1)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-14)
        # All entries in [0, 1]
        assert np.all(K >= -1e-15) and np.all(K <= 1.0 + 1e-15)

    def test_hellinger_rbf_requires_sigma(self) -> None:
        """kernel_matrix raises ValueError without sigma for hellinger_rbf."""
        X = np.array([[0.5, 0.5], [0.3, 0.7]])
        with pytest.raises(ValueError, match="sigma"):
            kernel_matrix(X, kind="hellinger_rbf")


# ---------------------------------------------------------------------------
# Polynomial Fisher kernel (paper §2.7)
# ---------------------------------------------------------------------------


class TestPolynomialFisherKernel:
    """Tests for polynomial_fisher_kernel and kernel_matrix polynomial_fisher."""

    def test_degree_1_equals_bhattacharyya(self, rng: np.random.Generator) -> None:
        """K_1(a, b) = B(a, b) for degree 1."""
        a = random_simplex(5, 50, rng)
        b = random_simplex(5, 50, rng)
        for i in range(50):
            k1 = polynomial_fisher_kernel(a[i], b[i], 1)
            bc = bhattacharyya_coefficient(a[i], b[i])
            np.testing.assert_allclose(k1, bc, atol=1e-14)

    def test_degree_2_equals_bc_squared(self, rng: np.random.Generator) -> None:
        """K_2(a, b) = B(a, b)^2."""
        a = random_simplex(5, 50, rng)
        b = random_simplex(5, 50, rng)
        for i in range(50):
            k2 = polynomial_fisher_kernel(a[i], b[i], 2)
            bc = bhattacharyya_coefficient(a[i], b[i])
            np.testing.assert_allclose(k2, bc**2, atol=1e-14)

    def test_self_kernel_is_one(self) -> None:
        """K_d(s, s) = B(s,s)^d = 1 for any d."""
        s = np.array([0.2, 0.3, 0.5])
        for d in (1, 2, 3, 5):
            np.testing.assert_allclose(
                polynomial_fisher_kernel(s, s, d), 1.0, atol=1e-14
            )

    def test_kernel_matrix_polynomial(self, rng: np.random.Generator) -> None:
        """kernel_matrix with kind='polynomial_fisher' is PSD."""
        X = random_simplex(4, 10, rng)
        K = kernel_matrix(X, kind="polynomial_fisher", d=3)
        assert K.shape == (10, 10)
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_polynomial_requires_d(self) -> None:
        """kernel_matrix raises ValueError without d for polynomial_fisher."""
        X = np.array([[0.5, 0.5], [0.3, 0.7]])
        with pytest.raises(ValueError, match="d"):
            kernel_matrix(X, kind="polynomial_fisher")

    def test_invalid_d_raises(self) -> None:
        """polynomial_fisher_kernel raises on d < 1."""
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.1, 0.6, 0.3])
        with pytest.raises(ValueError, match="positive integer"):
            polynomial_fisher_kernel(a, b, 0)
        with pytest.raises(ValueError, match="positive integer"):
            polynomial_fisher_kernel(a, b, -1)

    def test_batch_input(self, rng: np.random.Generator) -> None:
        """polynomial_fisher_kernel works with (M, N) input."""
        a = random_simplex(4, 10, rng)
        b = random_simplex(4, 10, rng)
        result = polynomial_fisher_kernel(a, b, 3)
        assert result.shape == (10,)


# ---------------------------------------------------------------------------
# Fisher RBF kernel (paper §2.7)
# ---------------------------------------------------------------------------


class TestFisherRBFKernel:
    """Tests for fisher_rbf_kernel and kernel_matrix fisher_rbf."""

    def test_self_kernel_is_one(self) -> None:
        """exp(-0^2/(2*sigma^2)) = 1."""
        s = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(fisher_rbf_kernel(s, s, 1.0), 1.0, atol=1e-14)

    def test_formula_matches_manual(self, rng: np.random.Generator) -> None:
        """fisher_rbf_kernel(a, b, sigma) = exp(-d_F^2 / (2*sigma^2))."""
        a = random_simplex(5, 50, rng)
        b = random_simplex(5, 50, rng)
        sigma = 0.7
        for i in range(50):
            k = fisher_rbf_kernel(a[i], b[i], sigma)
            d_f = fisher_distance(a[i], b[i])
            expected = np.exp(-(d_f**2) / (2.0 * sigma**2))
            np.testing.assert_allclose(k, expected, atol=1e-14)

    def test_kernel_matrix_fisher_rbf(self, rng: np.random.Generator) -> None:
        """kernel_matrix with kind='fisher_rbf' has unit diagonal and is PSD."""
        X = random_simplex(4, 10, rng)
        K = kernel_matrix(X, kind="fisher_rbf", sigma=1.0)
        assert K.shape == (10, 10)
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-14)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_fisher_rbf_requires_sigma(self) -> None:
        """kernel_matrix raises ValueError without sigma for fisher_rbf."""
        X = np.array([[0.5, 0.5], [0.3, 0.7]])
        with pytest.raises(ValueError, match="sigma"):
            kernel_matrix(X, kind="fisher_rbf")

    def test_invalid_sigma_raises(self) -> None:
        """fisher_rbf_kernel raises on sigma <= 0."""
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.1, 0.6, 0.3])
        with pytest.raises(ValueError, match="positive"):
            fisher_rbf_kernel(a, b, 0.0)
        with pytest.raises(ValueError, match="positive"):
            fisher_rbf_kernel(a, b, -1.0)

    def test_batch_input(self, rng: np.random.Generator) -> None:
        """fisher_rbf_kernel works with (M, N) input."""
        a = random_simplex(4, 10, rng)
        b = random_simplex(4, 10, rng)
        result = fisher_rbf_kernel(a, b, 0.5)
        assert result.shape == (10,)


# ---------------------------------------------------------------------------
# Geodesic — extras
# ---------------------------------------------------------------------------


class TestGeodesicExtras:
    """Additional geodesic tests."""

    def test_degenerate_case(self) -> None:
        """geodesic(a, a, t) returns a for all t."""
        a = np.array([0.2, 0.3, 0.5])
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = fisher_geodesic(a, a, t)
            np.testing.assert_allclose(result, a, atol=1e-14)

    def test_geodesic_interpolate_shape(self, rng: np.random.Generator) -> None:
        """geodesic_interpolate returns correct shape."""
        a = random_simplex(4, 1, rng)[0]
        b = random_simplex(4, 1, rng)[0]
        ts = np.linspace(0, 1, 11)
        result = geodesic_interpolate(a, b, ts)
        assert result.shape == (11, 4)


# ---------------------------------------------------------------------------
# Distances — extras
# ---------------------------------------------------------------------------


class TestDistanceExtras:
    """Additional distance tests."""

    def test_pairwise_hellinger_properties(self, rng: np.random.Generator) -> None:
        """pairwise_hellinger_distances: shape, symmetry, zero diagonal."""
        X = random_simplex(4, 10, rng)
        D = pairwise_hellinger_distances(X)
        assert D.shape == (10, 10)
        np.testing.assert_allclose(D, D.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-14)

    def test_bhattacharyya_self(self) -> None:
        """BC(s, s) = 1 for valid simplex."""
        s = np.array([0.2, 0.3, 0.5])
        bc = bhattacharyya_coefficient(s, s)
        np.testing.assert_allclose(bc, 1.0, atol=1e-14)

    def test_bhattacharyya_batch(self, rng: np.random.Generator) -> None:
        """bhattacharyya_coefficient works with (M,N) input."""
        a = random_simplex(4, 10, rng)
        b = random_simplex(4, 10, rng)
        bc = bhattacharyya_coefficient(a, b)
        assert bc.shape == (10,)


# ---------------------------------------------------------------------------
# Means — extras
# ---------------------------------------------------------------------------


class TestMeanExtras:
    """Additional mean tests."""

    def test_fisher_barycenter_is_mean(self, rng: np.random.Generator) -> None:
        """fisher_barycenter returns same as fisher_mean."""
        X = random_simplex(4, 10, rng)
        m = fisher_mean(X)
        bc = fisher_barycenter(X)
        np.testing.assert_allclose(bc, m, atol=1e-14)

    def test_fisher_mean_with_weights(self, rng: np.random.Generator) -> None:
        """fisher_mean with weights returns valid simplex."""
        X = random_simplex(4, 10, rng)
        weights = rng.random(10)
        m = fisher_mean(X, weights=weights)
        np.testing.assert_allclose(m.sum(), 1.0, atol=1e-12)
        assert np.all(m >= -1e-15)

    def test_fisher_mean_zero_weights_raises(self) -> None:
        """All-zero weights raise ValueError."""
        X = np.array([[0.5, 0.5], [0.3, 0.7]])
        with pytest.raises(ValueError, match="[Ww]eight"):
            fisher_mean(X, weights=np.array([0.0, 0.0]))

    def test_fisher_mean_negative_weights_raises(self) -> None:
        """Negative weights raise ValueError."""
        X = np.array([[0.5, 0.5], [0.3, 0.7]])
        with pytest.raises(ValueError, match="[Ww]eight"):
            fisher_mean(X, weights=np.array([-1.0, 2.0]))

    def test_fisher_mean_wrong_length_weights_raises(self) -> None:
        """Wrong-length weights raise ValueError."""
        X = np.array([[0.5, 0.5], [0.3, 0.7]])
        with pytest.raises(ValueError, match="[Ww]eight"):
            fisher_mean(X, weights=np.array([1.0]))


# ---------------------------------------------------------------------------
# 1D input rejection (batch APIs require 2D)
# ---------------------------------------------------------------------------


class TestBatchInputValidation:
    """Batch geometry APIs must reject 1D input with a clear error."""

    def test_pairwise_fisher_distances_rejects_1d(self) -> None:
        x = np.array([0.2, 0.3, 0.5])
        with pytest.raises(ValueError, match="batch"):
            pairwise_fisher_distances(x)

    def test_pairwise_hellinger_distances_rejects_1d(self) -> None:
        x = np.array([0.2, 0.3, 0.5])
        with pytest.raises(ValueError, match="batch"):
            pairwise_hellinger_distances(x)

    def test_kernel_matrix_rejects_1d(self) -> None:
        x = np.array([0.2, 0.3, 0.5])
        with pytest.raises(ValueError, match="batch"):
            kernel_matrix(x)

    def test_fisher_mean_rejects_1d(self) -> None:
        x = np.array([0.2, 0.3, 0.5])
        with pytest.raises(ValueError, match="batch"):
            fisher_mean(x)


# ---------------------------------------------------------------------------
# OnlineFisherMean
# ---------------------------------------------------------------------------


class TestOnlineFisherMean:
    """Tests for OnlineFisherMean incremental mean."""

    def test_single_update(self) -> None:
        """Single update: mean equals that single point."""
        s = np.array([0.2, 0.3, 0.5])
        ofm = OnlineFisherMean(3)
        ofm.update(s)
        np.testing.assert_allclose(ofm.mean.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(ofm.mean, s, atol=1e-12)
        assert ofm.count == 1

    def test_batch_update(self, rng: np.random.Generator) -> None:
        """update_batch produces same result as sequential updates."""
        X = random_simplex(4, 20, rng)
        # Sequential
        ofm1 = OnlineFisherMean(4)
        for row in X:
            ofm1.update(row)
        # Batch
        ofm2 = OnlineFisherMean(4)
        ofm2.update_batch(X)
        np.testing.assert_allclose(ofm1.mean, ofm2.mean, atol=1e-12)
        assert ofm1.count == ofm2.count

    def test_weighted_update(self) -> None:
        """Weighted update skews mean toward heavier point."""
        a = np.array([0.9, 0.05, 0.05])
        b = np.array([0.05, 0.9, 0.05])
        ofm = OnlineFisherMean(3)
        ofm.update(a, weight=10.0)
        ofm.update(b, weight=1.0)
        # Mean should be closer to a
        d_a = fisher_distance(ofm.mean, a)
        d_b = fisher_distance(ofm.mean, b)
        assert d_a < d_b

    def test_reset(self) -> None:
        """reset() clears state."""
        ofm = OnlineFisherMean(3)
        ofm.update(np.array([0.2, 0.3, 0.5]))
        assert ofm.count == 1
        ofm.reset()
        assert ofm.count == 0

    def test_equivalence_with_fisher_mean(self, rng: np.random.Generator) -> None:
        """Online mean approximates batch fisher_mean for uniform weights."""
        X = random_simplex(5, 50, rng)
        ofm = OnlineFisherMean(5)
        ofm.update_batch(X)
        batch_m = fisher_mean(X)
        np.testing.assert_allclose(ofm.mean, batch_m, atol=1e-12)


# ---------------------------------------------------------------------------
# WindowedFisherStats
# ---------------------------------------------------------------------------


class TestWindowedFisherStats:
    """Tests for WindowedFisherStats rolling statistics."""

    def test_push_and_count(self) -> None:
        """push increments count up to window size."""
        wfs = WindowedFisherStats(3, window_size=5)
        for i in range(3):
            s = np.array([0.2 + 0.1 * i, 0.3, 0.5 - 0.1 * i])
            s = s / s.sum()
            wfs.push(s)
        assert wfs.count == 3

    def test_window_overflow(self) -> None:
        """After filling window, count stays at window_size."""
        wfs = WindowedFisherStats(3, window_size=3)
        for i in range(10):
            s = np.array([0.2 + 0.01 * i, 0.3, 0.5 - 0.01 * i])
            s = s / s.sum()
            wfs.push(s)
        assert wfs.count == 3

    def test_mean_valid_simplex(self, rng: np.random.Generator) -> None:
        """mean is a valid simplex composition."""
        X = random_simplex(4, 10, rng)
        wfs = WindowedFisherStats(4, window_size=10)
        wfs.push_batch(X)
        m = wfs.mean
        np.testing.assert_allclose(m.sum(), 1.0, atol=1e-12)
        assert np.all(m >= -1e-15)

    def test_dispersion_nonneg(self, rng: np.random.Generator) -> None:
        """dispersion is nonnegative."""
        X = random_simplex(4, 10, rng)
        wfs = WindowedFisherStats(4, window_size=10)
        wfs.push_batch(X)
        assert wfs.dispersion >= 0

    def test_forced_pair_mean_shape(self, rng: np.random.Generator) -> None:
        """forced_pair_mean returns (2,) array."""
        X = random_simplex(4, 10, rng)
        wfs = WindowedFisherStats(4, window_size=10)
        wfs.push_batch(X)
        fp = wfs.forced_pair_mean
        assert fp.shape == (2,)

    def test_shift_from(self, rng: np.random.Generator) -> None:
        """shift_from returns a nonneg scalar."""
        X = random_simplex(4, 10, rng)
        wfs = WindowedFisherStats(4, window_size=10)
        wfs.push_batch(X)
        ref = random_simplex(4, 1, rng)[0]
        shift = wfs.shift_from(ref)
        assert np.isscalar(shift) or shift.ndim == 0
        assert shift >= 0

    def test_push_batch(self, rng: np.random.Generator) -> None:
        """push_batch produces same state as sequential pushes."""
        X = random_simplex(3, 5, rng)
        wfs1 = WindowedFisherStats(3, window_size=5)
        for row in X:
            wfs1.push(row)
        wfs2 = WindowedFisherStats(3, window_size=5)
        wfs2.push_batch(X)
        np.testing.assert_allclose(wfs1.mean, wfs2.mean, atol=1e-12)


# ---------------------------------------------------------------------------
# cross_fisher_distances
# ---------------------------------------------------------------------------


class TestCrossFisherDistances:
    """Tests for cross_fisher_distances between two clouds."""

    def test_shape(self, rng: np.random.Generator) -> None:
        """Returns (M, K) distance matrix."""
        X = random_simplex(4, 5, rng)
        Y = random_simplex(4, 8, rng)
        D = cross_fisher_distances(X, Y)
        assert D.shape == (5, 8)

    def test_symmetry(self, rng: np.random.Generator) -> None:
        """D(X, Y)[i, j] == D(Y, X)[j, i]."""
        X = random_simplex(4, 5, rng)
        Y = random_simplex(4, 3, rng)
        D_xy = cross_fisher_distances(X, Y)
        D_yx = cross_fisher_distances(Y, X)
        np.testing.assert_allclose(D_xy, D_yx.T, atol=1e-12)

    def test_zero_distance_identical(self, rng: np.random.Generator) -> None:
        """Distance to self is zero."""
        X = random_simplex(4, 5, rng)
        D = cross_fisher_distances(X, X)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-7)

    def test_nonneg(self, rng: np.random.Generator) -> None:
        """All distances are nonnegative."""
        X = random_simplex(4, 5, rng)
        Y = random_simplex(4, 3, rng)
        D = cross_fisher_distances(X, Y)
        assert np.all(D >= -1e-15)

    def test_matches_pairwise(self, rng: np.random.Generator) -> None:
        """cross_fisher_distances(X, X) matches pairwise_fisher_distances(X).

        Note: pairwise_fisher_distances forces exact zero diagonal and
        symmetry, so we use atol=1e-7 (same as G1 self-distance tolerance).
        """
        X = random_simplex(4, 6, rng)
        D_cross = cross_fisher_distances(X, X)
        D_pair = pairwise_fisher_distances(X)
        np.testing.assert_allclose(D_cross, D_pair, atol=1e-7)
