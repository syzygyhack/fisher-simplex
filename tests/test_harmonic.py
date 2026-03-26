"""Tests for fisher_simplex.harmonic module."""

from __future__ import annotations

import numpy as np
import pytest

from fisher_simplex.harmonic import (
    _partition_count,
    enrichment_space,
    filtration_decomposition,
    project_to_degree,
    selective_frontier,
    symmetric_even_basis,
    symmetric_even_dimension,
)

# ---------------------------------------------------------------------------
# _partition_count known values
# ---------------------------------------------------------------------------


class TestPartitionCount:
    """Validate _partition_count against known partition numbers."""

    def test_p_0_any(self) -> None:
        """P(0, N) = 1 for any N."""
        for n in (1, 3, 5, 10):
            assert _partition_count(0, n) == 1

    def test_p_1_any(self) -> None:
        """P(1, N) = 1 for any N >= 1."""
        for n in (1, 3, 5, 10):
            assert _partition_count(1, n) == 1

    def test_p_2_3(self) -> None:
        assert _partition_count(2, 3) == 2

    def test_p_3_3(self) -> None:
        assert _partition_count(3, 3) == 3

    def test_p_4_3(self) -> None:
        assert _partition_count(4, 3) == 4

    def test_p_5_5(self) -> None:
        assert _partition_count(5, 5) == 7

    def test_negative_k(self) -> None:
        """P(k, N) = 0 for k < 0."""
        assert _partition_count(-1, 5) == 0


# ---------------------------------------------------------------------------
# symmetric_even_dimension known values
# ---------------------------------------------------------------------------


class TestSymmetricEvenDimension:
    """Validate dimension formula against spec section 4.6.3."""

    def test_dim_n3_degree4(self) -> None:
        """dim(N=3, degree=4) = P(2,3) - P(1,3) = 2 - 1 = 1."""
        assert symmetric_even_dimension(3, 4) == 1

    def test_dim_n3_degree6(self) -> None:
        """dim(N=3, degree=6) = P(3,3) - P(2,3) = 3 - 2 = 1."""
        assert symmetric_even_dimension(3, 6) == 1

    def test_dim_n4_degree8(self) -> None:
        """dim(N=4, degree=8) = P(4,4) - P(3,4) = 5 - 3 = 2."""
        assert symmetric_even_dimension(4, 8) == 2

    def test_dim_degree0(self) -> None:
        """dim(N=any, degree=0) = 1 (constant)."""
        for n in (2, 3, 5, 10):
            assert symmetric_even_dimension(n, 0) == 1

    def test_dim_degree2(self) -> None:
        """dim(N=any, degree=2) = 0 (no nonconstant symmetric-even at degree 2)."""
        for n in (2, 3, 5, 10):
            assert symmetric_even_dimension(n, 2) == 0

    def test_odd_degree_raises(self) -> None:
        """Odd degree raises ValueError."""
        with pytest.raises(ValueError, match="even"):
            symmetric_even_dimension(3, 3)
        with pytest.raises(ValueError, match="even"):
            symmetric_even_dimension(5, 7)


# ---------------------------------------------------------------------------
# symmetric_even_basis
# ---------------------------------------------------------------------------


class TestSymmetricEvenBasis:
    """Validate basis construction and element structure."""

    @pytest.mark.parametrize(
        ("n", "degree"),
        [(3, 0), (3, 4), (3, 6), (4, 8), (5, 8)],
    )
    def test_basis_count_matches_dimension(self, n: int, degree: int) -> None:
        """Number of basis elements matches dimension formula."""
        basis = symmetric_even_basis(n, degree)
        expected = symmetric_even_dimension(n, degree)
        assert len(basis) == expected

    def test_degree2_empty(self) -> None:
        """Degree 2 returns empty basis."""
        assert symmetric_even_basis(3, 2) == []

    @pytest.mark.parametrize("degree", [0, 4, 6, 8])
    def test_precomputed_retrieval(self, degree: int) -> None:
        """Precomputed basis retrieval succeeds for degrees 0, 4, 6, 8."""
        basis = symmetric_even_basis(5, degree, method="precomputed")
        assert isinstance(basis, list)

    def test_precomputed_raises_above_8(self) -> None:
        """method='precomputed' raises for degree > 8."""
        with pytest.raises(ValueError, match="precomputed"):
            symmetric_even_basis(5, 10, method="precomputed")

    def test_basis_element_structure(self) -> None:
        """Each basis element has required keys."""
        basis = symmetric_even_basis(4, 4)
        for elem in basis:
            assert "coefficients" in elem
            assert "degree" in elem
            assert "description" in elem
            assert isinstance(elem["coefficients"], dict)
            assert elem["degree"] == 4

    def test_degree8_elements_have_degree_8(self) -> None:
        """Degree 8 basis elements report degree 8."""
        basis = symmetric_even_basis(4, 8)
        for elem in basis:
            assert elem["degree"] == 8


# ---------------------------------------------------------------------------
# selective_frontier
# ---------------------------------------------------------------------------


class TestSelectiveFrontier:
    """Validate selective_frontier info."""

    @pytest.mark.parametrize("n", [4, 5, 10])
    def test_enrichment_dimension_2(self, n: int) -> None:
        """enrichment_dimension = 2 for N >= 4."""
        info = selective_frontier(n)
        assert info["enrichment_dimension"] == 2

    def test_degree_is_8(self) -> None:
        info = selective_frontier(5)
        assert info["degree"] == 8

    def test_keys_present(self) -> None:
        info = selective_frontier(5)
        for key in (
            "degree",
            "total_dimension",
            "forced_dimension",
            "enrichment_dimension",
            "description",
        ):
            assert key in info

    def test_dimension_consistency(self) -> None:
        """total = forced + enrichment."""
        for n in (3, 4, 5, 10):
            info = selective_frontier(n)
            assert (
                info["total_dimension"]
                == info["forced_dimension"] + info["enrichment_dimension"]
            )

    def test_n3_enrichment_dimension_1(self) -> None:
        """For N=3, enrichment is 1 (not 2)."""
        info = selective_frontier(3)
        assert info["enrichment_dimension"] == 1


# ---------------------------------------------------------------------------
# enrichment_space
# ---------------------------------------------------------------------------


class TestEnrichmentSpace:
    """Validate enrichment_space output."""

    @pytest.mark.parametrize("n", [4, 5, 10])
    def test_degree8_length_2(self, n: int) -> None:
        """enrichment_space(N, 8) returns list of length 2 for N >= 4."""
        basis = enrichment_space(n, 8)
        assert isinstance(basis, list)
        assert len(basis) == 2

    def test_degree8_n3_length_1(self) -> None:
        """enrichment_space(N=3, 8) returns 1 element."""
        basis = enrichment_space(3, 8)
        assert len(basis) == 1

    def test_elements_have_correct_degree(self) -> None:
        """Enrichment basis elements at degree 8 report degree 8."""
        basis = enrichment_space(5, 8)
        for elem in basis:
            assert elem["degree"] == 8


# ---------------------------------------------------------------------------
# Experimental stubs
# ---------------------------------------------------------------------------


class TestExperimentalStubs:
    """project_to_degree and filtration_decomposition raise."""

    def test_project_to_degree_raises(self) -> None:
        f = np.ones(10)
        with pytest.raises(NotImplementedError):
            project_to_degree(f, 4)

    def test_filtration_decomposition_raises(self) -> None:
        f = np.ones(10)
        with pytest.raises(NotImplementedError):
            filtration_decomposition(f)
