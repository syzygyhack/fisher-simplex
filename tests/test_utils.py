"""Tests for fisher_simplex.utils module."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fisher_simplex.utils import closure, project_to_simplex, validate_simplex

# ---------------------------------------------------------------------------
# validate_simplex
# ---------------------------------------------------------------------------


class TestValidateSimplexNever:
    """renormalize='never' mode: reject anything not summing to 1."""

    def test_valid_input_passes(self) -> None:
        x = np.array([0.2, 0.3, 0.5])
        result = validate_simplex(x, renormalize="never")
        np.testing.assert_allclose(result, x)

    def test_invalid_sum_raises(self) -> None:
        x = np.array([0.2, 0.3, 0.6])  # sums to 1.1
        with pytest.raises(ValueError, match="sum"):
            validate_simplex(x, renormalize="never")

    def test_batch_valid(self) -> None:
        x = np.array([[0.5, 0.5], [0.3, 0.7]])
        result = validate_simplex(x, renormalize="never")
        np.testing.assert_allclose(result, x)

    def test_batch_invalid_raises(self) -> None:
        x = np.array([[0.5, 0.5], [0.3, 0.6]])  # second row sums to 0.9
        with pytest.raises(ValueError, match="sum"):
            validate_simplex(x, renormalize="never")


class TestValidateSimplexWarn:
    """renormalize='warn' mode: renormalize mild drift with warning."""

    def test_mild_drift_warns_and_renormalizes(self) -> None:
        x = np.array([0.5, 0.5 + 1e-10])  # mild drift
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_simplex(x, renormalize="warn")
            assert len(w) == 1
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_valid_input_no_warning(self) -> None:
        x = np.array([0.25, 0.25, 0.25, 0.25])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_simplex(x, renormalize="warn")
            assert len(w) == 0
        np.testing.assert_allclose(result, x)

    def test_clips_near_zero_negatives(self) -> None:
        x = np.array([1.0 + 1e-13, -1e-13])  # entry in [-tol, 0)
        result = validate_simplex(x, renormalize="warn")
        assert np.all(result >= 0)


class TestValidateSimplexAlways:
    """renormalize='always' mode: silent renormalization."""

    def test_silent_renormalize(self) -> None:
        x = np.array([0.5, 0.6])  # sums to 1.1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_simplex(x, renormalize="always")
            assert len(w) == 0
        np.testing.assert_allclose(result.sum(), 1.0)
        assert np.all(result >= 0)

    def test_clips_near_zero_negatives(self) -> None:
        x = np.array([1.0 + 1e-13, -1e-13])
        result = validate_simplex(x, renormalize="always")
        assert np.all(result >= 0)


class TestValidateSimplexEdgeCases:
    """Edge cases common to all modes."""

    def test_below_neg_tol_raises(self) -> None:
        """Entries below -tol always raise ValueError regardless of mode."""
        x = np.array([1.1, -0.1])  # -0.1 well below -tol
        for mode in ("never", "warn", "always"):
            with pytest.raises(ValueError, match="negative"):
                validate_simplex(x, renormalize=mode)

    def test_1d_shape(self) -> None:
        x = np.array([0.3, 0.7])
        result = validate_simplex(x, renormalize="warn")
        assert result.ndim == 1

    def test_2d_shape(self) -> None:
        x = np.array([[0.3, 0.7], [0.5, 0.5]])
        result = validate_simplex(x, renormalize="warn")
        assert result.ndim == 2
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# project_to_simplex
# ---------------------------------------------------------------------------


class TestProjectToSimplex:
    """Euclidean projection onto the probability simplex."""

    def test_already_on_simplex(self) -> None:
        x = np.array([0.2, 0.3, 0.5])
        result = project_to_simplex(x)
        np.testing.assert_allclose(result, x, atol=1e-14)

    def test_negative_entries(self) -> None:
        x = np.array([-0.5, 0.5, 1.0])
        result = project_to_simplex(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-14)
        assert np.all(result >= -1e-15)

    def test_all_equal(self) -> None:
        x = np.array([1.0, 1.0, 1.0])
        result = project_to_simplex(x)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-14)

    def test_single_large(self) -> None:
        x = np.array([10.0, 0.0, 0.0])
        result = project_to_simplex(x)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-14)

    def test_batch(self) -> None:
        x = np.array([[1.0, 1.0, 1.0], [10.0, 0.0, 0.0]])
        result = project_to_simplex(x)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-14)
        assert np.all(result >= -1e-15)


# ---------------------------------------------------------------------------
# closure
# ---------------------------------------------------------------------------


class TestClosure:
    """Proportional normalization."""

    def test_basic(self) -> None:
        x = np.array([2.0, 3.0, 5.0])
        result = closure(x)
        np.testing.assert_allclose(result, [0.2, 0.3, 0.5])

    def test_already_normalized(self) -> None:
        x = np.array([0.25, 0.75])
        result = closure(x)
        np.testing.assert_allclose(result, x)

    def test_batch(self) -> None:
        x = np.array([[1.0, 1.0], [3.0, 1.0]])
        result = closure(x)
        np.testing.assert_allclose(result, [[0.5, 0.5], [0.75, 0.25]])

    def test_sums_to_one(self) -> None:
        x = np.array([7.0, 3.0, 11.0, 2.0])
        result = closure(x)
        np.testing.assert_allclose(result.sum(), 1.0)
