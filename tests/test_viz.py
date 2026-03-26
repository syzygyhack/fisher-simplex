"""Tests for the viz module — V-series per spec section 4.7."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tests.conftest import random_simplex

# Lazy-import guard: matplotlib must not be loaded by importing viz
plt = pytest.importorskip("matplotlib.pyplot")
import matplotlib  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


class TestNoModuleLevelImport:
    """V0: matplotlib must not be imported at module level."""

    def test_no_matplotlib_at_import(self):
        """V0: Importing fisher_simplex.viz must not pull in matplotlib."""
        # Remove matplotlib and viz from sys.modules to test fresh import
        mpl_mods = [k for k in sys.modules if k.startswith("matplotlib")]
        viz_mods = [k for k in sys.modules if "viz" in k and "fisher" in k]
        saved = {}
        for k in mpl_mods + viz_mods:
            saved[k] = sys.modules.pop(k)

        try:
            import importlib

            import fisher_simplex.viz as viz_mod

            importlib.reload(viz_mod)
            assert not any(
                k.startswith("matplotlib") for k in sys.modules if k not in saved
            ), "matplotlib was imported at module level in viz"
        finally:
            # Restore modules
            sys.modules.update(saved)


class TestScatterForcedPair:
    """V1: scatter_forced_pair smoke tests."""

    def test_returns_axes(self, rng: np.random.Generator):
        """V1a: Returns Axes for 50 random 5-component compositions."""
        from fisher_simplex.viz import scatter_forced_pair

        X = random_simplex(5, 50, rng)
        ax = scatter_forced_pair(X)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_color(self, rng: np.random.Generator):
        """V1b: Accepts color parameter."""
        from fisher_simplex.viz import scatter_forced_pair

        X = random_simplex(5, 50, rng)
        ax = scatter_forced_pair(X, color="red")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_existing_ax(self, rng: np.random.Generator):
        """V1c: Accepts existing Axes."""
        from fisher_simplex.viz import scatter_forced_pair

        X = random_simplex(5, 50, rng)
        fig, ax_in = plt.subplots()
        ax_out = scatter_forced_pair(X, ax=ax_in)
        assert ax_out is ax_in


class TestDivergenceHeatmap:
    """V2: divergence_heatmap smoke tests."""

    def test_returns_axes(self):
        """V2a: Returns Axes for n_range=[3, 5, 10]."""
        from fisher_simplex.viz import divergence_heatmap

        ax = divergence_heatmap([3, 5, 10])
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_existing_ax(self):
        """V2b: Accepts existing Axes."""
        from fisher_simplex.viz import divergence_heatmap

        fig, ax_in = plt.subplots()
        ax_out = divergence_heatmap([3, 5, 10], ax=ax_in)
        assert ax_out is ax_in


class TestFisherSphere:
    """V3: fisher_sphere smoke tests."""

    def test_returns_axes_3d(self, rng: np.random.Generator):
        """V3a: Returns Axes for 20 random 3-component compositions."""
        from fisher_simplex.viz import fisher_sphere

        X = random_simplex(3, 20, rng)
        ax = fisher_sphere(X)
        # mpl_toolkits.mplot3d.axes3d.Axes3D is a subclass of Axes
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_raises_for_non_3(self, rng: np.random.Generator):
        """V3b: Raises ValueError for N != 3."""
        from fisher_simplex.viz import fisher_sphere

        X = random_simplex(5, 20, rng)
        with pytest.raises(ValueError, match="N=3"):
            fisher_sphere(X)

    def test_with_existing_ax(self, rng: np.random.Generator):
        """V3c: Accepts existing 3D Axes."""
        from fisher_simplex.viz import fisher_sphere

        X = random_simplex(3, 20, rng)
        fig = plt.figure()
        ax_in = fig.add_subplot(111, projection="3d")
        ax_out = fisher_sphere(X, ax=ax_in)
        assert ax_out is ax_in


class TestDisagreementCurve:
    """V4: disagreement_curve smoke tests."""

    def test_returns_axes(self):
        """V4a: Returns Axes for n_range=[3, 5, 10]."""
        from fisher_simplex.viz import disagreement_curve

        ax = disagreement_curve([3, 5, 10])
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_existing_ax(self):
        """V4b: Accepts existing Axes."""
        from fisher_simplex.viz import disagreement_curve

        fig, ax_in = plt.subplots()
        ax_out = disagreement_curve([3, 5, 10], ax=ax_in)
        assert ax_out is ax_in


class TestForcedPairHistogram:
    """V5: forced_pair_histogram smoke tests."""

    def test_returns_axes(self, rng: np.random.Generator):
        """V5a: Returns Axes for 50 random compositions."""
        from fisher_simplex.viz import forced_pair_histogram

        X = random_simplex(5, 50, rng)
        ax = forced_pair_histogram(X)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_existing_ax(self, rng: np.random.Generator):
        """V5b: Accepts existing Axes."""
        from fisher_simplex.viz import forced_pair_histogram

        X = random_simplex(5, 50, rng)
        fig, ax_in = plt.subplots()
        ax_out = forced_pair_histogram(X, ax=ax_in)
        assert ax_out is ax_in


class TestGeodesicPlot:
    """V6: geodesic_plot smoke tests."""

    def test_returns_axes(self, rng: np.random.Generator):
        """V6a: Returns Axes for two random 4-component compositions."""
        from fisher_simplex.viz import geodesic_plot

        a = rng.dirichlet(np.ones(4))
        b = rng.dirichlet(np.ones(4))
        ax = geodesic_plot(a, b)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_ternary_n3(self, rng: np.random.Generator):
        """V6b: N=3 uses ternary coordinates."""
        from fisher_simplex.viz import geodesic_plot

        a = rng.dirichlet(np.ones(3))
        b = rng.dirichlet(np.ones(3))
        ax = geodesic_plot(a, b)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_with_existing_ax(self, rng: np.random.Generator):
        """V6c: Accepts existing Axes."""
        from fisher_simplex.viz import geodesic_plot

        a = rng.dirichlet(np.ones(4))
        b = rng.dirichlet(np.ones(4))
        fig, ax_in = plt.subplots()
        ax_out = geodesic_plot(a, b, ax=ax_in)
        assert ax_out is ax_in
