"""Visualization utilities for Fisher-simplex analysis.

Matplotlib-based plotting for simplex compositions, forced-pair
coordinates, geodesics, and overlap divergence as specified in
spec section 4.7.

**Design rule:** matplotlib is never imported at module level.
Every function performs a lazy import inside its body, keeping
the core library lightweight for users who do not need plots.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def scatter_forced_pair(X: ArrayLike, *, color=None, ax=None):
    """Scatter plot of data in forced-pair (Q_delta, H_3) space.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    color : color-like, optional
        Matplotlib color for the scatter points.
    ax : matplotlib Axes, optional
        Axes to plot on. If ``None``, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from fisher_simplex.core import forced_coordinates

    coords = forced_coordinates(np.asarray(X, dtype=np.float64))
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    if ax is None:
        _fig, ax = plt.subplots()

    kwargs = {}
    if color is not None:
        kwargs["color"] = color

    ax.scatter(coords[:, 0], coords[:, 1], **kwargs)
    ax.set_xlabel(r"$Q_\Delta$")
    ax.set_ylabel(r"$H_3$")
    return ax


def divergence_heatmap(n_range, *, ax=None):
    """Bar chart of mean overlap divergence |Phi - Psi| vs dimensionality.

    For each N in *n_range*, generates 200 Dirichlet(1) samples and
    computes the mean absolute overlap divergence.

    Parameters
    ----------
    n_range : iterable of int
        Simplex dimensionalities to evaluate.
    ax : matplotlib Axes, optional
        Axes to plot on. If ``None``, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from fisher_simplex.core import overlap_divergence

    rng = np.random.default_rng(0)
    ns = list(n_range)
    means = []
    for n in ns:
        samples = rng.dirichlet(np.ones(n), size=200)
        divs = np.abs(overlap_divergence(samples))
        means.append(float(np.mean(divs)))

    if ax is None:
        _fig, ax = plt.subplots()

    ax.bar([str(n) for n in ns], means)
    ax.set_xlabel("N")
    ax.set_ylabel(r"Mean $|\Phi - \Psi|$")
    ax.set_title("Overlap divergence vs dimensionality")
    return ax


def fisher_sphere(X: ArrayLike, *, ax=None):
    """3D scatter of Fisher-lifted data on the unit sphere S^2.

    Only valid for N=3 compositions. Raises ``ValueError`` otherwise.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, 3)``.
    ax : matplotlib Axes3D, optional
        3D axes to plot on. If ``None``, a new 3D figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from fisher_simplex.core import fisher_lift

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[-1] != 3:
        msg = f"fisher_sphere requires N=3 compositions, got N={X.shape[-1]}"
        raise ValueError(msg)

    lifted = fisher_lift(X)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Reference sphere wireframe
    u = np.linspace(0, np.pi / 2, 30)
    v = np.linspace(0, np.pi / 2, 30)
    u, v = np.meshgrid(u, v)
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, alpha=0.1, color="gray")

    # Data points
    ax.scatter(lifted[:, 0], lifted[:, 1], lifted[:, 2])
    ax.set_xlabel(r"$\sqrt{s_1}$")
    ax.set_ylabel(r"$\sqrt{s_2}$")
    ax.set_zlabel(r"$\sqrt{s_3}$")
    return ax


def disagreement_curve(n_range, *, ax=None):
    """Line plot of pairwise ranking disagreement between Phi and Psi vs N.

    For each N in *n_range*, generates 100 Dirichlet(1) samples and
    computes the pairwise ranking disagreement rate.

    Parameters
    ----------
    n_range : iterable of int
        Simplex dimensionalities to evaluate.
    ax : matplotlib Axes, optional
        Axes to plot on. If ``None``, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from fisher_simplex.analysis import pairwise_ranking_disagreement
    from fisher_simplex.core import phi, psi

    rng = np.random.default_rng(0)
    ns = list(n_range)
    rates = []
    for n in ns:
        samples = rng.dirichlet(np.ones(n), size=100)
        phi_vals = phi(samples)
        psi_vals = psi(samples)
        result = pairwise_ranking_disagreement(phi_vals, psi_vals)
        rates.append(result["disagreement_rate"])

    if ax is None:
        _fig, ax = plt.subplots()

    ax.plot(ns, rates, marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel("Disagreement rate")
    ax.set_title(r"$\Phi$ vs $\Psi$ ranking disagreement")
    return ax


def forced_pair_histogram(X: ArrayLike, *, ax=None):
    """Side-by-side histograms of Q_delta and H_3 distributions.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    ax : matplotlib Axes, optional
        Axes to plot on. If ``None``, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from fisher_simplex.core import forced_coordinates

    coords = forced_coordinates(np.asarray(X, dtype=np.float64))
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    if ax is None:
        _fig, ax = plt.subplots()

    ax.hist(coords[:, 0], alpha=0.6, label=r"$Q_\Delta$")
    ax.hist(coords[:, 1], alpha=0.6, label=r"$H_3$")
    ax.legend()
    ax.set_title("Forced-pair coordinate distributions")
    return ax


def geodesic_plot(a: ArrayLike, b: ArrayLike, *, n_points: int = 50, ax=None):
    """Plot the geodesic path between two simplex compositions.

    For N=3, uses ternary (barycentric) coordinates. For N>3, projects
    to the first two principal coordinates via Fisher PCA.

    Parameters
    ----------
    a, b : array_like
        Simplex compositions of shape ``(N,)``.
    n_points : int, optional
        Number of interpolation points. Default ``50``.
    ax : matplotlib Axes, optional
        Axes to plot on. If ``None``, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from fisher_simplex.geometry import fisher_pca, geodesic_interpolate

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, n_points)
    path = geodesic_interpolate(a, b, ts)

    n = a.shape[-1]

    if ax is None:
        _fig, ax = plt.subplots()

    if n == 3:
        # Ternary (barycentric) projection to 2D
        # x = s2 + s3/2, y = s3 * sqrt(3)/2
        x = path[:, 1] + path[:, 2] / 2.0
        y = path[:, 2] * np.sqrt(3.0) / 2.0
        ax.plot(x, y, "-o", markersize=2)
        # Mark endpoints
        ax.plot(x[0], y[0], "go", markersize=8, label="start")
        ax.plot(x[-1], y[-1], "rs", markersize=8, label="end")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title("Geodesic path (ternary)")
    else:
        # Project to first 2 principal coordinates
        pca = fisher_pca(path, n_components=2)
        scores = pca["scores"]
        ax.plot(scores[:, 0], scores[:, 1], "-o", markersize=2)
        ax.plot(scores[0, 0], scores[0, 1], "go", markersize=8, label="start")
        ax.plot(scores[-1, 0], scores[-1, 1], "rs", markersize=8, label="end")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.set_title("Geodesic path (PCA projection)")

    return ax
