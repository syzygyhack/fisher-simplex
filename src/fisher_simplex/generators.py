"""Synthetic reference ensembles for tests, tutorials, and benchmarks.

Provides generators that produce valid simplex arrays of shape ``(m, n)``
for a variety of distributional archetypes (spec section 4.4).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _default_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return *rng* if given, otherwise a fresh default Generator."""
    if rng is None:
        return np.random.default_rng()
    return rng


# ---------------------------------------------------------------------------
# 4.4.1 — Dirichlet community
# ---------------------------------------------------------------------------


def dirichlet_community(
    n: int,
    m: int,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Draw *m* samples from a symmetric Dirichlet distribution.

    Synthetic benchmark.

    Parameters
    ----------
    n : int
        Number of components.
    m : int
        Number of samples (rows).
    alpha : float, optional
        Concentration parameter (same for all components). Default ``1.0``.
    rng : numpy.random.Generator or None, optional
        Random generator. ``None`` creates a fresh default.

    Returns
    -------
    ndarray
        Shape ``(m, n)`` simplex array.
    """
    rng = _default_rng(rng)
    return rng.dirichlet([alpha] * n, size=m)


# ---------------------------------------------------------------------------
# 4.4.2 — Broken stick
# ---------------------------------------------------------------------------


def broken_stick(
    n: int,
    m: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """MacArthur's broken-stick model.

    Synthetic benchmark.

    Generate *n - 1* uniform breaks in ``[0, 1]``, sort, and compute gaps.
    Components are sorted descending by convention.

    Parameters
    ----------
    n : int
        Number of components.
    m : int
        Number of samples (rows).
    rng : numpy.random.Generator or None, optional
        Random generator. ``None`` creates a fresh default.

    Returns
    -------
    ndarray
        Shape ``(m, n)`` simplex array with descending-sorted rows.
    """
    rng = _default_rng(rng)
    # n-1 uniform breaks in [0, 1]
    breaks = rng.uniform(0.0, 1.0, size=(m, n - 1))
    breaks.sort(axis=-1)
    # Gaps: prepend 0 and append 1, then diff
    zeros = np.zeros((m, 1))
    ones = np.ones((m, 1))
    padded = np.concatenate([zeros, breaks, ones], axis=-1)
    gaps = np.diff(padded, axis=-1)
    # Sort descending by convention
    gaps.sort(axis=-1)
    return gaps[:, ::-1].copy()


# ---------------------------------------------------------------------------
# 4.4.3 — Lognormal community
# ---------------------------------------------------------------------------


def lognormal_community(
    n: int,
    m: int,
    *,
    mu: float = 2.0,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Draw lognormal abundances, then close to the simplex.

    Synthetic benchmark.

    Parameters
    ----------
    n : int
        Number of components.
    m : int
        Number of samples (rows).
    mu : float, optional
        Mean of the underlying normal distribution. Default ``2.0``.
    sigma : float, optional
        Std dev of the underlying normal distribution. Default ``1.0``.
    rng : numpy.random.Generator or None, optional
        Random generator. ``None`` creates a fresh default.

    Returns
    -------
    ndarray
        Shape ``(m, n)`` simplex array.
    """
    rng = _default_rng(rng)
    raw = rng.lognormal(mean=mu, sigma=sigma, size=(m, n))
    return raw / raw.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 4.4.4 — Geometric series
# ---------------------------------------------------------------------------


def geometric_series(
    n: int,
    m: int,
    *,
    k_range: tuple[float, float] = (0.3, 0.8),
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Geometric-series community model.

    Synthetic benchmark.

    For each sample, draw dominance ``k ~ Uniform(k_range)``, generate
    ``s_i ∝ k * (1 - k)^(i-1)``, normalize, and randomly permute.

    Parameters
    ----------
    n : int
        Number of components.
    m : int
        Number of samples (rows).
    k_range : tuple of float, optional
        ``(low, high)`` for the uniform draw of dominance *k*.
        Default ``(0.3, 0.8)``.
    rng : numpy.random.Generator or None, optional
        Random generator. ``None`` creates a fresh default.

    Returns
    -------
    ndarray
        Shape ``(m, n)`` simplex array.
    """
    rng = _default_rng(rng)
    k = rng.uniform(k_range[0], k_range[1], size=(m, 1))
    i = np.arange(n)  # 0, 1, ..., n-1
    raw = k * (1.0 - k) ** i  # (m, n) via broadcast
    normalized = raw / raw.sum(axis=-1, keepdims=True)
    # Randomly permute component order per sample
    result = np.empty_like(normalized)
    for row in range(m):
        result[row] = rng.permutation(normalized[row])
    return result


# ---------------------------------------------------------------------------
# 4.4.5 — Market shares
# ---------------------------------------------------------------------------


def market_shares(
    n: int,
    m: int,
    *,
    concentration: str = "mixed",
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Simulate market-share distributions at varying concentration.

    Synthetic benchmark.

    Parameters
    ----------
    n : int
        Number of components.
    m : int
        Number of samples (rows).
    concentration : {"low", "medium", "high", "mixed"}, optional
        Controls Dirichlet alpha:

        - ``"low"``: alpha >= 2 (dispersed).
        - ``"medium"``: alpha ≈ 1 (uniform Dirichlet).
        - ``"high"``: alpha < 0.5 (concentrated).
        - ``"mixed"``: random mix of the above.

        Default ``"mixed"``.
    rng : numpy.random.Generator or None, optional
        Random generator. ``None`` creates a fresh default.

    Returns
    -------
    ndarray
        Shape ``(m, n)`` simplex array.
    """
    rng = _default_rng(rng)
    _alpha_map = {
        "low": lambda: rng.uniform(2.0, 5.0),
        "medium": lambda: rng.uniform(0.8, 1.2),
        "high": lambda: rng.uniform(0.05, 0.5),
    }
    if concentration == "mixed":
        levels = rng.choice(["low", "medium", "high"], size=m)
        rows = np.empty((m, n))
        for i, lvl in enumerate(levels):
            alpha = _alpha_map[lvl]()
            rows[i] = rng.dirichlet([alpha] * n)
        return rows

    if concentration not in _alpha_map:
        raise ValueError(
            f"Unknown concentration: {concentration!r}. "
            "Choose from 'low', 'medium', 'high', 'mixed'."
        )
    alpha_fn = _alpha_map[concentration]
    rows = np.empty((m, n))
    for i in range(m):
        alpha = alpha_fn()
        rows[i] = rng.dirichlet([alpha] * n)
    return rows


# ---------------------------------------------------------------------------
# 4.4.6 — Portfolio weights
# ---------------------------------------------------------------------------


def portfolio_weights(
    n: int,
    m: int,
    *,
    style: str = "mixed",
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Simulate portfolio weight distributions.

    Synthetic benchmark.

    Parameters
    ----------
    n : int
        Number of components.
    m : int
        Number of samples (rows).
    style : {"equal", "concentrated", "diversified", "mixed"}, optional
        Controls the weight distribution shape:

        - ``"equal"``: near-uniform with small noise.
        - ``"concentrated"``: few dominant weights.
        - ``"diversified"``: moderate spread.
        - ``"mixed"``: random mix of styles.

        Default ``"mixed"``.
    rng : numpy.random.Generator or None, optional
        Random generator. ``None`` creates a fresh default.

    Returns
    -------
    ndarray
        Shape ``(m, n)`` simplex array.
    """
    rng = _default_rng(rng)

    def _equal() -> NDArray[np.floating]:
        """Near-uniform with small noise."""
        base = np.full(n, 1.0 / n)
        noise = rng.normal(0, 0.01 / n, size=n)
        raw = np.maximum(base + noise, 0.0)
        return raw / raw.sum()

    def _concentrated() -> NDArray[np.floating]:
        """Few dominant weights via low-alpha Dirichlet."""
        return rng.dirichlet([0.1] * n)

    def _diversified() -> NDArray[np.floating]:
        """Moderate spread via mid-alpha Dirichlet."""
        return rng.dirichlet([2.0] * n)

    _style_map = {
        "equal": _equal,
        "concentrated": _concentrated,
        "diversified": _diversified,
    }

    if style == "mixed":
        styles = rng.choice(["equal", "concentrated", "diversified"], size=m)
        rows = np.empty((m, n))
        for i, s in enumerate(styles):
            rows[i] = _style_map[s]()
        return rows

    if style not in _style_map:
        raise ValueError(
            f"Unknown style: {style!r}. "
            "Choose from 'equal', 'concentrated', 'diversified', 'mixed'."
        )
    rows = np.empty((m, n))
    for i in range(m):
        rows[i] = _style_map[style]()
    return rows
