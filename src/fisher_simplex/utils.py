"""Simplex validation and projection utilities.

Provides tools for ensuring arrays lie on the probability simplex,
projecting arbitrary vectors onto it, and proportional normalization.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_simplex(
    x: ArrayLike,
    *,
    axis: int = -1,
    tol: float = 1e-12,
    renormalize: str = "warn",
) -> NDArray[np.floating]:
    """Validate and optionally renormalize a simplex composition.

    Parameters
    ----------
    x : array_like
        Input array of shape ``(N,)`` (single composition) or ``(M, N)``
        (batch of compositions).
    axis : int, optional
        Axis along which components sum to 1. Default ``-1``.
    tol : float, optional
        Tolerance for sum-to-one and negativity checks. Default ``1e-12``.
    renormalize : {"never", "warn", "always"}, optional
        - ``"never"``: reject if any row's sum differs from 1 beyond *tol*;
          raise ``ValueError``.
        - ``"warn"``: renormalize mild sum drift and emit ``warnings.warn``;
          clip entries in ``[-tol, 0)`` to 0.
        - ``"always"``: silently renormalize nonneg vectors; clip entries in
          ``[-tol, 0)`` to 0.

    Returns
    -------
    ndarray
        Validated (and possibly renormalized) simplex array.

    Raises
    ------
    ValueError
        If any entry is below ``-tol`` (regardless of mode), or if
        ``renormalize="never"`` and row sums deviate from 1 beyond *tol*.
    """
    x = np.asarray(x, dtype=np.float64)

    # Check for strongly negative entries — always an error
    if np.any(x < -tol):
        raise ValueError(
            f"Input contains negative entries below -{tol}. "
            "Cannot form a valid simplex composition."
        )

    # Clip near-zero negatives for warn/always modes
    if renormalize in ("warn", "always"):
        x = np.where((x < 0) & (x >= -tol), 0.0, x)

    sums = x.sum(axis=axis, keepdims=True)

    # Guard against zero-sum vectors (all-zero input)
    if np.any(sums == 0):
        raise ValueError(
            "Input contains rows summing to zero. "
            "Cannot form a valid simplex composition."
        )

    drift = np.abs(sums - 1.0)

    if renormalize == "never":
        if np.any(drift > tol):
            raise ValueError(
                f"Simplex sum deviates from 1 beyond tol={tol}. "
                f"Max deviation: {float(drift.max()):.2e}."
            )
        return x

    if renormalize == "warn":
        if np.any(drift > tol):
            warnings.warn(
                f"Simplex sum drift detected (max {float(drift.max()):.2e}). "
                "Renormalizing.",
                stacklevel=2,
            )
        # Only renormalize if there is drift
        if np.any(drift > 0):
            x = x / sums
        return x

    if renormalize == "always":
        x = x / sums
        return x

    raise ValueError(f"Unknown renormalize mode: {renormalize!r}")


def project_to_simplex(x: ArrayLike) -> NDArray[np.floating]:
    """Euclidean projection onto the probability simplex.

    Implements the algorithm of Duchi et al. (2008): sort in descending
    order, find the breakpoint, and threshold.

    Parameters
    ----------
    x : array_like
        Input vector(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Projected vector(s) on the simplex, same shape as input.
    """
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]

    n = x.shape[-1]
    # Sort descending along last axis
    u = np.sort(x, axis=-1)[:, ::-1]
    cumsum = np.cumsum(u, axis=-1)
    k_range = np.arange(1, n + 1)
    # rho: largest k where u_k - (cumsum_k - 1) / k > 0
    condition = u - (cumsum - 1.0) / k_range > 0
    # Find the last True index along each row
    rho = n - np.argmax(condition[:, ::-1], axis=-1)
    # Gather cumsum at rho indices
    rho_idx = (rho - 1)[:, np.newaxis]
    cumsum_at_rho = np.take_along_axis(cumsum, rho_idx, axis=-1)
    theta = (cumsum_at_rho - 1.0) / rho[:, np.newaxis]
    result = np.maximum(x - theta, 0.0)

    if single:
        return result[0]
    return result


def closure(x: ArrayLike) -> NDArray[np.floating]:
    """Proportional normalization (closure operation).

    Divides each vector by its sum along the last axis, mapping
    nonnegative vectors to the probability simplex.

    Parameters
    ----------
    x : array_like
        Nonnegative input of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Normalized array summing to 1 along the last axis.

    Raises
    ------
    ValueError
        If any row sums to zero.
    """
    x = np.asarray(x, dtype=np.float64)
    sums = x.sum(axis=-1, keepdims=True)
    if np.any(sums == 0):
        raise ValueError(
            "Input contains rows summing to zero. Cannot normalize."
        )
    return x / sums
