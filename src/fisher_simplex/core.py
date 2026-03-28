"""Core simplex invariants and transforms.

Implements the Fisher amplitude lift, overlap families, forced-pair
invariants, bridge statistics, and binary helpers as specified in
spec sections 4.1.1 through 4.1.7.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fisher_simplex.utils import validate_simplex

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validated(x: ArrayLike) -> NDArray[np.floating]:
    """Validate and return a simplex array."""
    return validate_simplex(x, renormalize="warn")


def _n(x: NDArray) -> int:
    """Number of components (last axis length)."""
    return x.shape[-1]


# ---------------------------------------------------------------------------
# 4.1.5 — Fisher amplitude lift
# ---------------------------------------------------------------------------


def fisher_lift(s: ArrayLike) -> NDArray[np.floating]:
    """Fisher amplitude lift: psi_i = sqrt(s_i).

    Maps a simplex composition to the positive orthant of the unit
    sphere S^{N-1} (the Fisher--Rao embedding).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Amplitude vectors on S^{N-1}, same shape as input.
    """
    s = _validated(s)
    return np.sqrt(s)


def fisher_project(psi: ArrayLike) -> NDArray[np.floating]:
    """Project amplitudes back to the simplex: s_i = psi_i^2.

    Parameters
    ----------
    psi : array_like
        Nonnegative unit-norm vector(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Simplex composition(s), same shape as input.

    Raises
    ------
    ValueError
        If any element of *psi* is negative.

    Warns
    -----
    UserWarning
        If *psi* is not unit-norm (tolerance 1e-6).
    """
    psi = np.asarray(psi, dtype=np.float64)
    if np.any(psi < 0):
        raise ValueError(
            "fisher_project requires nonnegative amplitudes (psi_i >= 0)."
        )
    norms = np.linalg.norm(psi, axis=-1)
    if np.any(np.abs(norms - 1.0) > 1e-6):
        import warnings

        warnings.warn(
            "fisher_project received non-unit-norm input "
            f"(||psi|| deviates from 1 by up to {np.max(np.abs(norms - 1.0)):.2e}). "
            "Results may not lie on the simplex.",
            stacklevel=2,
        )
    return psi**2


# ---------------------------------------------------------------------------
# 4.1.2 — Overlap families
# ---------------------------------------------------------------------------


def phi(s: ArrayLike) -> NDArray[np.floating]:
    """Phi overlap: Phi_N(s) = (N/(N-1)) * (1 - sum(s_i^2)).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    n = _n(s)
    hhi = np.sum(s**2, axis=-1)
    return (n / (n - 1)) * (1.0 - hhi)


def psi_overlap(s: ArrayLike) -> NDArray[np.floating]:
    """Psi overlap: Psi_N(s) = N^N * prod(s_i).

    Uses log-sum-exp for numerical stability:
    log(Psi) = N*log(N) + sum(log(s_i)), then exp.
    Returns 0.0 when any s_i = 0.

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    n = _n(s)

    # Detect zeros: product is 0 if any component is 0
    has_zero = np.any(s == 0, axis=-1)

    # Safe log: replace zeros with 1 to avoid -inf, mask later
    safe_s = np.where(s > 0, s, 1.0)
    log_psi = n * np.log(n) + np.sum(np.log(safe_s), axis=-1)
    result = np.exp(log_psi)
    result = np.where(has_zero, 0.0, result)
    return result


# Alias per spec
psi = psi_overlap


def overlap_divergence(s: ArrayLike) -> NDArray[np.floating]:
    """Overlap divergence: D(s) = Phi_N(s) - Psi_N(s).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    n = _n(s)
    # Inline phi
    hhi = np.sum(s**2, axis=-1)
    phi_val = (n / (n - 1)) * (1.0 - hhi)
    # Inline psi
    has_zero = np.any(s == 0, axis=-1)
    safe_s = np.where(s > 0, s, 1.0)
    log_psi = n * np.log(n) + np.sum(np.log(safe_s), axis=-1)
    psi_val = np.where(has_zero, 0.0, np.exp(log_psi))
    return phi_val - psi_val


# ---------------------------------------------------------------------------
# 4.1.3 — Forced-pair invariants
# ---------------------------------------------------------------------------


def q_delta(s: ArrayLike) -> NDArray[np.floating]:
    """Q_delta = sum(s_i^2) - 1/N.

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    n = _n(s)
    return np.sum(s**2, axis=-1) - 1.0 / n


def h3(s: ArrayLike) -> NDArray[np.floating]:
    """H_3 = sum(s_i^3) - (3/(N+2))*sum(s_i^2) + 2/(N*(N+2)).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    n = _n(s)
    s2 = np.sum(s**2, axis=-1)
    s3 = np.sum(s**3, axis=-1)
    return s3 - (3.0 / (n + 2)) * s2 + 2.0 / (n * (n + 2))


def forced_pair(s: ArrayLike) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return the forced-pair invariants (Q_delta, H_3).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    tuple of ndarray
        ``(q_delta(s), h3(s))``.
    """
    s = _validated(s)
    n = _n(s)
    s2 = np.sum(s**2, axis=-1)
    s3 = np.sum(s**3, axis=-1)
    q = s2 - 1.0 / n
    h = s3 - (3.0 / (n + 2)) * s2 + 2.0 / (n * (n + 2))
    return q, h


def forced_coordinates(s: ArrayLike) -> NDArray[np.floating]:
    """Column-stacked forced-pair coordinates [Q_delta, H_3].

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Shape ``(2,)`` for single input or ``(M, 2)`` for batch.
    """
    s = _validated(s)
    n = _n(s)
    s2 = np.sum(s**2, axis=-1)
    s3 = np.sum(s**3, axis=-1)
    q = s2 - 1.0 / n
    h = s3 - (3.0 / (n + 2)) * s2 + 2.0 / (n * (n + 2))
    return np.stack([q, h], axis=-1)


# ---------------------------------------------------------------------------
# 4.1.4 — Heuristic diagnostic
# ---------------------------------------------------------------------------


def qh_ratio(
    s: ArrayLike,
    *,
    eps: float = 1e-12,
    on_zero: str = "nan",
) -> NDArray[np.floating]:
    """Ratio Q_delta / H_3.

    .. warning::
        Empirical heuristic. Unstable near H_3 = 0.

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.
    eps : float, optional
        Clipping threshold for the denominator when ``on_zero="clip"``.
    on_zero : {"nan", "inf", "clip"}, optional
        Behaviour when H_3 = 0:

        - ``"nan"``: return NaN.
        - ``"inf"``: return +/-inf.
        - ``"clip"``: clip denominator to +/-eps.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    q = q_delta(s)
    h = h3(s)

    if on_zero == "clip":
        h = np.where(np.abs(h) < eps, np.sign(h) * eps, h)
        # Handle exact zero sign -> positive eps
        h = np.where(h == 0, eps, h)
        return q / h

    if on_zero == "inf":
        with np.errstate(divide="ignore", invalid="ignore"):
            result = q / h
        # Where h==0, set to inf with sign of q
        zero_mask = h == 0
        result = np.where(zero_mask, np.sign(q) * np.inf, result)
        return result

    if on_zero == "nan":
        with np.errstate(divide="ignore", invalid="ignore"):
            return q / h

    raise ValueError(f"Unknown on_zero mode: {on_zero!r}")


# ---------------------------------------------------------------------------
# 4.1.6 — Bridge statistics
# ---------------------------------------------------------------------------


def herfindahl(s: ArrayLike) -> NDArray[np.floating]:
    """Herfindahl index: sum(s_i^2).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    return np.sum(s**2, axis=-1)


def simpson_index(s: ArrayLike) -> NDArray[np.floating]:
    """Simpson (Gini-Simpson) diversity index: 1 - sum(s_i^2).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    return 1.0 - np.sum(s**2, axis=-1)


# Alias per spec
gini_simpson = simpson_index


def shannon_entropy(
    s: ArrayLike,
    *,
    base: float | None = None,
) -> NDArray[np.floating]:
    """Shannon entropy: -sum(s_i * log(s_i)).

    Uses the convention 0 * log(0) = 0.

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.
    base : float or None, optional
        Logarithm base. ``None`` (default) uses natural log.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    s = _validated(s)
    # Safe log: avoid log(0) by replacing 0 with 1 (since 0*log(1)=0)
    safe_s = np.where(s > 0, s, 1.0)
    h = -np.sum(s * np.log(safe_s), axis=-1)
    if base is not None:
        h = h / np.log(base)
    return h


def effective_number_herfindahl(s: ArrayLike) -> NDArray[np.floating]:
    """Effective number of species (Herfindahl): 1 / sum(s_i^2).

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    return 1.0 / herfindahl(s)


def effective_number_shannon(
    s: ArrayLike,
    *,
    base: float | None = None,
) -> NDArray[np.floating]:
    """Effective number of species (Shannon): exp(H) or base^H.

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.
    base : float or None, optional
        Logarithm base. ``None`` uses natural log (and exp).

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    h = shannon_entropy(s, base=base)
    if base is not None:
        return base**h
    return np.exp(h)


# ---------------------------------------------------------------------------
# 4.1.7 — Binary helpers
# ---------------------------------------------------------------------------


def binary_overlap(x: ArrayLike) -> NDArray[np.floating]:
    """Binary overlap: 4*x*(1-x) for x in [0, 1].

    Parameters
    ----------
    x : array_like
        Scalar or array of values in [0, 1].

    Returns
    -------
    ndarray
        Binary overlap values.
    """
    x = np.asarray(x, dtype=np.float64)
    return 4.0 * x * (1.0 - x)


def binary_fisher_angle(x: ArrayLike) -> NDArray[np.floating]:
    """Binary Fisher angle: 2*arccos(sqrt(x)).

    Parameters
    ----------
    x : array_like
        Scalar or array of values in [0, 1].

    Returns
    -------
    ndarray
        Fisher angle values.
    """
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * np.arccos(np.sqrt(x))


# ---------------------------------------------------------------------------
# Top-k to simplex
# ---------------------------------------------------------------------------


def topk_to_simplex(
    values: ArrayLike,
    *,
    mode: str = "single_remainder",
    tail_cardinality: int | None = None,
) -> NDArray[np.floating]:
    """Construct a simplex vector from top-k probabilities.

    Parameters
    ----------
    values : array_like
        Non-negative top-k probability values of shape ``(..., K)``.
    mode : {"single_remainder", "renormalize", "known_tail"}, optional
        How to handle the residual mass. Default ``"single_remainder"``.

        - ``"single_remainder"``: append one bin for residual ``1 - sum``.
          Output shape ``(..., K+1)``.
        - ``"renormalize"``: renormalize to sum to 1. Output shape ``(..., K)``.
        - ``"known_tail"``: distribute residual across *tail_cardinality* bins.
          Output shape ``(..., K + tail_cardinality)``.

    tail_cardinality : int or None, optional
        Number of tail bins for ``"known_tail"`` mode.

    Returns
    -------
    ndarray
        Simplex composition(s).

    Raises
    ------
    ValueError
        If values are negative, sum exceeds 1 (for remainder modes),
        or mode is unknown.
    """
    values = np.asarray(values, dtype=np.float64)

    if np.any(values < 0):
        raise ValueError("Values must be non-negative.")

    if mode == "renormalize":
        total = np.sum(values, axis=-1, keepdims=True)
        return values / total

    s = np.sum(values, axis=-1)
    if np.any(s > 1.0 + 1e-10):
        raise ValueError("Sum of values exceeds 1.")

    if mode == "single_remainder":
        remainder = np.maximum(1.0 - s, 0.0)
        return np.concatenate(
            [values, remainder[..., np.newaxis]], axis=-1
        )

    if mode == "known_tail":
        if tail_cardinality is None:
            raise ValueError("known_tail mode requires tail_cardinality.")
        remainder = np.maximum(1.0 - s, 0.0)
        per_bin = remainder / tail_cardinality
        tail_shape = values.shape[:-1] + (tail_cardinality,)
        tail = np.broadcast_to(per_bin[..., np.newaxis], tail_shape)
        return np.concatenate([values, tail], axis=-1)

    raise ValueError(f"Unknown mode: {mode!r}")
