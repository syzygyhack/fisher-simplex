"""Frontier module — experimental degree-8 enrichment tools.

Provides experimental tools for the first non-forced symmetric-even
enrichment at degree 8 in the Fisher-lifted sector. All public
functions in this module are **experimental**.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fisher_simplex.core import h3, q_delta
from fisher_simplex.utils import validate_simplex

# ---------------------------------------------------------------------------
# Internal helpers — Dirichlet(1) moments via set-partition expansion
# ---------------------------------------------------------------------------


def _set_partitions(n: int):
    """Yield all set partitions of {0, ..., n-1} as lists of blocks."""
    if n == 0:
        yield []
        return
    for partition in _set_partitions(n - 1):
        for i in range(len(partition)):
            new = partition[:i] + [partition[i] + [n - 1]] + partition[i + 1 :]
            yield new
        yield partition + [[n - 1]]


def _power_sum_product_moment(N: int, ks: tuple[int, ...]) -> float:
    """E[p_{k1} * ... * p_{km}] under Dirichlet(1,...,1) on Delta^{N-1}.

    Uses the set-partition expansion: each set partition of the m factors
    corresponds to a pattern of index coincidences in the sum over
    N simplex components.
    """
    if not ks:
        return 1.0

    m = len(ks)
    K = sum(ks)

    # Pochhammer (N)_K = N * (N+1) * ... * (N+K-1)
    poch = 1.0
    for i in range(K):
        poch *= N + i

    total = 0.0
    for partition in _set_partitions(m):
        r = len(partition)
        if r > N:
            continue
        # Falling factorial N * (N-1) * ... * (N-r+1)
        falling = 1.0
        for i in range(r):
            falling *= N - i
        # Product of merged-exponent factorials
        fact_prod = 1.0
        for block in partition:
            merged = sum(ks[j] for j in block)
            fact_prod *= factorial(merged)
        total += falling * fact_prod

    return total / poch


# ---------------------------------------------------------------------------
# Internal — orthogonalized E8 basis coefficients
# ---------------------------------------------------------------------------

# Basis B = {1, p_2, p_3, p_4, p_2^2} where p_k = sum(s_i^k).
# Map (i, j) -> power-sum indices for the product b_i * b_j.
_BASIS_PRODUCTS: dict[tuple[int, int], tuple[int, ...]] = {
    (0, 0): (),
    (0, 1): (2,),
    (0, 2): (3,),
    (0, 3): (4,),
    (0, 4): (2, 2),
    (1, 1): (2, 2),
    (1, 2): (2, 3),
    (1, 3): (2, 4),
    (1, 4): (2, 2, 2),
    (2, 2): (3, 3),
    (2, 3): (3, 4),
    (2, 4): (2, 2, 3),
    (3, 3): (4, 4),
    (3, 4): (2, 2, 4),
    (4, 4): (2, 2, 2, 2),
}


@lru_cache(maxsize=32)
def _e8_coefficients(N: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute E8_1 and E8_2 coefficients in basis {1, p2, p3, p4, p2^2}.

    Modified Gram-Schmidt under the Dirichlet(1) inner product produces
    five orthonormal directions. The first three span {1, p2, p3} (the
    forced block); the last two are the degree-8 enrichment directions.
    """
    # Build 5x5 Gram matrix: G[i,j] = E[b_i * b_j]
    G = np.zeros((5, 5))
    for (i, j), ks in _BASIS_PRODUCTS.items():
        val = _power_sum_product_moment(N, ks)
        G[i, j] = val
        G[j, i] = val

    # Modified Gram-Schmidt with inner product <c, d> = c^T G d
    ortho = np.zeros((5, 5))
    for i in range(5):
        v = np.zeros(5)
        v[i] = 1.0
        for j in range(i):
            gj = ortho[j]
            denom = gj @ G @ gj
            if denom < 1e-30:
                continue
            v -= ((v @ G @ gj) / denom) * gj
        norm_sq = v @ G @ v
        if norm_sq > 1e-30:
            v /= np.sqrt(norm_sq)
        ortho[i] = v

    return tuple(ortho[3].tolist()), tuple(ortho[4].tolist())


def _eval_basis(s: NDArray) -> NDArray:
    """Evaluate {1, p2, p3, p4, p2^2} at simplex composition(s)."""
    p2 = np.sum(s**2, axis=-1)
    p3 = np.sum(s**3, axis=-1)
    p4 = np.sum(s**4, axis=-1)
    p2_sq = p2**2
    ones = np.ones_like(p2)
    return np.stack([ones, p2, p3, p4, p2_sq], axis=-1)


# ---------------------------------------------------------------------------
# 4.5.1 — Degree-8 coordinates
# ---------------------------------------------------------------------------


def frontier8_coordinates(s: ArrayLike) -> NDArray[np.floating]:
    """Degree-8 enrichment coordinates [Q_delta, H_3, E8_1, E8_2].

    **Experimental.** Returns the forced-pair coordinates augmented with
    two degree-8 enrichment directions orthogonal to (Q_delta, H_3) under
    the Dirichlet(1) uniform measure on the simplex.

    E8_1 and E8_2 are constructed by Gram-Schmidt orthogonalization of
    the degree-4 (in s) symmetric polynomial candidates {p_4, p_2^2}
    against the forced block {1, p_2, p_3}.

    Parameters
    ----------
    s : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Shape ``(4,)`` for single input or ``(M, 4)`` for batch.
    """
    s = validate_simplex(s, renormalize="warn")
    N = s.shape[-1]
    c1, c2 = _e8_coefficients(N)
    c1_arr = np.array(c1)
    c2_arr = np.array(c2)

    basis = _eval_basis(s)
    qd = q_delta(s)
    h = h3(s)
    e8_1 = basis @ c1_arr
    e8_2 = basis @ c2_arr

    return np.stack([qd, h, e8_1, e8_2], axis=-1)


def frontier8_batch(X: ArrayLike) -> NDArray[np.floating]:
    """Batch degree-8 enrichment coordinates.

    **Experimental.** Convenience wrapper ensuring batch output
    with shape ``(M, 4)``.

    Parameters
    ----------
    X : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Shape ``(M, 4)`` — always 2-d.
    """
    result = frontier8_coordinates(X)
    if result.ndim == 1:
        result = result[np.newaxis, :]
    return result


# ---------------------------------------------------------------------------
# 4.5.2 — Residual diagnostics
# ---------------------------------------------------------------------------


def _r_squared(X: NDArray, y: NDArray) -> float:
    """R-squared from OLS regression of y on X (intercept included)."""
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-30:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def frontier8_residual(
    X: ArrayLike,
    target: ArrayLike,
    *,
    model: str = "linear",
) -> dict:
    """Residual diagnostics for degree-8 enrichment.

    **Experimental.** Quantifies whether a target observable requires
    degree-8 enrichment beyond the forced pair (Q_delta, H_3).

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    target : array_like
        Target values of shape ``(M,)``.
    model : {"linear", "quadratic"}, optional
        Whether the forced-block model uses linear or quadratic terms
        in (Q_delta, H_3). Default ``"linear"``.

    Returns
    -------
    dict
        Keys:

        - ``"r_squared_forced"`` -- R-squared of target against forced model.
        - ``"r_squared_frontier"`` -- R-squared against full frontier model.
        - ``"frontier_improvement"`` -- difference.
        - ``"needs_frontier"`` -- True if improvement exceeds 0.01.
    """
    X_arr = validate_simplex(X, renormalize="warn")
    target_arr = np.asarray(target, dtype=np.float64)

    coords = frontier8_coordinates(X_arr)
    qd = coords[:, 0]
    h = coords[:, 1]
    e8_1 = coords[:, 2]
    e8_2 = coords[:, 3]

    if model == "linear":
        X_forced = np.column_stack([qd, h])
    elif model == "quadratic":
        X_forced = np.column_stack([qd, h, qd**2, qd * h, h**2])
    else:
        raise ValueError(f"Unknown model: {model!r}")

    X_frontier = np.column_stack([X_forced, e8_1, e8_2])

    r2_forced = _r_squared(X_forced, target_arr)
    r2_frontier = _r_squared(X_frontier, target_arr)
    improvement = r2_frontier - r2_forced

    return {
        "r_squared_forced": r2_forced,
        "r_squared_frontier": r2_frontier,
        "frontier_improvement": improvement,
        "needs_frontier": improvement > 0.01,
    }
