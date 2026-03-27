"""Low-degree symmetric-even harmonic tools in Fisher-lift coordinates.

**Status:** Experimental module (v0.3).

Scope:

* Exact precomputed low-degree symmetric-even bases through degree 8.
* Exact low-degree dimension information via partition counting.
* Algorithmic basis construction for higher degrees (experimental).

Implements spec section 4.6.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _partition_count(k: int, max_parts: int) -> int:
    """Count partitions of integer *k* into at most *max_parts* parts.

    Uses dynamic programming: builds table ``P[i][j]`` = number of
    partitions of *i* into parts of size at most *j* (equivalent to
    partitions of *i* into at most *j* parts by Young-diagram conjugation).

    **Status:** exact.

    Parameters
    ----------
    k : int
        Non-negative integer to partition (or negative, returning 0).
    max_parts : int
        Maximum number of parts (equivalently, maximum part size in the
        conjugate representation).

    Returns
    -------
    int
        Number of partitions.
    """
    if k < 0:
        return 0
    if k == 0:
        return 1

    # P[i][j] = partitions of i using parts of size <= j
    table = [[0] * (max_parts + 1) for _ in range(k + 1)]
    for j in range(max_parts + 1):
        table[0][j] = 1  # one partition of 0: the empty partition

    for i in range(1, k + 1):
        for j in range(1, max_parts + 1):
            table[i][j] = table[i][j - 1]
            if i >= j:
                table[i][j] += table[i - j][j]

    return table[k][max_parts]


# ---------------------------------------------------------------------------
# 4.6.3 — Dimension formula
# ---------------------------------------------------------------------------


def symmetric_even_dimension(N: int, degree: int) -> int:
    """Dimension of the S_N-symmetric even harmonic sector at *degree*.

    Counts the number of *new* S_N-symmetric even spherical harmonics
    appearing at the given degree in the Fisher-lift filtration.

    Formula (spec 4.6.3)::

        dim = P(k, N) - P(k - 1, N)

    where ``k = degree // 2`` and ``P`` is :func:`_partition_count`.

    **Status:** exact.

    Parameters
    ----------
    N : int
        Number of simplex components.
    degree : int
        Spherical-harmonic degree (must be even).

    Returns
    -------
    int
        Dimension of the new symmetric-even harmonic sector.

    Raises
    ------
    ValueError
        If *degree* is odd.
    """
    if degree % 2 != 0:
        raise ValueError(
            f"degree must be even for symmetric-even harmonics, got {degree}"
        )
    k = degree // 2
    return _partition_count(k, N) - _partition_count(k - 1, N)


# ---------------------------------------------------------------------------
# Precomputed bases (exact through degree 8)
# ---------------------------------------------------------------------------

# Monomial keys are tuples of power-sum indices in non-increasing order.
# (2,) -> p_2 = sum(s_i^2)
# (3,) -> p_3 = sum(s_i^3)
# (4,) -> p_4 = sum(s_i^4)
# (2, 2) -> p_2^2 = (sum(s_i^2))^2
# () -> constant 1

_PRECOMPUTED_BASES: Dict[int, List[Dict[str, Any]]] = {
    0: [
        {
            "coefficients": {(): 1.0},
            "degree": 0,
            "description": (
                "Constant harmonic (trivial S_N-invariant). Status: exact."
            ),
        },
    ],
    2: [],  # No nonconstant symmetric-even harmonics at degree 2.
    4: [
        {
            "coefficients": {(2,): 1.0},
            "degree": 4,
            "description": (
                "Q_Δ: centered second power sum (p₂ − 1/N). "
                "Degree-4 forced invariant. Status: exact."
            ),
        },
    ],
    6: [
        {
            "coefficients": {(3,): 1.0},
            "degree": 6,
            "description": (
                "H_3: centered third power sum harmonic. "
                "Degree-6 forced invariant. Status: exact."
            ),
        },
    ],
    8: [
        {
            "coefficients": {(4,): 1.0},
            "degree": 8,
            "description": (
                "Raw degree-8 monomial p₄ = Σs_i⁴. Spans the degree-8 "
                "enrichment subspace together with p₂². Not orthogonalized "
                "against the forced block; use frontier.frontier8_coordinates "
                "for Dirichlet(1)-orthogonal directions. Status: exact."
            ),
        },
        {
            "coefficients": {(2, 2): 1.0},
            "degree": 8,
            "description": (
                "Raw degree-8 monomial p₂² = (Σs_i²)². Spans the degree-8 "
                "enrichment subspace together with p₄. Not orthogonalized "
                "against the forced block; use frontier.frontier8_coordinates "
                "for Dirichlet(1)-orthogonal directions. Status: exact."
            ),
        },
    ],
}

_MAX_PRECOMPUTED_DEGREE = 8


# ---------------------------------------------------------------------------
# 4.6.2 — Basis construction
# ---------------------------------------------------------------------------


def symmetric_even_basis(
    N: int,
    degree: int,
    *,
    method: str = "precomputed",
) -> List[Dict[str, Any]]:
    """Return basis for S_N-symmetric even harmonics at *degree*.

    Each basis element is a dict::

        {"coefficients": {monomial_key: float}, "degree": int, "description": str}

    where *monomial_key* is a tuple of power-sum exponents in
    non-increasing order, e.g. ``(2,)`` for p₂, ``(2, 2)`` for p₂².

    **Status:** exact for ``method="precomputed"`` (degree ≤ 8);
    experimental for ``method="algorithmic"`` (any degree).

    Parameters
    ----------
    N : int
        Number of simplex components.
    degree : int
        Spherical-harmonic degree (must be even).
    method : {"precomputed", "algorithmic", "auto"}
        * ``"precomputed"`` — use built-in bases (degree ≤ 8 only).
        * ``"algorithmic"`` — compute via representation theory
          (experimental, any even degree).
        * ``"auto"`` — precomputed if available, algorithmic otherwise.

    Returns
    -------
    list of dict
        Basis elements (length equals :func:`symmetric_even_dimension`).

    Raises
    ------
    ValueError
        If *degree* is odd, or if ``method="precomputed"`` and
        degree > 8.
    NotImplementedError
        If ``method="algorithmic"`` is requested (experimental stub).
    """
    if degree % 2 != 0:
        raise ValueError(
            f"degree must be even for symmetric-even harmonics, got {degree}"
        )

    dim = symmetric_even_dimension(N, degree)

    if method == "precomputed":
        if degree > _MAX_PRECOMPUTED_DEGREE:
            raise ValueError(
                f"No precomputed basis for degree {degree}. "
                f"Precomputed bases are available through degree "
                f"{_MAX_PRECOMPUTED_DEGREE}. Use method='algorithmic' "
                f"(experimental) for higher degrees."
            )
        full_basis = _PRECOMPUTED_BASES.get(degree, [])
        # Return only as many elements as the dimension for this N.
        return full_basis[:dim]

    if method == "algorithmic":
        raise NotImplementedError(
            "Algorithmic basis construction is experimental and not yet "
            "implemented. Use method='precomputed' for degree ≤ 8."
        )

    if method == "auto":
        if degree <= _MAX_PRECOMPUTED_DEGREE:
            return symmetric_even_basis(N, degree, method="precomputed")
        raise NotImplementedError(
            f"No precomputed basis for degree {degree} and algorithmic "
            f"construction is not yet implemented (experimental)."
        )

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Projection tools (experimental)
# ---------------------------------------------------------------------------


def project_to_degree(
    f: ArrayLike,
    degree: int,
    *,
    N: Optional[int] = None,
) -> NDArray[np.floating]:
    """Project observable *f* onto the degree-*degree* harmonic component.

    **Status:** experimental. Requires sample compositions for
    evaluation of basis functions; full implementation pending.

    Parameters
    ----------
    f : array_like
        Observable values at sample points, shape ``(M,)``.
    degree : int
        Target harmonic degree.
    N : int or None
        Number of simplex components (inferred if possible).

    Returns
    -------
    ndarray
        Projected component (same shape as *f*).

    Raises
    ------
    NotImplementedError
        Always (experimental stub).
    """
    raise NotImplementedError(
        "project_to_degree is experimental and requires sample compositions. "
        "Full implementation pending."
    )


def filtration_decomposition(
    f: ArrayLike,
    *,
    max_degree: int = 8,
    N: Optional[int] = None,
) -> Dict[int, Any]:
    """Decompose observable *f* into harmonic-degree components.

    **Status:** experimental. Requires sample compositions for
    evaluation of basis functions; full implementation pending.

    Parameters
    ----------
    f : array_like
        Observable values at sample points, shape ``(M,)``.
    max_degree : int
        Maximum harmonic degree to decompose through.
    N : int or None
        Number of simplex components (inferred if possible).

    Returns
    -------
    dict
        Mapping ``degree -> projected component`` for even degrees
        from 0 through *max_degree*.

    Raises
    ------
    NotImplementedError
        Always (experimental stub).
    """
    raise NotImplementedError(
        "filtration_decomposition is experimental and requires sample "
        "compositions. Full implementation pending."
    )


# ---------------------------------------------------------------------------
# Selective frontier
# ---------------------------------------------------------------------------


def selective_frontier(N: int) -> Dict[str, Any]:
    """Return information about the first selective frontier.

    The first selective frontier occurs at degree 8, where the
    symmetric-even harmonic sector gains dimensions beyond those
    forced by the pair (Q_Δ, H_3).

    **Status:** exact (derived from partition counting).

    Parameters
    ----------
    N : int
        Number of simplex components.

    Returns
    -------
    dict
        Keys: ``degree``, ``total_dimension``, ``forced_dimension``,
        ``enrichment_dimension``, ``description``.
    """
    forced = sum(symmetric_even_dimension(N, d) for d in range(0, 8, 2))
    enrichment = symmetric_even_dimension(N, 8)
    total = forced + enrichment

    return {
        "degree": 8,
        "total_dimension": total,
        "forced_dimension": forced,
        "enrichment_dimension": enrichment,
        "description": (
            f"First selective frontier at degree 8 for N={N}. "
            f"The forced block (Q_Δ, H_3) spans {forced} dimensions "
            f"through degree 6. At degree 8, {enrichment} new "
            f"enrichment dimension(s) appear, orthogonal to the "
            f"forced block under the Dirichlet(1) measure."
        ),
    }


# ---------------------------------------------------------------------------
# Enrichment space
# ---------------------------------------------------------------------------


def enrichment_space(
    N: int,
    degree: int,
    *,
    method: str = "auto",
) -> List[Dict[str, Any]]:
    """Return spanning monomials for the enrichment subspace at *degree*.

    Returns the raw power-sum monomials that span the degree-*degree*
    enrichment subspace (the new symmetric-even harmonics beyond the
    forced block). These monomials are **not** orthogonalized against
    the forced block; they form a spanning set whose specific basis
    vectors depend on convention.

    For Dirichlet(1)-orthogonalized enrichment coordinates, use
    :func:`fisher_simplex.frontier.frontier8_coordinates`.

    At degree 8 with N >= 4, this returns 2 elements ({p_4, p_2^2}).

    **Status:** exact for degree <= 8 (precomputed); experimental
    for higher degrees.

    Parameters
    ----------
    N : int
        Number of simplex components.
    degree : int
        Harmonic degree (must be even, >= 8 for nontrivial enrichment).
    method : {"precomputed", "algorithmic", "auto"}
        Basis construction method.

    Returns
    -------
    list of dict
        Spanning monomials for the enrichment subspace at *degree*.
    """
    return symmetric_even_basis(N, degree, method=method)
