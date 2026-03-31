"""Mechanistic interpretability helpers for attention analysis.

Provides chart discovery, tangent decomposition, shared mode extraction,
and geometric reconstruction tools for analyzing ensembles of simplex
compositions (e.g., transformer attention distributions).

Typical workflow::

    from fisher_simplex.interp import (
        mean_overlap_matrix, discover_charts,
        extract_shared_modes, project_to_modes,
    )

    # 1. Organize distributions as (n_entities, n_conditions, N)
    # 2. Compute overlap and discover charts
    overlap = mean_overlap_matrix(X_3d)
    charts = discover_charts(overlap, tau=0.7)

    # 3. Extract shared modes within a chart
    chart_data = X_3d[charts["charts"][0]]  # subset to chart members
    modes = extract_shared_modes(chart_data, n_modes=3)

    # 4. Project / reconstruct individual distributions
    reconstructed = project_to_modes(
        distribution,
        centroid=modes["centroid"],
        tangent_basis=modes["tangent_basis"],
        shared_modes=modes["shared_modes"],
        tangent_mean=modes["tangent_mean"],
    )

All functions use the Fisher-Rao geometry provided by the core library.
No PyTorch or other ML framework dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fisher_simplex.core import fisher_lift
from fisher_simplex.geometry import (
    fisher_expmap,
    fisher_logmap,
    fisher_mean,
)
from fisher_simplex.utils import validate_simplex

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validated(x: ArrayLike) -> NDArray[np.floating]:
    """Validate and return a simplex array."""
    return validate_simplex(x, renormalize="warn")


# ---------------------------------------------------------------------------
# Overlap matrices
# ---------------------------------------------------------------------------


def overlap_matrix(X: ArrayLike) -> NDArray[np.floating]:
    """Bhattacharyya coefficient (overlap) matrix.

    Computes the pairwise Bhattacharyya coefficient between all pairs
    of simplex compositions. The coefficient is the inner product of
    Fisher-lifted amplitude vectors: ``BC(a, b) = sum(sqrt(a_i * b_i))``.

    Parameters
    ----------
    X : array_like, shape (M, N)
        M simplex compositions of dimension N.

    Returns
    -------
    ndarray, shape (M, M)
        Symmetric matrix with values in [0, 1] and unit diagonal.

    Raises
    ------
    ValueError
        If *X* is not at least 2-D.
    """
    X = _validated(X)
    if X.ndim < 2:
        raise ValueError(
            f"overlap_matrix requires shape (M, N), got shape {X.shape}."
        )
    psi = fisher_lift(X)
    bc = psi @ psi.T
    bc = np.clip(bc, 0.0, 1.0)
    np.fill_diagonal(bc, 1.0)
    return (bc + bc.T) / 2.0


def mean_overlap_matrix(X_3d: ArrayLike) -> NDArray[np.floating]:
    """Mean Bhattacharyya overlap matrix averaged across conditions.

    For each condition (e.g., prompt), computes the pairwise overlap
    matrix between entities (e.g., attention heads), then averages
    across conditions.

    Parameters
    ----------
    X_3d : array_like, shape (n_entities, n_conditions, N)
        Distributions organized as (entity, condition, simplex_dim).

    Returns
    -------
    ndarray, shape (n_entities, n_entities)
        Mean overlap matrix averaged across conditions.

    Raises
    ------
    ValueError
        If *X_3d* is not 3-D.
    """
    X = _validated(X_3d)
    if X.ndim != 3:
        raise ValueError(
            f"mean_overlap_matrix requires shape "
            f"(n_entities, n_conditions, N), got ndim={X.ndim}."
        )
    n_entities, n_conditions, _N = X.shape
    result = np.zeros((n_entities, n_entities))
    for c in range(n_conditions):
        result += overlap_matrix(X[:, c, :])
    result /= n_conditions
    return result


# ---------------------------------------------------------------------------
# Chart discovery
# ---------------------------------------------------------------------------


def discover_charts(
    overlap: ArrayLike,
    *,
    tau: float = 0.7,
) -> dict:
    """Discover geometric charts via thresholded overlap clustering.

    Finds connected components in the graph where entities are connected
    if their Bhattacharyya overlap exceeds the threshold *tau*.

    Parameters
    ----------
    overlap : array_like, shape (M, M)
        Pairwise overlap (Bhattacharyya coefficient) matrix.
    tau : float, optional
        Overlap threshold. Pairs with ``overlap > tau`` are connected.
        Default ``0.7``.

    Returns
    -------
    dict
        Keys:

        - ``"n_charts"`` (int) — number of charts found.
        - ``"labels"`` (ndarray shape (M,)) — chart label per entity.
        - ``"charts"`` (dict) — mapping ``chart_id -> list of entity indices``.
        - ``"sizes"`` (dict) — mapping ``chart_id -> chart size``.
    """
    from scipy.sparse.csgraph import connected_components

    overlap = np.asarray(overlap, dtype=np.float64)
    adj = overlap > tau
    n_charts, labels = connected_components(adj, directed=False)

    charts = {}
    sizes = {}
    for c in range(n_charts):
        members = np.where(labels == c)[0].tolist()
        charts[c] = members
        sizes[c] = len(members)

    return {
        "n_charts": n_charts,
        "labels": labels,
        "charts": charts,
        "sizes": sizes,
    }


def chart_stability(
    X_3d: ArrayLike,
    *,
    tau: float = 0.7,
    min_fraction: float = 0.7,
) -> dict:
    """Assess chart stability across conditions.

    For each pair of entities, computes the fraction of conditions
    (e.g., prompts) where both are assigned to the same chart. Stable
    charts are those whose membership is consistent across conditions.

    Parameters
    ----------
    X_3d : array_like, shape (n_entities, n_conditions, N)
        Distributions organized as (entity, condition, simplex_dim).
    tau : float, optional
        Overlap threshold for per-condition chart discovery. Default ``0.7``.
    min_fraction : float, optional
        Minimum co-occurrence fraction for stable chart membership.
        Default ``0.7``.

    Returns
    -------
    dict
        Keys:

        - ``"co_occurrence"`` (ndarray shape (M, M)) — fraction of
          conditions where each pair co-clusters.
        - ``"stable_labels"`` (ndarray shape (M,)) — chart labels from
          thresholding the co-occurrence matrix.
        - ``"n_stable_charts"`` (int)
        - ``"per_condition_labels"`` (ndarray shape (n_conditions, M))
    """
    from scipy.sparse.csgraph import connected_components

    X = _validated(X_3d)
    if X.ndim != 3:
        raise ValueError(
            f"chart_stability requires shape "
            f"(n_entities, n_conditions, N), got ndim={X.ndim}."
        )
    n_entities, n_conditions, _N = X.shape

    co_occurrence = np.zeros((n_entities, n_entities))
    per_condition_labels = np.zeros(
        (n_conditions, n_entities), dtype=np.int32,
    )

    for c in range(n_conditions):
        ov = overlap_matrix(X[:, c, :])
        charts = discover_charts(ov, tau=tau)
        labels = charts["labels"]
        per_condition_labels[c] = labels
        # Vectorized co-occurrence update
        co_occurrence += (
            labels[:, np.newaxis] == labels[np.newaxis, :]
        ).astype(np.float64)

    co_occurrence /= n_conditions

    # Stable charts from thresholded co-occurrence
    stable_adj = co_occurrence >= min_fraction
    n_stable, stable_labels = connected_components(
        stable_adj, directed=False,
    )

    return {
        "co_occurrence": co_occurrence,
        "stable_labels": stable_labels,
        "n_stable_charts": n_stable,
        "per_condition_labels": per_condition_labels,
    }


# ---------------------------------------------------------------------------
# Shared mode extraction
# ---------------------------------------------------------------------------


def extract_shared_modes(
    X_3d: ArrayLike,
    *,
    centroid: ArrayLike | None = None,
    n_modes: int = 3,
    tangent_dim: int | None = None,
) -> dict:
    """Extract shared condition-variation modes across entities.

    Computes a tangent-space decomposition of input distributions,
    removes per-entity effects (e.g., per-head offsets), and extracts
    shared modes (common directions of variation across conditions)
    via SVD.

    The algorithm:

    1. Flatten to ``(M, N)`` and compute Fisher mean as base point.
    2. Map all distributions to tangent space via :func:`fisher_logmap`.
    3. PCA of tangent vectors for dimensionality reduction.
    4. Reshape to ``(n_entities, n_conditions, K)`` in tangent subspace.
    5. Remove per-entity mean (head effect).
    6. SVD of all centered trajectories to find shared modes.
    7. Compute per-entity R-squared for the shared-mode fit.

    Parameters
    ----------
    X_3d : array_like, shape (n_entities, n_conditions, N)
        Distributions organized as (entity, condition, simplex_dim).
        For attention analysis: entities = heads, conditions = prompts.
    centroid : array_like shape (N,), optional
        Base point for tangent-space projection. If ``None``, uses
        the Fisher mean of all distributions.
    n_modes : int, optional
        Number of shared modes to extract. Default ``3``.
    tangent_dim : int, optional
        Tangent subspace dimension for intermediate PCA projection.
        If ``None``, uses ``min(7, N - 1, M - 1)``.

    Returns
    -------
    dict
        Keys:

        - ``"centroid"`` (ndarray (N,)) — base point on simplex.
        - ``"tangent_mean"`` (ndarray (N,)) — mean tangent vector
          (needed for centering in :func:`project_to_modes`).
        - ``"tangent_basis"`` (ndarray (K, N)) — top-K tangent PCA
          components.
        - ``"shared_modes"`` (ndarray (n_modes, K)) — shared modes
          in the tangent subspace.
        - ``"shared_singular_values"`` (ndarray) — singular values
          of the shared mode decomposition.
        - ``"head_effects"`` (ndarray (n_entities, K)) — per-entity
          mean tangent offset.
        - ``"per_entity_r2"`` (ndarray (n_entities,)) — R-squared
          of shared-mode fit per entity.
        - ``"mean_r2"`` (float) — mean R-squared across entities.
        - ``"dim_90"`` (int) — tangent dimensions for 90% variance.
        - ``"dim_95"`` (int) — tangent dimensions for 95% variance.
        - ``"tangent_singular_values"`` (ndarray) — singular values
          of the full tangent decomposition.

    Raises
    ------
    ValueError
        If *X_3d* is not 3-D.
    """
    X = _validated(X_3d)
    if X.ndim != 3:
        raise ValueError(
            f"extract_shared_modes requires shape "
            f"(n_entities, n_conditions, N), got ndim={X.ndim}."
        )
    n_entities, n_conditions, N = X.shape
    M = n_entities * n_conditions

    # Flatten for batch operations
    X_flat = X.reshape(M, N)

    # Base point
    if centroid is None:
        centroid_arr = fisher_mean(X_flat)
    else:
        centroid_arr = _validated(centroid)

    # Tangent vectors via Riemannian log map
    V = fisher_logmap(X_flat, base=centroid_arr)  # (M, N)

    # Center tangent vectors
    tangent_mean = V.mean(axis=0)  # (N,)
    V_centered = V - tangent_mean

    # SVD for tangent basis (PCA)
    U_t, S_t, Vt_t = np.linalg.svd(V_centered, full_matrices=False)

    # Tangent subspace dimension
    if tangent_dim is None:
        K = min(7, N - 1, M - 1)
    else:
        K = tangent_dim
    K = min(K, len(S_t))
    K = max(K, 1)

    # Variance explained
    var_total = np.sum(S_t**2)
    if var_total > 0:
        var_cumulative = np.cumsum(S_t**2) / var_total
        dim_90 = int(np.searchsorted(var_cumulative, 0.90) + 1)
        dim_95 = int(np.searchsorted(var_cumulative, 0.95) + 1)
    else:
        dim_90 = 1
        dim_95 = 1

    tangent_basis = Vt_t[:K]  # (K, N)

    # Project to K-dim subspace
    Z = V_centered @ tangent_basis.T  # (M, K)

    # Reshape to (n_entities, n_conditions, K)
    Z_3d = Z.reshape(n_entities, n_conditions, K)

    # Per-entity mean (head effects)
    head_effects = Z_3d.mean(axis=1)  # (n_entities, K)

    # Center out entity effects
    Z_centered = Z_3d - head_effects[:, np.newaxis, :]

    # SVD of all centered trajectories to find shared modes
    Z_concat = Z_centered.reshape(-1, K)  # (n_entities * n_conditions, K)
    _U_s, S_s, Vt_s = np.linalg.svd(Z_concat, full_matrices=False)

    n_modes_actual = min(n_modes, K, len(S_s))
    shared_modes = Vt_s[:n_modes_actual]  # (n_modes, K)

    # Per-entity R-squared
    per_entity_r2 = np.zeros(n_entities)
    for h in range(n_entities):
        traj = Z_centered[h]  # (n_conditions, K)
        proj = traj @ shared_modes.T @ shared_modes
        ss_res = np.sum((traj - proj) ** 2)
        ss_tot = np.sum(traj**2)
        per_entity_r2[h] = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    return {
        "centroid": centroid_arr,
        "tangent_mean": tangent_mean,
        "tangent_basis": tangent_basis,
        "shared_modes": shared_modes,
        "shared_singular_values": S_s[:n_modes_actual],
        "head_effects": head_effects,
        "per_entity_r2": per_entity_r2,
        "mean_r2": float(np.mean(per_entity_r2)),
        "dim_90": dim_90,
        "dim_95": dim_95,
        "tangent_singular_values": S_t,
    }


# ---------------------------------------------------------------------------
# Mode projection and reconstruction
# ---------------------------------------------------------------------------


def project_to_modes(
    s: ArrayLike,
    *,
    centroid: ArrayLike,
    tangent_basis: ArrayLike,
    shared_modes: ArrayLike,
    tangent_mean: ArrayLike | None = None,
    n_modes: int | None = None,
    head_effect: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """Project a distribution onto shared modes and reconstruct.

    Maps the input through the tangent-space decomposition, projects
    onto the shared modes, and reconstructs back to the simplex.

    The pipeline:

    1. Map to tangent space via :func:`fisher_logmap`.
    2. Center (subtract *tangent_mean*).
    3. Project to tangent subspace via *tangent_basis*.
    4. Remove *head_effect* (if provided).
    5. Project onto *shared_modes*.
    6. Add back *head_effect*.
    7. Reconstruct to ambient tangent space.
    8. Uncenter (add *tangent_mean*).
    9. Map back to simplex via :func:`fisher_expmap`.

    Parameters
    ----------
    s : array_like, shape (N,) or (M, N)
        Simplex composition(s) to project.
    centroid : array_like, shape (N,)
        Base point for tangent-space projection (from
        :func:`extract_shared_modes`).
    tangent_basis : array_like, shape (K, N)
        Tangent subspace basis (from :func:`extract_shared_modes`).
    shared_modes : array_like, shape (n_modes_max, K)
        Shared modes (from :func:`extract_shared_modes`).
    tangent_mean : array_like, shape (N,), optional
        Mean tangent vector for centering (from
        :func:`extract_shared_modes`). If ``None``, no centering.
    n_modes : int, optional
        Number of shared modes to use. If ``None``, uses all
        available modes.
    head_effect : array_like, shape (K,), optional
        Per-entity offset (from ``modes["head_effects"][i]``).
        If provided, subtracted before mode projection and added
        back after.

    Returns
    -------
    ndarray, shape (N,) or (M, N)
        Reconstructed simplex composition(s).
    """
    s = _validated(s)
    centroid_arr = _validated(centroid)
    tangent_basis_arr = np.asarray(tangent_basis, dtype=np.float64)
    shared_modes_arr = np.asarray(shared_modes, dtype=np.float64)

    single = s.ndim == 1
    if single:
        s = s[np.newaxis, :]

    if n_modes is not None:
        shared_modes_arr = shared_modes_arr[:n_modes]

    # Map to tangent space
    V = fisher_logmap(s, base=centroid_arr)  # (M, N)

    # Center
    if tangent_mean is not None:
        V = V - np.asarray(tangent_mean, dtype=np.float64)

    # Project to tangent subspace
    Z = V @ tangent_basis_arr.T  # (M, K)

    # Remove head effect
    if head_effect is not None:
        Z = Z - np.asarray(head_effect, dtype=np.float64)

    # Project onto shared modes
    Z_proj = Z @ shared_modes_arr.T @ shared_modes_arr  # (M, K)

    # Add back head effect
    if head_effect is not None:
        Z_proj = Z_proj + np.asarray(head_effect, dtype=np.float64)

    # Back to ambient tangent space
    V_recon = Z_proj @ tangent_basis_arr  # (M, N)

    # Uncenter
    if tangent_mean is not None:
        V_recon = V_recon + np.asarray(tangent_mean, dtype=np.float64)

    # Back to simplex
    result = fisher_expmap(V_recon, base=centroid_arr)  # (M, N)

    # Safety: clip and normalize
    result = np.maximum(result, 0.0)
    sums = result.sum(axis=-1, keepdims=True)
    safe_sums = np.where(sums > 1e-30, sums, 1.0)
    result = np.where(sums > 1e-30, result / safe_sums, centroid_arr)

    if single:
        return result[0]
    return result
