"""Geometry of the simplex via the Fisher amplitude lift.

Distances, kernels, means, geodesics, tangent-space tools, and PCA
as specified in spec section 4.2.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fisher_simplex.core import fisher_lift, fisher_project, forced_pair
from fisher_simplex.utils import (  # noqa: F401
    closure,
    project_to_simplex,
    validate_simplex,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validated(x: ArrayLike) -> NDArray[np.floating]:
    """Validate and return a simplex array."""
    return validate_simplex(x, renormalize="warn")


def _require_batch(X: NDArray, name: str) -> NDArray:
    """Ensure *X* is 2-D; raise ValueError on 1-D input."""
    if X.ndim < 2:
        raise ValueError(
            f"{name} requires a batch of compositions with shape (M, N), "
            f"got shape {X.shape}. Pass a single composition as X[np.newaxis, :]."
        )
    return X


# ---------------------------------------------------------------------------
# 4.2.1 — Distances and coefficients
# ---------------------------------------------------------------------------


def bhattacharyya_coefficient(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """Bhattacharyya coefficient: BC(a, b) = sum(sqrt(a_i) * sqrt(b_i)).

    Computed as the inner product of Fisher-lifted amplitude vectors,
    which gives better floating-point behaviour (exact BC=1 for self-pairs).

    Parameters
    ----------
    a, b : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    a = _validated(a)
    b = _validated(b)
    psi_a = np.sqrt(a)
    psi_b = np.sqrt(b)
    return np.sum(psi_a * psi_b, axis=-1)


def hellinger_distance(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """Hellinger distance: d_H = (1/sqrt(2)) * ||sqrt(a) - sqrt(b)||_2.

    Uses the normalized convention where ``d_H in [0, 1]`` and
    ``d_H**2 = 1 - B(a, b)``.  Related to Fisher distance by
    ``d_F = 2 * arccos(1 - d_H**2)``.

    Parameters
    ----------
    a, b : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    a = _validated(a)
    b = _validated(b)
    diff = np.sqrt(a) - np.sqrt(b)
    return np.sqrt(np.sum(diff**2, axis=-1)) / np.sqrt(2.0)


def fisher_distance(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """Fisher (geodesic) distance: d_F = 2 * arccos(clip(BC(a, b), -1, 1)).

    Clips the arccos argument to ``[-1, 1]`` for floating-point safety.

    Parameters
    ----------
    a, b : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    bc = bhattacharyya_coefficient(a, b)
    return 2.0 * np.arccos(np.clip(bc, -1.0, 1.0))


def pairwise_fisher_distances(X: ArrayLike) -> NDArray[np.floating]:
    """Pairwise Fisher distance matrix.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.

    Returns
    -------
    ndarray
        Shape ``(M, M)`` symmetric matrix with zero diagonal.
    """
    X = _validated(X)
    X = _require_batch(X, "pairwise_fisher_distances")
    psi = fisher_lift(X)
    # BC matrix via matrix multiply
    bc = psi @ psi.T
    bc = np.clip(bc, -1.0, 1.0)
    D = 2.0 * np.arccos(bc)
    # Enforce exact zero diagonal and symmetry
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D


def pairwise_hellinger_distances(X: ArrayLike) -> NDArray[np.floating]:
    """Pairwise Hellinger distance matrix.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.

    Returns
    -------
    ndarray
        Shape ``(M, M)`` symmetric matrix with zero diagonal.
    """
    X = _validated(X)
    X = _require_batch(X, "pairwise_hellinger_distances")
    sqrt_X = np.sqrt(X)
    m = X.shape[0]
    D = np.zeros((m, m))
    for i in range(m):
        diff = sqrt_X[i] - sqrt_X[i + 1 :]  # (m-i-1, N)
        D[i, i + 1 :] = np.sqrt(np.sum(diff**2, axis=-1)) / np.sqrt(2.0)
    D = D + D.T
    return D


# ---------------------------------------------------------------------------
# 4.2.2 — Kernels
# ---------------------------------------------------------------------------


def fisher_kernel(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.floating]:
    """Fisher kernel: same as :func:`bhattacharyya_coefficient`.

    Parameters
    ----------
    a, b : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    return bhattacharyya_coefficient(a, b)


# Alias per spec
fisher_cosine = fisher_kernel


def polynomial_fisher_kernel(
    a: ArrayLike,
    b: ArrayLike,
    d: int,
) -> NDArray[np.floating]:
    """Polynomial Fisher kernel: K_d(a, b) = B(a, b)^d.

    Raises the Bhattacharyya coefficient to integer power *d*, giving a
    positive-definite kernel for every ``d >= 1``.

    Parameters
    ----------
    a, b : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.
    d : int
        Polynomial degree (must be a positive integer).

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError(f"d must be a positive integer, got {d!r}.")
    bc = bhattacharyya_coefficient(a, b)
    return bc**d


def fisher_rbf_kernel(
    a: ArrayLike,
    b: ArrayLike,
    sigma: float,
) -> NDArray[np.floating]:
    """Radial basis function kernel in Fisher distance.

    Computes ``exp(-d_F(a, b)^2 / (2 * sigma^2))`` where *d_F* is the
    Fisher geodesic distance.

    Parameters
    ----------
    a, b : array_like
        Simplex composition(s) of shape ``(N,)`` or ``(M, N)``.
    sigma : float
        Bandwidth (must be positive).

    Returns
    -------
    ndarray
        Scalar or array of shape ``(M,)``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    d = fisher_distance(a, b)
    return np.exp(-(d**2) / (2.0 * sigma**2))


def kernel_matrix(
    X: ArrayLike,
    *,
    kind: str = "fisher",
    **kwargs: float,
) -> NDArray[np.floating]:
    """Kernel matrix.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    kind : {"fisher", "polynomial_fisher", "fisher_rbf", "hellinger_rbf"}, optional
        Kernel type. Default ``"fisher"``.

        - ``"fisher"`` — linear Fisher kernel ``B(p, q)`` (degree 1).
        - ``"polynomial_fisher"`` — ``B(p, q)^d``. Requires ``d``.
        - ``"fisher_rbf"`` — ``exp(-d_F^2 / (2*sigma^2))``. Requires ``sigma``.
        - ``"hellinger_rbf"`` — ``exp(-d_H^2 / sigma^2)``. Requires ``sigma``.

    **kwargs
        Additional parameters (``d``, ``sigma``) depending on ``kind``.

    Returns
    -------
    ndarray
        Shape ``(M, M)`` symmetric positive semidefinite matrix.
    """
    X = _validated(X)
    X = _require_batch(X, "kernel_matrix")

    if kind == "fisher":
        psi = fisher_lift(X)
        K = psi @ psi.T
        return K

    if kind == "polynomial_fisher":
        d = kwargs.get("d")
        if d is None:
            raise ValueError("kind='polynomial_fisher' requires d parameter.")
        if not isinstance(d, (int, np.integer)) or d < 1:
            raise ValueError(f"d must be a positive integer, got {d!r}.")
        psi = fisher_lift(X)
        bc = psi @ psi.T
        return bc**d

    if kind == "fisher_rbf":
        sigma = kwargs.get("sigma")
        if sigma is None:
            raise ValueError("kind='fisher_rbf' requires sigma parameter.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        D = pairwise_fisher_distances(X)
        return np.exp(-(D**2) / (2.0 * sigma**2))

    if kind == "hellinger_rbf":
        sigma = kwargs.get("sigma")
        if sigma is None:
            raise ValueError("kind='hellinger_rbf' requires sigma parameter.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        D = pairwise_hellinger_distances(X)
        return np.exp(-(D**2) / sigma**2)

    raise ValueError(f"Unknown kernel kind: {kind!r}")


# ---------------------------------------------------------------------------
# 4.2.3 — Means and barycenters
# ---------------------------------------------------------------------------


def fisher_mean(
    X: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """Extrinsic (projected arithmetic) mean in Fisher-lift amplitude coordinates.

    This is the extrinsic (projected arithmetic) mean in Fisher-lift amplitude
    coordinates. It approximates but is not identical to the intrinsic Fréchet
    mean under the Fisher geodesic distance. For data concentrated in a small
    region of the simplex, the two coincide to first order. An iterative
    intrinsic Fréchet mean solver may be added in a future version.

    Status: exact-derived utility.

    Algorithm: lift to amplitude space, compute weighted average, normalize
    to the unit sphere, project back to the simplex.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    weights : array_like or None, optional
        Nonnegative weights of shape ``(M,)``. If ``None``, uniform weights.

    Returns
    -------
    ndarray
        Shape ``(N,)`` simplex composition.
    """
    X = _validated(X)
    X = _require_batch(X, "fisher_mean")
    psi = fisher_lift(X)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        m = X.shape[0]
        if w.shape != (m,):
            raise ValueError(
                f"Weights must have shape ({m},), got {w.shape}."
            )
        if np.any(w < 0):
            raise ValueError("Weights must be nonnegative.")
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("Weights must sum to a positive value.")
        w = w / w_sum
        avg = np.einsum("i,ij->j", w, psi)
    else:
        avg = psi.mean(axis=0)

    # Normalize to sphere
    norm = np.linalg.norm(avg)
    if norm < 1e-30:
        n = X.shape[-1]
        return np.full(n, 1.0 / n)
    avg = avg / norm

    return fisher_project(avg)


def fisher_barycenter(
    X: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """Alias for :func:`fisher_mean`."""
    return fisher_mean(X, weights=weights)


# ---------------------------------------------------------------------------
# 4.2.4 — Geodesics and interpolation
# ---------------------------------------------------------------------------


def fisher_geodesic(
    a: ArrayLike,
    b: ArrayLike,
    t: float,
) -> NDArray[np.floating]:
    """Geodesic (slerp) between two simplex compositions.

    Computes spherical linear interpolation in Fisher amplitude coordinates,
    then projects back to the simplex.

    Parameters
    ----------
    a, b : array_like
        Simplex compositions of shape ``(N,)``.
    t : float
        Interpolation parameter in ``[0, 1]``.

    Returns
    -------
    ndarray
        Shape ``(N,)`` simplex composition at parameter *t*.
    """
    a = _validated(a)
    b = _validated(b)
    psi_a = fisher_lift(a)
    psi_b = fisher_lift(b)

    dot_val = np.clip(np.dot(psi_a, psi_b), -1.0, 1.0)
    omega = np.arccos(dot_val)

    if omega < 1e-12:
        # Degenerate case: a ≈ b
        return a.copy()

    # Exact endpoints
    if t == 0.0:
        return a.copy()
    if t == 1.0:
        return b.copy()

    sin_omega = np.sin(omega)
    psi_t = (
        np.sin((1.0 - t) * omega) / sin_omega * psi_a
        + np.sin(t * omega) / sin_omega * psi_b
    )
    return psi_t**2


def geodesic_interpolate(
    a: ArrayLike,
    b: ArrayLike,
    ts: ArrayLike,
) -> NDArray[np.floating]:
    """Evaluate the geodesic at multiple parameter values.

    Parameters
    ----------
    a, b : array_like
        Simplex compositions of shape ``(N,)``.
    ts : array_like
        Array of parameter values.

    Returns
    -------
    ndarray
        Shape ``(len(ts), N)`` array of simplex compositions.
    """
    ts = np.asarray(ts, dtype=np.float64)
    return np.array([fisher_geodesic(a, b, float(t)) for t in ts])


# ---------------------------------------------------------------------------
# 4.2.5 — Tangent-space tools
# ---------------------------------------------------------------------------


def fisher_logmap(
    X: ArrayLike,
    *,
    base: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """Log map on S^{N-1} at the lifted base point.

    Maps simplex compositions to tangent vectors at the base point in the
    Fisher-lifted spherical geometry, restricted to the positive orthant.

    Status: exact-derived utility.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)`` or ``(N,)``.
    base : array_like or None, optional
        Base point on the simplex. If ``None``, uses ``fisher_mean(X)``.

    Returns
    -------
    ndarray
        Tangent vectors of shape ``(M, N)`` or ``(N,)``.

    Notes
    -----
    **Boundary behavior:** When the base point lies on the simplex boundary
    (has zero components), the effective tangent space dimension drops.
    The tangent vector has zeros in the corresponding components, and the
    tangent space is degenerate in those directions. The function remains
    well-defined but emits a warning. See the mathematical notes for details.
    """
    X = _validated(X)
    single = X.ndim == 1
    if single:
        X = X[np.newaxis, :]

    if base is None:
        base = fisher_mean(X)
    else:
        base = _validated(base)

    if np.any(base == 0):
        warnings.warn(
            "Base point has zero components; log map may be undefined.",
            stacklevel=2,
        )

    psi_base = fisher_lift(base)  # (N,)
    psi_X = fisher_lift(X)  # (M, N)

    dots = np.clip(np.sum(psi_X * psi_base, axis=-1), -1.0, 1.0)  # (M,)
    theta = np.arccos(dots)  # (M,)

    # Direction: project out the base component, then normalize
    direction = psi_X - dots[:, np.newaxis] * psi_base  # (M, N)
    dir_norms = np.linalg.norm(direction, axis=-1, keepdims=True)  # (M, 1)

    safe_norms = np.where(dir_norms > 1e-30, dir_norms, 1.0)
    unit_dir = direction / safe_norms

    V = theta[:, np.newaxis] * unit_dir
    V = np.where(dir_norms > 1e-30, V, 0.0)

    if single:
        return V[0]
    return V


def fisher_expmap(
    V: ArrayLike,
    *,
    base: ArrayLike,
) -> NDArray[np.floating]:
    """Exp map: inverse of the log map on S^{N-1}.

    Maps tangent vectors back to simplex compositions.

    Parameters
    ----------
    V : array_like
        Tangent vectors of shape ``(M, N)`` or ``(N,)``.
    base : array_like
        Base point on the simplex of shape ``(N,)``.

    Returns
    -------
    ndarray
        Simplex compositions of shape ``(M, N)`` or ``(N,)``.
    """
    V = np.asarray(V, dtype=np.float64)
    base = _validated(base)
    single = V.ndim == 1
    if single:
        V = V[np.newaxis, :]

    psi_base = fisher_lift(base)  # (N,)

    theta = np.linalg.norm(V, axis=-1, keepdims=True)  # (M, 1)

    safe_theta = np.where(theta > 1e-30, theta, 1.0)
    unit_V = V / safe_theta

    psi_result = np.cos(theta) * psi_base + np.sin(theta) * unit_V
    psi_result = np.where(theta > 1e-30, psi_result, psi_base)

    result = psi_result**2

    if single:
        return result[0]
    return result


def tangent_map(
    X: ArrayLike,
    *,
    base: ArrayLike | None = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute tangent vectors and base point.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    base : array_like or None, optional
        Base point. If ``None``, uses ``fisher_mean(X)``.

    Returns
    -------
    tuple of ndarray
        ``(tangent_vectors, base_point)`` where tangent_vectors has shape
        ``(M, N)`` and base_point has shape ``(N,)``.
    """
    X = _validated(X)
    if base is None:
        base_point = fisher_mean(X)
    else:
        base_point = _validated(base)
    V = fisher_logmap(X, base=base_point)
    return V, base_point


# ---------------------------------------------------------------------------
# 4.2.6 — Tangent PCA
# ---------------------------------------------------------------------------


def fisher_pca(
    X: ArrayLike,
    *,
    base: ArrayLike | None = None,
    n_components: int | None = None,
    center: bool = True,
) -> dict:
    """PCA in the tangent space of the Fisher-lifted spherical geometry.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    base : array_like or None, optional
        Base point. If ``None``, uses ``fisher_mean(X)``.
    n_components : int or None, optional
        Number of components to keep. ``None`` keeps ``min(M, N)``.
    center : bool, optional
        Whether to center the tangent vectors before SVD. Default ``True``.

    Returns
    -------
    dict
        Keys: ``"base"`` (N,), ``"tangent_coords"`` (M, N),
        ``"components"`` (K, N), ``"singular_values"`` (K,),
        ``"explained_variance"`` (K,), ``"explained_variance_ratio"`` (K,),
        ``"scores"`` (M, K).
    """
    X = _validated(X)
    m = X.shape[0]
    n = X.shape[-1]

    if base is None:
        base_point = fisher_mean(X)
    else:
        base_point = _validated(base)

    V = fisher_logmap(X, base=base_point)

    if center:
        V_centered = V - V.mean(axis=0)
    else:
        V_centered = V

    U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)

    k = min(m, n)
    if n_components is not None:
        k = min(n_components, k)

    components = Vt[:k]
    singular_values = S[:k]
    explained_variance = S[:k] ** 2 / max(m - 1, 1)
    total_var = np.sum(S**2) / max(m - 1, 1)
    if total_var > 0:
        explained_variance_ratio = explained_variance / total_var
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)
    scores = V_centered @ components.T

    return {
        "base": base_point,
        "tangent_coords": V,
        "components": components,
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# 4.2.7 — Utility geometry
# ---------------------------------------------------------------------------


def sample_near(
    s: ArrayLike,
    *,
    scale: float,
    geometry: str = "fisher",
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Sample a point near *s* on the simplex.

    Parameters
    ----------
    s : array_like
        Simplex composition of shape ``(N,)``.
    scale : float
        Perturbation scale.
    geometry : {"fisher"}, optional
        Perturbation geometry. Default ``"fisher"``.
    rng : numpy.random.Generator or None, optional
        Random generator. If ``None``, uses default.

    Returns
    -------
    ndarray
        Shape ``(N,)`` simplex composition near *s*.
    """
    s = _validated(s)
    if rng is None:
        rng = np.random.default_rng()
    n = s.shape[-1]

    if geometry == "fisher":
        psi_s = fisher_lift(s)
        # Random tangent vector orthogonal to psi_s
        v = rng.standard_normal(n)
        v = v - np.dot(v, psi_s) * psi_s
        norm = np.linalg.norm(v)
        if norm > 1e-30:
            v = v / norm * scale
        result = fisher_expmap(v, base=s)
        # Ensure valid simplex
        result = np.maximum(result, 0.0)
        result = result / result.sum()
        return result

    raise ValueError(f"Unknown geometry: {geometry!r}")


def perturb_simplex(
    s: ArrayLike,
    *,
    eps: float,
    mode: str = "fisher",
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Add a small perturbation to a simplex composition.

    Parameters
    ----------
    s : array_like
        Simplex composition of shape ``(N,)``.
    eps : float
        Perturbation magnitude.
    mode : {"fisher", "euclidean"}, optional
        Perturbation mode. Default ``"fisher"``.
    rng : numpy.random.Generator or None, optional
        Random generator.

    Returns
    -------
    ndarray
        Shape ``(N,)`` perturbed simplex composition.
    """
    s = _validated(s)
    if rng is None:
        rng = np.random.default_rng()

    if mode == "fisher":
        return sample_near(s, scale=eps, geometry="fisher", rng=rng)

    if mode == "euclidean":
        n = s.shape[-1]
        noise = rng.standard_normal(n) * eps
        perturbed = s + noise
        perturbed = np.maximum(perturbed, 0.0)
        total = perturbed.sum()
        if total < 1e-30:
            # Noise overwhelmed the signal; return original point
            return s.copy()
        perturbed = perturbed / total
        return perturbed

    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Online and windowed statistics
# ---------------------------------------------------------------------------


class OnlineFisherMean:
    """Incremental extrinsic Fisher mean via running amplitude average.

    Maintains a running weighted arithmetic mean in amplitude (sqrt) space
    and projects back to the simplex on query. Memory: O(N).

    Parameters
    ----------
    n_components : int
        Dimensionality of the simplex (number of bins).
    """

    def __init__(self, n_components: int) -> None:
        self._n = n_components
        self._amp_sum = np.zeros(n_components, dtype=np.float64)
        self._weight_sum = 0.0
        self._count = 0

    def update(self, s: ArrayLike, weight: float = 1.0) -> None:
        """Add a single simplex composition to the running mean.

        Parameters
        ----------
        s : array_like
            Simplex composition of shape ``(N,)``.
        weight : float, optional
            Non-negative weight. Default 1.0.
        """
        s = _validated(s)
        amp = fisher_lift(s)
        self._amp_sum += weight * amp
        self._weight_sum += weight
        self._count += 1

    def update_batch(self, X: ArrayLike, weights: ArrayLike | None = None) -> None:
        """Add a batch of simplex compositions.

        Parameters
        ----------
        X : array_like
            Simplex compositions of shape ``(M, N)``.
        weights : array_like or None, optional
            Non-negative weights of shape ``(M,)``. Default uniform.
        """
        X = _validated(X)
        X = _require_batch(X, "OnlineFisherMean.update_batch")
        amps = fisher_lift(X)
        m = X.shape[0]
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            self._amp_sum += np.einsum("i,ij->j", w, amps)
            self._weight_sum += w.sum()
        else:
            self._amp_sum += amps.sum(axis=0)
            self._weight_sum += m
        self._count += m

    @property
    def mean(self) -> NDArray[np.floating]:
        """Current extrinsic Fisher mean projected to the simplex."""
        if self._weight_sum <= 0:
            return np.full(self._n, 1.0 / self._n)
        avg = self._amp_sum / self._weight_sum
        norm = np.linalg.norm(avg)
        if norm < 1e-30:
            return np.full(self._n, 1.0 / self._n)
        avg = avg / norm
        return fisher_project(avg)

    @property
    def count(self) -> int:
        """Number of compositions seen."""
        return self._count

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._amp_sum[:] = 0.0
        self._weight_sum = 0.0
        self._count = 0


class WindowedFisherStats:
    """Rolling Fisher-geometric statistics over a fixed-size circular buffer.

    Memory: O(window_size * N).

    Parameters
    ----------
    n_components : int
        Dimensionality of the simplex (number of bins).
    window_size : int
        Maximum number of compositions to keep.
    """

    def __init__(self, n_components: int, window_size: int) -> None:
        self._n = n_components
        self._window_size = window_size
        self._buffer = np.zeros((window_size, n_components), dtype=np.float64)
        self._pos = 0
        self._count = 0

    def push(self, s: ArrayLike) -> None:
        """Add a single simplex composition.

        Parameters
        ----------
        s : array_like
            Simplex composition of shape ``(N,)``.
        """
        s = _validated(s)
        self._buffer[self._pos % self._window_size] = s
        self._pos += 1
        self._count = min(self._count + 1, self._window_size)

    def push_batch(self, X: ArrayLike) -> None:
        """Add a batch of simplex compositions.

        Parameters
        ----------
        X : array_like
            Simplex compositions of shape ``(M, N)``.
        """
        X = _validated(X)
        X = _require_batch(X, "WindowedFisherStats.push_batch")
        for row in X:
            self.push(row)

    def _active(self) -> NDArray[np.floating]:
        """Return the active portion of the buffer."""
        if self._count < self._window_size:
            return self._buffer[: self._count]
        return self._buffer

    @property
    def mean(self) -> NDArray[np.floating]:
        """Extrinsic Fisher mean of the window contents."""
        buf = self._active()
        if buf.shape[0] == 0:
            return np.full(self._n, 1.0 / self._n)
        return fisher_mean(buf)

    @property
    def dispersion(self) -> float:
        """Mean pairwise Fisher distance within the window."""
        buf = self._active()
        if buf.shape[0] < 2:
            return 0.0
        D = pairwise_fisher_distances(buf)
        m = buf.shape[0]
        return float(D.sum() / (m * (m - 1)))

    @property
    def forced_pair_mean(self) -> NDArray[np.floating]:
        """Mean forced-pair coordinates [Q_delta, H_3] over the window."""
        buf = self._active()
        if buf.shape[0] == 0:
            return np.zeros(2)
        q_vals, h_vals = forced_pair(buf)
        return np.array([q_vals.mean(), h_vals.mean()])

    @property
    def count(self) -> int:
        """Number of compositions in the window."""
        return self._count

    def shift_from(self, reference: ArrayLike) -> float:
        """Fisher distance from *reference* to the window mean.

        Parameters
        ----------
        reference : array_like
            Simplex composition of shape ``(N,)``.

        Returns
        -------
        float
            Fisher geodesic distance.
        """
        reference = _validated(reference)
        m = self.mean
        return float(fisher_distance(m, reference))


# ---------------------------------------------------------------------------
# Cross-cloud distances
# ---------------------------------------------------------------------------


def cross_fisher_distances(
    X: ArrayLike,
    Y: ArrayLike,
) -> NDArray[np.floating]:
    """Pairwise Fisher distances between two clouds.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    Y : array_like
        Simplex compositions of shape ``(K, N)``.

    Returns
    -------
    ndarray
        Shape ``(M, K)`` distance matrix.
    """
    X = _validated(X)
    Y = _validated(Y)
    X = _require_batch(X, "cross_fisher_distances X")
    Y = _require_batch(Y, "cross_fisher_distances Y")
    psi_x = fisher_lift(X)  # (M, N)
    psi_y = fisher_lift(Y)  # (K, N)
    bc = psi_x @ psi_y.T  # (M, K)
    bc = np.clip(bc, -1.0, 1.0)
    D = 2.0 * np.arccos(bc)
    return D
