"""High-level analysis workflows combining exact tools into interpretable reports.

Implements batch diagnostics, overlap divergence analysis, forced-block regression,
text reports, community classification, and distributional shift detection
as specified in spec section 4.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from fisher_simplex.core import (
    forced_coordinates,
    h3,
    herfindahl,
    overlap_divergence,
    phi,
    psi_overlap,
    q_delta,
    qh_ratio,
    shannon_entropy,
    simpson_index,
)
from fisher_simplex.geometry import fisher_distance, fisher_mean
from fisher_simplex.utils import validate_simplex

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validated(x: ArrayLike) -> NDArray[np.floating]:
    """Validate and return a simplex array."""
    return validate_simplex(x, renormalize="warn")


def _ensure_2d(x: NDArray) -> NDArray:
    """Ensure array is at least 2D for batch operations."""
    if x.ndim == 1:
        return x[np.newaxis, :]
    return x


# ---------------------------------------------------------------------------
# 4.3.1 — Batch diagnostics
# ---------------------------------------------------------------------------


def batch_diagnostic(
    X: ArrayLike,
    *,
    include_heuristics: bool = False,
) -> dict:
    """Compute all core diagnostics for a batch of simplex compositions.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    include_heuristics : bool, optional
        If ``True``, include heuristic diagnostics (e.g. ``qh_ratio``).
        Default ``False``.

    Returns
    -------
    dict
        Keys: ``"n_components"`` (int), ``"n_compositions"`` (int),
        ``"phi"``, ``"psi"``, ``"divergence"``, ``"q_delta"``, ``"h3"``,
        ``"herfindahl"``, ``"simpson"``, ``"shannon"`` (ndarray shape (M,)),
        ``"fisher_coords"`` (ndarray shape (M, N)).
        If *include_heuristics* is ``True``, also ``"qh_ratio"`` (ndarray shape (M,)).
    """
    X = _validated(X)
    X = _ensure_2d(X)
    m, n = X.shape

    result: dict = {
        "n_components": n,
        "n_compositions": m,
        "phi": phi(X),
        "psi": psi_overlap(X),
        "divergence": overlap_divergence(X),
        "q_delta": q_delta(X),
        "h3": h3(X),
        "herfindahl": herfindahl(X),
        "simpson": simpson_index(X),
        "shannon": shannon_entropy(X),
        "fisher_coords": forced_coordinates(X),
    }

    if include_heuristics:
        result["qh_ratio"] = qh_ratio(X)

    return result


def full_diagnostic(s: ArrayLike) -> dict:
    """Compute all core diagnostics for a single simplex composition.

    Same keys as :func:`batch_diagnostic` but with scalar values.
    ``fisher_coords`` is shape ``(N,)`` instead of ``(M, N)``.

    Parameters
    ----------
    s : array_like
        Single simplex composition of shape ``(N,)``.

    Returns
    -------
    dict
        Diagnostic values as scalars (or shape ``(N,)`` for fisher_coords).
    """
    s = _validated(s)
    if s.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {s.shape}")
    n = s.shape[0]

    return {
        "n_components": n,
        "n_compositions": 1,
        "phi": float(phi(s)),
        "psi": float(psi_overlap(s)),
        "divergence": float(overlap_divergence(s)),
        "q_delta": float(q_delta(s)),
        "h3": float(h3(s)),
        "herfindahl": float(herfindahl(s)),
        "simpson": float(simpson_index(s)),
        "shannon": float(shannon_entropy(s)),
        "fisher_coords": forced_coordinates(s),
    }


# ---------------------------------------------------------------------------
# 4.3.2 — Overlap divergence analysis
# ---------------------------------------------------------------------------


def pairwise_ranking_disagreement(
    a: ArrayLike,
    b: ArrayLike,
) -> dict:
    """Compute pairwise ranking disagreement between two score vectors.

    Parameters
    ----------
    a, b : array_like
        Score vectors of shape ``(M,)``.

    Returns
    -------
    dict
        Keys: ``"disagreement_rate"`` (float), ``"concordance_tau"`` (float,
        Kendall's tau-b), ``"concordance_p"`` (float), ``"n_pairs"`` (int),
        ``"n_disagreements"`` (int).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m = len(a)
    n_pairs = m * (m - 1) // 2

    # Count disagreements: pairs where ordering differs
    n_disagreements = 0
    for i in range(m):
        for j in range(i + 1, m):
            sign_a = np.sign(a[i] - a[j])
            sign_b = np.sign(b[i] - b[j])
            if sign_a != 0 and sign_b != 0 and sign_a != sign_b:
                n_disagreements += 1

    disagreement_rate = n_disagreements / n_pairs if n_pairs > 0 else 0.0

    tau_result = stats.kendalltau(a, b)
    concordance_tau = float(tau_result.statistic)
    concordance_p = float(tau_result.pvalue)

    return {
        "disagreement_rate": disagreement_rate,
        "concordance_tau": concordance_tau,
        "concordance_p": concordance_p,
        "n_pairs": n_pairs,
        "n_disagreements": n_disagreements,
    }


def divergence_analysis(
    X: ArrayLike,
    *,
    threshold: float = 0.10,
) -> dict:
    """Analyse overlap divergence across a batch of compositions.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    threshold : float, optional
        Divergence threshold (unused in current implementation but reserved
        for future recommendation logic). Default ``0.10``.

    Returns
    -------
    dict
        Keys: ``"n_components"`` (int), ``"mean_divergence"``,
        ``"max_divergence"``, ``"divergence_std"`` (float),
        ``"fraction_consequential"`` (float, fraction where ``|D| > 0.01``),
        ``"ranking_disagreement"`` (dict from :func:`pairwise_ranking_disagreement`),
        ``"recommendation"`` (str, labeled as empirical heuristic).
    """
    X = _validated(X)
    X = _ensure_2d(X)
    n = X.shape[-1]

    div = overlap_divergence(X)
    abs_div = np.abs(div)

    mean_div = float(np.mean(div))
    max_div = float(np.max(abs_div))
    div_std = float(np.std(div))
    fraction_consequential = float(np.mean(abs_div > 0.01))

    # Ranking disagreement between phi and psi
    phi_vals = phi(X)
    psi_vals = psi_overlap(X)
    ranking = pairwise_ranking_disagreement(phi_vals, psi_vals)

    # Recommendation (empirical heuristic)
    if mean_div < 0.01 and max_div < threshold:
        recommendation = (
            "[Empirical heuristic] Phi and Psi are interchangeable for this data. "
            "Divergence is negligible."
        )
    elif max_div < threshold:
        recommendation = (
            "[Empirical heuristic] Moderate divergence detected. "
            "Phi and Psi may yield similar rankings but differ in magnitude."
        )
    else:
        recommendation = (
            "[Empirical heuristic] Substantial divergence detected. "
            "Phi and Psi may disagree on rankings. Consider using both "
            "and reporting the divergence."
        )

    return {
        "n_components": n,
        "mean_divergence": mean_div,
        "max_divergence": max_div,
        "divergence_std": div_std,
        "fraction_consequential": fraction_consequential,
        "ranking_disagreement": ranking,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 4.3.3 — Forced-block analysis
# ---------------------------------------------------------------------------


def _build_features(
    coords: NDArray,
    model: str,
) -> NDArray:
    """Build feature matrix from forced coordinates.

    Parameters
    ----------
    coords : ndarray
        Shape ``(M, 2)`` forced coordinates ``[Q, H]``.
    model : {"linear", "quadratic"}
        Feature set.

    Returns
    -------
    ndarray
        Feature matrix with intercept column prepended.
    """
    q = coords[:, 0]
    h = coords[:, 1]

    if model == "linear":
        return np.column_stack([np.ones_like(q), q, h])
    if model == "quadratic":
        return np.column_stack([
            np.ones_like(q), q, h, q**2, q * h, h**2,
        ])
    raise ValueError(f"Unknown model: {model!r}")


def _ols_r_squared(features: NDArray, target: NDArray) -> tuple[float, NDArray]:
    """Fit OLS and return (R-squared, residuals)."""
    coeffs, residuals_sum, _, _ = np.linalg.lstsq(features, target, rcond=None)
    predictions = features @ coeffs
    resid = target - predictions
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r_squared, resid


def sufficient_statistic_efficiency(
    X: ArrayLike,
    target: ArrayLike,
    *,
    model: str = "linear",
) -> dict:
    """Assess whether a target variable lies in the forced block.

    Builds features from :func:`~fisher_simplex.core.forced_coordinates`:
    linear = ``[1, Q, H]``, quadratic = ``[1, Q, H, Q^2, QH, H^2]``.
    Fits OLS via ``numpy.linalg.lstsq``.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    target : array_like
        Target values of shape ``(M,)``.
    model : {"linear", "quadratic"}, optional
        Feature set. Default ``"linear"``.

    Returns
    -------
    dict
        Keys: ``"r_squared_linear"`` (float), ``"r_squared_quadratic"`` (float),
        ``"in_forced_block"`` (bool, ``R^2_quad > 0.999``),
        ``"residual_std"`` (float).
    """
    X = _validated(X)
    X = _ensure_2d(X)
    target = np.asarray(target, dtype=np.float64)

    coords = forced_coordinates(X)

    feat_lin = _build_features(coords, "linear")
    r2_lin, _ = _ols_r_squared(feat_lin, target)

    feat_quad = _build_features(coords, "quadratic")
    r2_quad, resid_quad = _ols_r_squared(feat_quad, target)

    return {
        "r_squared_linear": r2_lin,
        "r_squared_quadratic": r2_quad,
        "in_forced_block": r2_quad > 0.999,
        "residual_std": float(np.std(resid_quad)),
    }


def forced_block_regression(
    X: ArrayLike,
    target: ArrayLike,
    *,
    basis: str = "pair",
) -> dict:
    """Regression of a target variable onto forced-block coordinates.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    target : array_like
        Target values of shape ``(M,)``.
    basis : {"pair"}, optional
        Coordinate basis. Default ``"pair"`` uses ``(Q_delta, H_3)``.

    Returns
    -------
    dict
        Keys: ``"coefficients"`` (ndarray), ``"predictions"`` (ndarray shape (M,)),
        ``"residuals"`` (ndarray shape (M,)), ``"r_squared"`` (float).
    """
    X = _validated(X)
    X = _ensure_2d(X)
    target = np.asarray(target, dtype=np.float64)

    if basis != "pair":
        raise ValueError(f"Unknown basis: {basis!r}")

    coords = forced_coordinates(X)
    features = _build_features(coords, "quadratic")

    coeffs, _, _, _ = np.linalg.lstsq(features, target, rcond=None)
    predictions = features @ coeffs
    residuals = target - predictions

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "coefficients": coeffs,
        "predictions": predictions,
        "residuals": residuals,
        "r_squared": r_squared,
    }


# ---------------------------------------------------------------------------
# 4.3.4 — Reports
# ---------------------------------------------------------------------------


def comparative_index_report(X: ArrayLike) -> str:
    """Generate a text report comparing one-number summary indices.

    Economist-friendly language describing where indices agree and disagree.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.

    Returns
    -------
    str
        Multi-line text report.
    """
    X = _validated(X)
    X = _ensure_2d(X)
    m, n = X.shape

    diag = batch_diagnostic(X)
    div_info = divergence_analysis(X)

    hhi_m = np.mean(diag["herfindahl"])
    hhi_s = np.std(diag["herfindahl"])
    sim_m = np.mean(diag["simpson"])
    sim_s = np.std(diag["simpson"])
    sha_m = np.mean(diag["shannon"])
    sha_s = np.std(diag["shannon"])
    phi_m = np.mean(diag["phi"])
    phi_s = np.std(diag["phi"])
    psi_m = np.mean(diag["psi"])
    psi_s = np.std(diag["psi"])
    tau = div_info["ranking_disagreement"]["concordance_tau"]

    lines = [
        "Comparative Index Report",
        "=" * 50,
        f"Compositions: {m}  Components: {n}",
        "",
        "Summary Statistics (mean +/- std):",
        f"  HHI (Herfindahl):  {hhi_m:.4f} +/- {hhi_s:.4f}",
        f"  Simpson (1 - HHI): {sim_m:.4f} +/- {sim_s:.4f}",
        f"  Shannon entropy:   {sha_m:.4f} +/- {sha_s:.4f}",
        f"  Phi overlap:       {phi_m:.4f} +/- {phi_s:.4f}",
        f"  Psi overlap:       {psi_m:.4f} +/- {psi_s:.4f}",
        "",
        "Overlap Divergence (Phi - Psi):",
        f"  Mean divergence:    {div_info['mean_divergence']:.6f}",
        f"  Max |divergence|:   {div_info['max_divergence']:.6f}",
        f"  Fraction |D|>0.01: "
        f"{div_info['fraction_consequential']:.2%}",
        "",
        f"Ranking agreement (Kendall tau-b): {tau:.4f}",
        "",
        div_info["recommendation"],
    ]
    return "\n".join(lines)


def concentration_profile_report(X: ArrayLike) -> str:
    """Generate a text report on concentration using HHI and shape correction.

    Reports HHI alongside the forced-pair invariants ``(Q_delta, H_3)``
    which provide shape correction beyond the single-number HHI summary.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.

    Returns
    -------
    str
        Multi-line text report.
    """
    X = _validated(X)
    X = _ensure_2d(X)
    m, n = X.shape

    hhi_vals = herfindahl(X)
    q_vals = q_delta(X)
    h_vals = h3(X)

    lines = [
        "Concentration Profile Report",
        "=" * 50,
        f"Compositions: {m}  Components: {n}",
        "",
        "HHI (Herfindahl-Hirschman Index):",
        f"  Mean: {np.mean(hhi_vals):.4f}  Std: {np.std(hhi_vals):.4f}",
        f"  Min:  {np.min(hhi_vals):.4f}  Max: {np.max(hhi_vals):.4f}",
        "",
        "Shape Correction via Forced-Pair (Q_delta, H_3):",
        f"  Q_delta mean: {np.mean(q_vals):.6f}  std: {np.std(q_vals):.6f}",
        f"  H_3 mean:     {np.mean(h_vals):.6f}  std: {np.std(h_vals):.6f}",
        "",
        "Interpretation:",
        "  Q_delta measures excess concentration beyond uniform.",
        "  H_3 captures skewness in the share distribution.",
        "  Together they provide a two-dimensional shape correction",
        "  that distinguishes distributions with identical HHI values.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4.3.5 — Classification
# ---------------------------------------------------------------------------


def community_type_discriminant(
    X: ArrayLike,
    *,
    method: str = "pair",
    calibrator: str | None = None,
) -> dict:
    """Classify community types based on forced-pair coordinates.

    .. warning::
        Empirical heuristic. Classification boundaries are data-dependent
        and should not be treated as universal thresholds.

    Parameters
    ----------
    X : array_like
        Simplex compositions of shape ``(M, N)``.
    method : {"pair"}, optional
        Classification method. ``"pair"`` operates on 2D forced coordinates
        directly. Default ``"pair"``.
    calibrator : {None, "synthetic_v0"}, optional
        Optional synthetic calibration presets. ``None`` uses default
        thresholds. ``"synthetic_v0"`` uses presets tuned on synthetic
        generators. Not universal defaults.

    Returns
    -------
    dict
        Keys: ``"labels"`` (ndarray of str, shape (M,)),
        ``"scores"`` (ndarray shape (M, 2)),
        ``"method"`` (str), ``"calibrator"`` (str or None).
    """
    X = _validated(X)
    X = _ensure_2d(X)
    m = X.shape[0]

    if method != "pair":
        raise ValueError(f"Unknown method: {method!r}")

    coords = forced_coordinates(X)
    q_vals = coords[:, 0]
    h_vals = coords[:, 1]

    # Default thresholds
    q_thresh = 0.05
    h_thresh = 0.001

    if calibrator == "synthetic_v0":
        q_thresh = 0.08
        h_thresh = 0.002
    elif calibrator is not None:
        raise ValueError(f"Unknown calibrator: {calibrator!r}")

    labels = np.empty(m, dtype=object)
    for i in range(m):
        if q_vals[i] > q_thresh and h_vals[i] > h_thresh:
            labels[i] = "concentrated"
        elif q_vals[i] < -q_thresh * 0.5:
            labels[i] = "dispersed"
        else:
            labels[i] = "moderate"

    return {
        "labels": labels,
        "scores": coords,
        "method": method,
        "calibrator": calibrator,
    }


# ---------------------------------------------------------------------------
# 4.3.6 — Distributional shift
# ---------------------------------------------------------------------------


def distributional_shift(
    X_ref: ArrayLike,
    X_test: ArrayLike,
    *,
    summary: str = "mean",
) -> dict:
    """Detect distributional shift between reference and test ensembles.

    Parameters
    ----------
    X_ref : array_like
        Reference simplex compositions of shape ``(M_ref, N)``.
    X_test : array_like
        Test simplex compositions of shape ``(M_test, N)``.
    summary : {"mean"}, optional
        Summary statistic for aggregation. Default ``"mean"``.

    Returns
    -------
    dict
        Keys: ``"mean_distance"`` (float, mean Fisher distance between
        nearest pairs), ``"cloud_distance"`` (float, Fisher distance
        between Fisher means), ``"ref_dispersion"`` (float, mean Fisher
        distance from ref points to ref mean), ``"test_dispersion"``
        (float, same for test).
    """
    X_ref = _validated(X_ref)
    X_test = _validated(X_test)
    X_ref = _ensure_2d(X_ref)
    X_test = _ensure_2d(X_test)

    # Fisher means
    ref_mean = fisher_mean(X_ref)
    test_mean = fisher_mean(X_test)

    # Cloud distance: Fisher distance between means
    cloud_dist = float(fisher_distance(ref_mean, test_mean))

    # Pairwise Fisher distances between ref and test for nearest-pair matching
    m_ref = X_ref.shape[0]
    m_test = X_test.shape[0]

    # Build cross-distance matrix
    cross_dists = np.zeros((m_ref, m_test))
    for i in range(m_ref):
        for j in range(m_test):
            cross_dists[i, j] = fisher_distance(X_ref[i], X_test[j])

    # Mean distance between nearest pairs (from ref to nearest test)
    min_dists_ref = np.min(cross_dists, axis=1)
    min_dists_test = np.min(cross_dists, axis=0)
    mean_distance = float(np.mean(np.concatenate([min_dists_ref, min_dists_test])))

    # Dispersions
    ref_dists = np.array([
        float(fisher_distance(X_ref[i], ref_mean)) for i in range(m_ref)
    ])
    ref_dispersion = float(np.mean(ref_dists))

    test_dists = np.array([
        float(fisher_distance(X_test[i], test_mean)) for i in range(m_test)
    ])
    test_dispersion = float(np.mean(test_dists))

    return {
        "mean_distance": mean_distance,
        "cloud_distance": cloud_dist,
        "ref_dispersion": ref_dispersion,
        "test_dispersion": test_dispersion,
    }
