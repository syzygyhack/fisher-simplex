"""Fisher-Simplex: canonical simplex geometry via the Fisher amplitude lift."""

__version__ = "0.3.0"

from fisher_simplex.analysis import (
    batch_diagnostic,
    community_type_discriminant,
    comparative_index_report,
    concentration_profile_report,
    distributional_shift,
    divergence_analysis,
    forced_block_regression,
    full_diagnostic,
    pairwise_ranking_disagreement,
    sufficient_statistic_efficiency,
)
from fisher_simplex.core import (
    binary_fisher_angle,
    binary_overlap,
    effective_number_herfindahl,
    effective_number_shannon,
    fisher_lift,
    fisher_project,
    forced_coordinates,
    forced_pair,
    gini_simpson,
    h3,
    herfindahl,
    overlap_divergence,
    phi,
    psi,
    psi_overlap,
    q_delta,
    qh_ratio,
    shannon_entropy,
    simpson_index,
)
from fisher_simplex.generators import (
    broken_stick,
    dirichlet_community,
    geometric_series,
    lognormal_community,
    market_shares,
    portfolio_weights,
)
from fisher_simplex.geometry import (
    bhattacharyya_coefficient,
    fisher_barycenter,
    fisher_cosine,
    fisher_distance,
    fisher_expmap,
    fisher_geodesic,
    fisher_kernel,
    fisher_logmap,
    fisher_mean,
    fisher_pca,
    geodesic_interpolate,
    hellinger_distance,
    kernel_matrix,
    pairwise_fisher_distances,
    pairwise_hellinger_distances,
    perturb_simplex,
    sample_near,
    tangent_map,
)
from fisher_simplex.utils import closure, project_to_simplex, validate_simplex

# Optional modules — available via fisher_simplex.frontier, fisher_simplex.harmonic,
# fisher_simplex.viz. Not re-exported at top level to avoid optional dependency issues.

__all__ = [
    # core — overlap families
    "phi",
    "psi",
    "psi_overlap",
    "overlap_divergence",
    # core — forced pair
    "q_delta",
    "h3",
    "forced_pair",
    "forced_coordinates",
    "qh_ratio",
    # core — Fisher lift
    "fisher_lift",
    "fisher_project",
    # core — bridge statistics
    "herfindahl",
    "simpson_index",
    "gini_simpson",
    "shannon_entropy",
    "effective_number_herfindahl",
    "effective_number_shannon",
    # core — binary helpers
    "binary_overlap",
    "binary_fisher_angle",
    # geometry — distances and coefficients
    "bhattacharyya_coefficient",
    "hellinger_distance",
    "fisher_distance",
    "pairwise_fisher_distances",
    "pairwise_hellinger_distances",
    # geometry — kernels
    "fisher_kernel",
    "fisher_cosine",
    "kernel_matrix",
    # geometry — means and barycenters
    "fisher_mean",
    "fisher_barycenter",
    # geometry — geodesics
    "fisher_geodesic",
    "geodesic_interpolate",
    # geometry — tangent space
    "fisher_logmap",
    "fisher_expmap",
    "tangent_map",
    "fisher_pca",
    # geometry — utility
    "sample_near",
    "perturb_simplex",
    # analysis
    "batch_diagnostic",
    "full_diagnostic",
    "divergence_analysis",
    "pairwise_ranking_disagreement",
    "sufficient_statistic_efficiency",
    "forced_block_regression",
    "comparative_index_report",
    "concentration_profile_report",
    "community_type_discriminant",
    "distributional_shift",
    # utils
    "validate_simplex",
    "project_to_simplex",
    "closure",
    # generators
    "dirichlet_community",
    "broken_stick",
    "lognormal_community",
    "geometric_series",
    "market_shares",
    "portfolio_weights",
]
