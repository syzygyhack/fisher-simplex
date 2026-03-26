"""Fisher-Simplex: canonical simplex geometry via the Fisher amplitude lift."""

__version__ = "0.3.0"

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
from fisher_simplex.utils import closure, project_to_simplex, validate_simplex

__all__ = [
    "binary_fisher_angle",
    "binary_overlap",
    "closure",
    "effective_number_herfindahl",
    "effective_number_shannon",
    "fisher_lift",
    "fisher_project",
    "forced_coordinates",
    "forced_pair",
    "gini_simpson",
    "h3",
    "herfindahl",
    "overlap_divergence",
    "phi",
    "project_to_simplex",
    "psi",
    "psi_overlap",
    "q_delta",
    "qh_ratio",
    "shannon_entropy",
    "simpson_index",
    "validate_simplex",
]
