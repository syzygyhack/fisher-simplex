# fisher-simplex

A boundary-safe information-geometry toolkit for probability-simplex data, built around the Fisher/Hellinger square-root embedding and extended with canonical low-complexity invariant diagnostics.

Use it when your data are normalized nonnegative vectors and you need principled distances, means, interpolation, and low-order structural summaries without ad hoc zero handling.

Need a different language than Python? `fisher-simplex` maintains a [SPEC.md](/SPEC.md) in order to serve as a ghost library. Clone the repository as reference and point your favorite coding agent at the specification to reproduce with your unique requirements met.

## Positioning

**Primarily a geometry toolkit** — exact Fisher/Hellinger distances, means, geodesics, tangent PCA, and kernel methods on the probability simplex.

**Secondarily a low-complexity invariant toolkit** — canonical overlap diagnostics and forced-pair summaries (Q_delta, H_3) grounded in the Fisher-lifted symmetric-even harmonic sector.

## Installation

```bash
uv add fisher-simplex
# or
pip install fisher-simplex
```

For visualization support:

```bash
uv add "fisher-simplex[viz]"
```

## Quick usage

```python
import numpy as np
import fisher_simplex as fs

# Two probability vectors
a = np.array([0.5, 0.3, 0.2])
b = np.array([0.1, 0.6, 0.3])

# Fisher distance (geodesic distance on the simplex)
d = fs.fisher_distance(a, b)
print(f"Fisher distance: {d:.4f}")

# Fisher mean of a cloud
cloud = np.array([
    [0.5, 0.3, 0.2],
    [0.1, 0.6, 0.3],
    [0.3, 0.3, 0.4],
])
mean = fs.fisher_mean(cloud)
print(f"Fisher mean: {mean}")

# Forced coordinates (Q_delta, H_3)
coords = fs.forced_coordinates(cloud)
print(f"Forced coordinates:\n{coords}")

# Divergence analysis
report = fs.divergence_analysis(cloud)
print(f"Mean divergence: {report['mean_divergence']:.6f}")
```

## Module overview

| Module | Purpose | Status |
|--------|---------|--------|
| `core` | Scalar invariants, overlap families, forced pair, Fisher lift, bridge statistics | Exact |
| `geometry` | Distances, geodesics, means, tangent maps, PCA, kernels | Exact / exact-derived |
| `analysis` | Batch diagnostics, divergence reports, forced-block analysis, text reports | Exact-derived / heuristic |
| `generators` | Synthetic reference ensembles (Dirichlet, broken stick, lognormal, etc.) | Engineering |
| `utils` | Validation, projection, closure | Engineering |
| `frontier` | Degree-8 enrichment coordinates and residual diagnostics | Experimental |
| `harmonic` | Low-degree symmetric-even harmonic tools | Experimental |
| `viz` | Matplotlib plotting utilities (requires `matplotlib`) | Optional |

## Examples

The `examples/` directory contains runnable scripts demonstrating domain-specific workflows:

| Example | Domain | Demonstrates |
|---------|--------|-------------|
| `basic_geometry.py` | General | Fisher distance, mean, geodesics, tangent PCA, kernels |
| `portfolio_rebalancing.py` | Finance | Fisher vs Euclidean distance, geodesic rebalancing, HHI shape correction |
| `biodiversity_concentration.py` | Ecology / Economics | Overlap diagnostics, forced-pair invariants, concentration profiling |
| `distributional_shift.py` | Machine learning | Shift detection between classifier output clouds |
| `sparse_zeros.py` | Microbiome | Boundary-safe geometry on zero-inflated compositional data |
| `frontier_enrichment.py` | Research | Degree-8 enrichment coordinates, residual diagnostics |

## Documentation

- [Quick start](docs/quickstart.md) — five steps to get going
- [User guide](docs/user_guide.md) — workflows for geometry, diagnostics, and analysis
- [Mathematical notes](docs/mathematical_notes.md) — derivations, theorems, and conventions

## Dependencies

**Required:** Python >= 3.9, NumPy >= 1.21, SciPy >= 1.7

**Optional:** matplotlib >= 3.5 (viz module)

## License

MIT

## Citation

Software:

> `fisher-simplex`: Canonical simplex geometry via the Fisher amplitude lift, with low-complexity invariant diagnostics. Version 0.3.

Mathematical attribution:

> Fisher-Harmonic programme: overlap-family divergence, low-degree forced compression in the Fisher-lifted symmetric-even sector, and degree-8 selective frontier tools for simplex observables.
