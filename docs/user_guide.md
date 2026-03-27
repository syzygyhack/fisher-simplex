# User guide

## Geometry workflows

### Distances

Fisher distance is the geodesic distance under the Fisher information metric: `d_F(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))`. Hellinger distance is the Euclidean distance in square-root-transformed coordinates, scaled by `1/sqrt(2)` so that `d_H in [0, 1]`. The two are related by `d_F = 2 * arccos(1 - d_H^2)`.

```python
import numpy as np
import fisher_simplex as fs

a = np.array([0.5, 0.3, 0.2])
b = np.array([0.1, 0.6, 0.3])

# Point-to-point distances
d_f = fs.fisher_distance(a, b)
d_h = fs.hellinger_distance(a, b)
bc = fs.bhattacharyya_coefficient(a, b)

# Pairwise distance matrices
cloud = fs.dirichlet_community(5, 50, rng=np.random.default_rng(0))
D_fisher = fs.pairwise_fisher_distances(cloud)    # (50, 50)
D_hellinger = fs.pairwise_hellinger_distances(cloud)  # (50, 50)
```

### Means

The Fisher mean computes a weighted arithmetic mean in amplitude (square-root) coordinates, normalizes to the unit sphere, and projects back to the simplex.

```python
mean = fs.fisher_mean(cloud)
weighted_mean = fs.fisher_mean(cloud, weights=np.ones(50) / 50)

# fisher_barycenter is an alias
bary = fs.fisher_barycenter(cloud)
```

> **Note:** This is the extrinsic (projected arithmetic) mean in Fisher-lift amplitude coordinates. It approximates but is not identical to the intrinsic Frechet mean under the Fisher geodesic distance. For data concentrated in a small region of the simplex, the two coincide to first order.

### Geodesics

Geodesics are computed via spherical linear interpolation (slerp) in amplitude coordinates.

```python
# Single point on the geodesic
midpoint = fs.fisher_geodesic(a, b, 0.5)

# Multiple points along the geodesic
ts = np.linspace(0, 1, 20)
path = fs.geodesic_interpolate(a, b, ts)  # (20, 3)
```

### Kernel methods

The Fisher kernel equals the Bhattacharyya coefficient (cosine similarity in amplitude coordinates). The library provides the full kernel family from the paper (§2.7):

```python
# Linear Fisher kernel: K(a, b) = B(a, b)
k = fs.fisher_kernel(a, b)
K = fs.kernel_matrix(cloud, kind="fisher")  # (50, 50) PSD matrix

# Polynomial Fisher kernel: K_d(a, b) = B(a, b)^d
k3 = fs.polynomial_fisher_kernel(a, b, d=3)
K_poly = fs.kernel_matrix(cloud, kind="polynomial_fisher", d=3)

# Fisher RBF kernel: exp(-d_F^2 / (2*sigma^2))
k_rbf = fs.fisher_rbf_kernel(a, b, sigma=0.5)
K_frbf = fs.kernel_matrix(cloud, kind="fisher_rbf", sigma=0.5)

# Hellinger RBF kernel: exp(-d_H^2 / sigma^2)
K_hrbf = fs.kernel_matrix(cloud, kind="hellinger_rbf", sigma=0.5)
```

The polynomial kernel `B(p,q)^d` corresponds to the inner product in the degree-d tensor power of amplitude space. The Fisher RBF kernel uses the geodesic distance directly.

## Overlap diagnostics

The two canonical overlap families on the simplex are:

- **Phi** (quadratic): `Phi_N(s) = (N/(N-1)) * (1 - sum(s_i^2))`
- **Psi** (multiplicative): `Psi_N(s) = N^N * prod(s_i)`

For N=2, they are identical. For N >= 3, they generically diverge.

```python
phi_vals = fs.phi(cloud)
psi_vals = fs.psi(cloud)
div = fs.overlap_divergence(cloud)  # Phi - Psi
```

### Divergence analysis

```python
report = fs.divergence_analysis(cloud)
# report keys: n_components, mean_divergence, max_divergence,
#   divergence_std, fraction_consequential, ranking_disagreement,
#   recommendation (labeled as empirical heuristic)
```

### Ranking disagreement

Check whether Phi and Psi disagree on the ordering of compositions:

```python
ranking = fs.pairwise_ranking_disagreement(phi_vals, psi_vals)
# ranking keys: disagreement_rate, concordance_tau, concordance_p,
#   n_pairs, n_disagreements
```

## Forced-pair interpretation

The forced pair (Q_delta, H_3) captures all nonconstant S_N-symmetric even structure through degree 6 in the Fisher-lifted sector.

- **Q_delta = sum(s_i^2) - 1/N** measures excess concentration beyond uniform.
- **H_3 = sum(s_i^3) - (3/(N+2))*sum(s_i^2) + 2/(N*(N+2))** captures cubic skewness.

```python
q = fs.q_delta(cloud)
h = fs.h3(cloud)
q_val, h_val = fs.forced_pair(cloud[0])
coords = fs.forced_coordinates(cloud)  # (M, 2) array of [Q, H]
```

### Testing forced-block membership

Test whether a target observable lies in the forced block:

```python
# p_2 (Herfindahl) is in the forced block — should show R^2 ~ 1
target = np.sum(cloud**2, axis=1)
result = fs.sufficient_statistic_efficiency(cloud, target)
print(f"R^2 linear: {result['r_squared_linear']:.4f}")
print(f"R^2 quadratic: {result['r_squared_quadratic']:.4f}")
print(f"In forced block: {result['in_forced_block']}")

# p_4 has a degree-8 enrichment component beyond the forced block —
# R^2 will be high (p_4 correlates with p_2) but not perfect
target_p4 = np.sum(cloud**4, axis=1)
result_p4 = fs.sufficient_statistic_efficiency(cloud, target_p4)
print(f"p_4 in forced block: {result_p4['in_forced_block']}")  # may be False
```

## Tangent PCA

PCA in the tangent space of the Fisher-lifted spherical geometry, not ordinary Euclidean PCA on raw simplex coordinates.

```python
pca = fs.fisher_pca(cloud, n_components=3)
print(f"Base point: {pca['base']}")
print(f"Explained variance ratio: {pca['explained_variance_ratio']}")
scores = pca['scores']  # (M, 3) projected coordinates
components = pca['components']  # (3, N) principal directions
```

The tangent map and log/exp maps are also available directly:

```python
tangent_vecs, base = fs.tangent_map(cloud)
V = fs.fisher_logmap(cloud, base=base)
recovered = fs.fisher_expmap(V, base=base)
```

## Concentration profiling

Generate economist-friendly reports comparing concentration indices:

```python
# Comparative index report
print(fs.comparative_index_report(cloud))

# Concentration profile (HHI + forced-pair shape correction)
print(fs.concentration_profile_report(cloud))
```

## Distributional shift

Compare two clouds of probability vectors under Fisher distance:

```python
ref = fs.dirichlet_community(5, 30, alpha=1.0, rng=np.random.default_rng(0))
test = fs.dirichlet_community(5, 30, alpha=0.5, rng=np.random.default_rng(1))

shift = fs.distributional_shift(ref, test)
print(f"Cloud distance: {shift['cloud_distance']:.4f}")
print(f"Mean nearest-pair distance: {shift['mean_distance']:.4f}")
print(f"Ref dispersion: {shift['ref_dispersion']:.4f}")
print(f"Test dispersion: {shift['test_dispersion']:.4f}")
```

## Forced-block regression

Regress a target variable directly onto forced-block coordinates:

```python
target = np.sum(cloud**4, axis=1)
reg = fs.forced_block_regression(cloud, target)
print(f"R^2: {reg['r_squared']:.4f}")
print(f"Coefficients: {reg['coefficients']}")
# Also returns: predictions (M,), residuals (M,)
```

This is a lower-level tool than `sufficient_statistic_efficiency` — it returns the actual regression coefficients and residuals rather than just an R^2 summary.

## Community type discriminant

Classify compositions into structural types based on forced-pair position:

```python
result = fs.community_type_discriminant(cloud, calibrator="synthetic_v0")
print(result["labels"])   # array of labels per composition
print(result["scores"])   # raw scores
print(result["method"])   # "pair"
```

> **Warning:** Empirical heuristic. Classification boundaries are data-dependent and should not be treated as universal thresholds. The `"synthetic_v0"` calibrator uses presets tuned on synthetic generators.

## Utilities

Validation, projection, and normalization for simplex data:

```python
# Validate simplex data (configurable: "never", "warn", "always")
validated = fs.validate_simplex(data, renormalize="warn")

# Euclidean projection onto the simplex
projected = fs.project_to_simplex(np.array([0.5, -0.1, 0.8]))

# Proportional normalization (closure)
counts = np.array([10, 25, 5, 60])
composition = fs.closure(counts)

# Single-composition diagnostics (scalar version of batch_diagnostic)
diag = fs.full_diagnostic(composition)
```

## Frontier tools (experimental)

The first genuinely free symmetric-even enrichment appears at degree 8. The frontier module exposes degree-8 coordinates orthogonal to (Q_delta, H_3) under the Dirichlet(1) measure.

```python
from fisher_simplex.frontier import frontier8_coordinates, frontier8_residual

# Extended coordinates: [Q_delta, H_3, E8_1, E8_2]
coords = frontier8_coordinates(cloud)  # (M, 4)

# Test whether a target needs degree-8 enrichment
target = np.sum(cloud**4, axis=1)
result = frontier8_residual(cloud, target)
print(f"R^2 forced: {result['r_squared_forced']:.4f}")
print(f"R^2 frontier: {result['r_squared_frontier']:.4f}")
print(f"Needs frontier: {result['needs_frontier']}")
```
