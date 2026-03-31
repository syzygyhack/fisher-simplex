# Quick start

Five steps from zero to Fisher-simplex analysis.

## Step 1: Load or create simplex data

Simplex data are nonnegative vectors summing to 1. They arise naturally as probability distributions, species abundances, market shares, and portfolio weights.

```python
import numpy as np
import fisher_simplex as fs

# Create simplex data directly
a = np.array([0.5, 0.3, 0.2])
b = np.array([0.1, 0.6, 0.3])

# Or generate synthetic data
cloud = fs.dirichlet_community(n=5, m=100, alpha=1.0, rng=np.random.default_rng(42))

# Or normalize raw counts to the simplex
raw_counts = np.array([10, 25, 5, 60])
composition = fs.closure(raw_counts)
```

## Step 2: Compute Fisher distance

The Fisher distance is the geodesic distance induced by the Fisher information metric on the simplex. It handles zeros naturally.

```python
d = fs.fisher_distance(a, b)
print(f"Fisher distance: {d:.4f}")

# Also available: Hellinger distance and Bhattacharyya coefficient
d_h = fs.hellinger_distance(a, b)
bc = fs.bhattacharyya_coefficient(a, b)

# Pairwise distance matrix for a cloud
D = fs.pairwise_fisher_distances(cloud)
print(f"Distance matrix shape: {D.shape}")  # (100, 100)
```

## Step 3: Compute Fisher mean

The Fisher mean is the extrinsic (projected arithmetic) mean in Fisher-lift amplitude coordinates. It approximates but is not identical to the intrinsic Frechet mean.

```python
mean = fs.fisher_mean(cloud)
print(f"Fisher mean: {mean}")
print(f"Sum: {mean.sum():.6f}")  # 1.0 — always on the simplex
```

## Step 4: Compute forced coordinates

The forced pair (Q_delta, H_3) captures the canonical low-degree symmetric-even structure in the Fisher-lifted sector.

```python
coords = fs.forced_coordinates(cloud)
print(f"Shape: {coords.shape}")  # (100, 2)

# Q_delta: excess concentration beyond uniform
# H_3: skewness in the share distribution
q_vals = coords[:, 0]
h_vals = coords[:, 1]

print(f"Q_delta range: [{q_vals.min():.4f}, {q_vals.max():.4f}]")
print(f"H_3 range:     [{h_vals.min():.6f}, {h_vals.max():.6f}]")
```

## Step 5: Run divergence analysis

Divergence analysis tells you when quadratic (Phi) and multiplicative (Psi) overlap summaries disagree.

```python
report = fs.divergence_analysis(cloud)
print(f"Mean divergence: {report['mean_divergence']:.6f}")
print(f"Max |divergence|: {report['max_divergence']:.6f}")
print(f"Fraction consequential: {report['fraction_consequential']:.2%}")
print(f"Ranking disagreement: {report['ranking_disagreement']['disagreement_rate']:.4f}")
print(report['recommendation'])
```

---

For deeper workflows — tangent PCA, kernel methods, distributional shift, degree-8 frontier tools, and mechanistic interpretability (chart discovery, shared modes) — see the [user guide](user_guide.md). For mathematical background, see the [mathematical notes](mathematical_notes.md).
