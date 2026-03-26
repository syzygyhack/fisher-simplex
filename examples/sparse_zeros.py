"""Sparse compositional data with zeros (microbiome-style).

Demonstrates that the Fisher/Hellinger geometry handles zero-inflated
data naturally, without the singularities that affect log-ratio methods.
"""

import numpy as np

import fisher_simplex as fs

print("=== Sparse Compositional Data with Zeros ===\n")

# ── Create sparse microbiome-style data ──────────────────────────────────
# Low-alpha Dirichlet produces many near-zero entries; we threshold to
# create genuine zeros (simulating OTU tables with absent taxa).
rng = np.random.default_rng(42)
n_taxa = 20
n_samples = 60

raw = rng.dirichlet([0.1] * n_taxa, size=n_samples)

# Zero out entries below a detection threshold, then renormalize
threshold = 0.005
sparse = np.where(raw < threshold, 0.0, raw)
sparse = sparse / sparse.sum(axis=-1, keepdims=True)

zeros_per_row = (sparse == 0).sum(axis=-1)
print(f"Taxa: {n_taxa}, Samples: {n_samples}")
print(f"Zeros per sample: mean={zeros_per_row.mean():.1f}, "
      f"min={zeros_per_row.min()}, max={zeros_per_row.max()}")
print(f"Overall sparsity: {(sparse == 0).mean():.1%}\n")

# ── Fisher distances work with zeros ─────────────────────────────────────
print("--- Fisher Distances (boundary-safe) ---")
D = fs.pairwise_fisher_distances(sparse)
upper = D[np.triu_indices(n_samples, k=1)]
print(f"  Mean pairwise Fisher distance: {upper.mean():.4f}")
print(f"  Std:  {upper.std():.4f}")
print(f"  Range: [{upper.min():.4f}, {upper.max():.4f}]")
print(f"  No NaN or Inf: {not (np.any(np.isnan(D)) or np.any(np.isinf(D)))}")

# ── Fisher mean of sparse data ───────────────────────────────────────────
print("\n--- Fisher Mean ---")
mean = fs.fisher_mean(sparse)
print(f"  Sum: {mean.sum():.10f}")
print(f"  Non-zero components: {(mean > 0).sum()} / {n_taxa}")
print(f"  Min component: {mean.min():.6e}")
# The Fisher mean will generally have all positive entries even if some
# input rows have zeros, because the square-root lift smooths them.

# ── Geodesic between a sparse and a dense sample ─────────────────────────
print("\n--- Geodesic: Sparse to Dense ---")
sparse_sample = sparse[zeros_per_row.argmax()]  # most sparse
dense_sample = sparse[zeros_per_row.argmin()]    # least sparse
n_sparse_zeros = (sparse_sample == 0).sum()
n_dense_zeros = (dense_sample == 0).sum()
print(f"  Sparse sample zeros: {n_sparse_zeros}/{n_taxa}")
print(f"  Dense sample zeros:  {n_dense_zeros}/{n_taxa}")

ts = np.linspace(0, 1, 11)
path = fs.geodesic_interpolate(sparse_sample, dense_sample, ts)
# Verify all points are valid simplex compositions
for i, pt in enumerate(path):
    assert pt.sum() > 0.999 and np.all(pt >= -1e-15), f"Invalid at t={ts[i]}"
print("  All 11 geodesic points are valid simplex compositions.")

d = fs.fisher_distance(sparse_sample, dense_sample)
print(f"  Fisher distance (sparse -> dense): {d:.4f}")

# ── Tangent PCA on sparse data ───────────────────────────────────────────
print("\n--- Tangent PCA ---")
pca = fs.fisher_pca(sparse, n_components=5)
cumvar = np.cumsum(pca["explained_variance_ratio"])
print("  Cumulative explained variance (first 5 PCs):")
for k in range(5):
    print(f"    PC{k+1}: {cumvar[k]:.4f}")

# ── Overlap diagnostics on sparse data ───────────────────────────────────
print("\n--- Overlap Diagnostics ---")
diag = fs.batch_diagnostic(sparse)
print(f"  Phi mean:  {diag['phi'].mean():.4f}")
print(f"  Psi mean:  {diag['psi'].mean():.4f}  (low because zeros kill the product)")
print(f"  Divergence mean: {diag['divergence'].mean():.4f}")

# Many samples have at least one zero, so Psi = 0 for those
n_psi_zero = (diag["psi"] == 0).sum()
print(f"  Samples with Psi = 0: {n_psi_zero}/{n_samples}")
print("  (Psi = N^N * prod(s_i); any zero component -> Psi = 0)")

# ── Forced-pair coordinates capture shape beyond sparsity ────────────────
print("\n--- Forced-Pair Coordinates ---")
coords = fs.forced_coordinates(sparse)
print(f"  Q_delta: mean={coords[:, 0].mean():.4f}, std={coords[:, 0].std():.4f}")
print(f"  H_3:     mean={coords[:, 1].mean():.6f}, std={coords[:, 1].std():.6f}")

# Compare sparse vs dense subsets
median_zeros = np.median(zeros_per_row)
sparse_mask = zeros_per_row > median_zeros
dense_mask = ~sparse_mask
q_sparse = coords[sparse_mask, 0].mean()
q_dense = coords[dense_mask, 0].mean()
print(f"  Q_delta (high-zero samples): {q_sparse:.4f}")
print(f"  Q_delta (low-zero samples):  {q_dense:.4f}")
print("  Sparser samples are more concentrated (higher Q_delta)")

print("\n--- Sparse zeros example complete ---")
