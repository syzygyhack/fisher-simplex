"""Basic Fisher geometry on simplex data.

End-to-end example covering Fisher distance, Fisher mean, geodesic
interpolation, and tangent PCA.
"""

import numpy as np

import fisher_simplex as fs

# ── Create simplex data ──────────────────────────────────────────────────
rng = np.random.default_rng(42)
cloud = fs.dirichlet_community(n=5, m=50, alpha=1.0, rng=rng)

a = cloud[0]
b = cloud[1]
print(f"Composition a: {a}")
print(f"Composition b: {b}")

# ── Fisher distance ──────────────────────────────────────────────────────
d = fs.fisher_distance(a, b)
print(f"\nFisher distance(a, b): {d:.4f}")

d_h = fs.hellinger_distance(a, b)
print(f"Hellinger distance(a, b): {d_h:.4f}")

bc = fs.bhattacharyya_coefficient(a, b)
print(f"Bhattacharyya coefficient(a, b): {bc:.4f}")

# Pairwise distances
D = fs.pairwise_fisher_distances(cloud)
print(f"\nPairwise distance matrix shape: {D.shape}")
print(f"Mean pairwise Fisher distance: {D[np.triu_indices(50, k=1)].mean():.4f}")

# ── Fisher mean ──────────────────────────────────────────────────────────
mean = fs.fisher_mean(cloud)
print(f"\nFisher mean: {mean}")
print(f"Fisher mean sums to: {mean.sum():.10f}")
assert np.allclose(mean.sum(), 1.0), "Fisher mean should sum to 1"
assert np.all(mean >= 0), "Fisher mean should be nonneg"

# ── Geodesic interpolation ───────────────────────────────────────────────
ts = np.linspace(0, 1, 11)
path = fs.geodesic_interpolate(a, b, ts)
print(f"\nGeodesic path shape: {path.shape}")

# Verify endpoints
assert np.allclose(path[0], a, atol=1e-12), "Geodesic start should match a"
assert np.allclose(path[-1], b, atol=1e-12), "Geodesic end should match b"

# Verify all points are on the simplex
for i, pt in enumerate(path):
    assert np.allclose(pt.sum(), 1.0, atol=1e-12), f"Point {i} not on simplex"
    assert np.all(pt >= -1e-15), f"Point {i} has negative entry"

print("All geodesic points verified on simplex.")

# Monotone distance from a along the path
dists = [float(fs.fisher_distance(a, path[i])) for i in range(len(ts))]
print(f"Distances from a along geodesic: {[f'{d:.3f}' for d in dists]}")

# ── Forced coordinates ───────────────────────────────────────────────────
coords = fs.forced_coordinates(cloud)
print(f"\nForced coordinates shape: {coords.shape}")
print(f"Q_delta range: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]")
print(f"H_3 range:     [{coords[:, 1].min():.6f}, {coords[:, 1].max():.6f}]")

# ── Tangent PCA ──────────────────────────────────────────────────────────
pca = fs.fisher_pca(cloud, n_components=3)
print(f"\nTangent PCA base point: {pca['base']}")
print(f"Explained variance ratio: {pca['explained_variance_ratio']}")
print(f"Total explained: {pca['explained_variance_ratio'].sum():.4f}")
print(f"Scores shape: {pca['scores'].shape}")
print(f"Components shape: {pca['components'].shape}")

# ── Kernel matrix ────────────────────────────────────────────────────────
K = fs.kernel_matrix(cloud, kind="fisher")
print(f"\nKernel matrix shape: {K.shape}")
print(f"Kernel matrix is symmetric: {np.allclose(K, K.T)}")
eigvals = np.linalg.eigvalsh(K)
print(f"Min eigenvalue: {eigvals.min():.6f} (should be >= 0 for PSD)")

print("\n--- Basic geometry example complete ---")
