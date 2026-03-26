"""Distributional shift detection between two clouds of probability vectors.

Demonstrates measuring geometric shift between reference and test
prediction sets under Fisher distance — e.g., comparing classifier
outputs across training stages or different datasets.
"""

import numpy as np

import fisher_simplex as fs

# ── Generate two clouds of probability vectors ───────────────────────────
# Reference: classifier outputs from training data (5-class, moderate spread)
rng_ref = np.random.default_rng(0)
ref_cloud = fs.dirichlet_community(n=5, m=40, alpha=1.0, rng=rng_ref)

# Test: classifier outputs that are more concentrated (confident)
rng_test = np.random.default_rng(1)
test_cloud = fs.dirichlet_community(n=5, m=40, alpha=0.3, rng=rng_test)

print("Reference cloud shape:", ref_cloud.shape)
print("Test cloud shape:", test_cloud.shape)

# ── Measure distributional shift ─────────────────────────────────────────
shift = fs.distributional_shift(ref_cloud, test_cloud)

print("\n--- Distributional Shift Report ---")
print(f"Cloud distance (Fisher distance between means): {shift['cloud_distance']:.4f}")
print(f"Mean nearest-pair distance:                     {shift['mean_distance']:.4f}")
print(f"Reference cloud dispersion:                     {shift['ref_dispersion']:.4f}")
print(f"Test cloud dispersion:                          {shift['test_dispersion']:.4f}")

# ── Compare Fisher means ─────────────────────────────────────────────────
ref_mean = fs.fisher_mean(ref_cloud)
test_mean = fs.fisher_mean(test_cloud)

print(f"\nReference Fisher mean: {ref_mean}")
print(f"Test Fisher mean:      {test_mean}")

# ── Concentration diagnostics ────────────────────────────────────────────
print("\n--- Concentration Comparison ---")
ref_coords = fs.forced_coordinates(ref_cloud)
test_coords = fs.forced_coordinates(test_cloud)

print(f"Reference Q_delta: mean={ref_coords[:, 0].mean():.4f}, "
      f"std={ref_coords[:, 0].std():.4f}")
print(f"Test Q_delta:      mean={test_coords[:, 0].mean():.4f}, "
      f"std={test_coords[:, 0].std():.4f}")
print(f"Reference H_3:     mean={ref_coords[:, 1].mean():.6f}, "
      f"std={ref_coords[:, 1].std():.6f}")
print(f"Test H_3:          mean={test_coords[:, 1].mean():.6f}, "
      f"std={test_coords[:, 1].std():.6f}")

# ── Overlap divergence comparison ────────────────────────────────────────
print("\n--- Divergence Analysis ---")
ref_div = fs.divergence_analysis(ref_cloud)
test_div = fs.divergence_analysis(test_cloud)

print(f"Reference mean divergence: {ref_div['mean_divergence']:.6f}")
print(f"Test mean divergence:      {test_div['mean_divergence']:.6f}")
print(f"Reference ranking disagreement: "
      f"{ref_div['ranking_disagreement']['disagreement_rate']:.4f}")
print(f"Test ranking disagreement:      "
      f"{test_div['ranking_disagreement']['disagreement_rate']:.4f}")

# ── No-shift baseline ────────────────────────────────────────────────────
print("\n--- No-Shift Baseline ---")
rng_same = np.random.default_rng(2)
same_cloud_a = fs.dirichlet_community(n=5, m=40, alpha=1.0, rng=rng_same)
rng_same2 = np.random.default_rng(3)
same_cloud_b = fs.dirichlet_community(n=5, m=40, alpha=1.0, rng=rng_same2)

no_shift = fs.distributional_shift(same_cloud_a, same_cloud_b)
print(f"Cloud distance (same distribution): {no_shift['cloud_distance']:.4f}")
print(f"Mean nearest-pair distance:         {no_shift['mean_distance']:.4f}")

# Compare: shifted vs baseline
print(f"\nShift ratio (cloud distance): "
      f"{shift['cloud_distance'] / max(no_shift['cloud_distance'], 1e-10):.2f}x")

print("\n--- Distributional shift example complete ---")
