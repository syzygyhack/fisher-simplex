"""Experimental degree-8 enrichment demo.

Demonstrates the frontier module: degree-8 enrichment coordinates
orthogonal to the forced pair (Q_delta, H_3), and residual diagnostics
showing when a target observable needs frontier enrichment.
"""

import numpy as np

import fisher_simplex as fs
from fisher_simplex.frontier import (
    frontier8_batch,
    frontier8_coordinates,
    frontier8_residual,
)
from fisher_simplex.harmonic import (
    enrichment_space,
    selective_frontier,
    symmetric_even_dimension,
)

print("=== Experimental Degree-8 Enrichment Demo ===\n")

# ── Selective frontier structure ─────────────────────────────────────────
print("--- Selective Frontier Info ---")
for N in [3, 5, 10, 20]:
    info = selective_frontier(N)
    print(f"  N={N:2d}: forced dims={info['forced_dimension']}, "
          f"enrichment dims={info['enrichment_dimension']}, "
          f"total={info['total_dimension']}")

print()
print("Dimension of symmetric-even harmonic sector by degree:")
for N in [5, 10]:
    dims = []
    for deg in range(0, 10, 2):
        dims.append(symmetric_even_dimension(N, deg))
    print(f"  N={N}: degree 0,2,4,6,8 -> dims {dims}")

# ── Enrichment space basis ───────────────────────────────────────────────
print("\n--- Enrichment Space at Degree 8 ---")
basis = enrichment_space(5, 8)
for i, elem in enumerate(basis):
    print(f"  E8_{i+1}: {elem['description']}")

# ── Compute frontier coordinates on synthetic data ───────────────────────
print("\n--- Frontier Coordinates ---")
rng = np.random.default_rng(42)
N = 10
data = fs.dirichlet_community(n=N, m=200, alpha=1.0, rng=rng)

coords = frontier8_batch(data)
print(f"  Shape: {coords.shape}  (M samples x 4 coordinates)")
print("  Columns: [Q_delta, H_3, E8_1, E8_2]")
print(f"  Q_delta:  mean={coords[:, 0].mean():.6f}, std={coords[:, 0].std():.6f}")
print(f"  H_3:      mean={coords[:, 1].mean():.6f}, std={coords[:, 1].std():.6f}")
print(f"  E8_1:     mean={coords[:, 2].mean():.6f}, std={coords[:, 2].std():.6f}")
print(f"  E8_2:     mean={coords[:, 3].mean():.6f}, std={coords[:, 3].std():.6f}")

# Single composition
single = frontier8_coordinates(data[0])
print(f"\n  Single composition: shape={single.shape}, values={single}")

# ── Target that lives in the forced block: p4 = sum(s_i^4) ──────────────
print("\n--- Residual Diagnostics ---")
print("\nTarget: p4 = sum(s_i^4) [should be in forced block]")
p4 = np.sum(data**4, axis=-1)
result_p4 = frontier8_residual(data, p4)
print(f"  R^2 (forced only):  {result_p4['r_squared_forced']:.6f}")
print(f"  R^2 (with frontier): {result_p4['r_squared_frontier']:.6f}")
print(f"  Improvement:         {result_p4['frontier_improvement']:.6f}")
print(f"  Needs frontier:      {result_p4['needs_frontier']}")

# ── Target that needs degree-8: a product of power sums ──────────────────
print("\nTarget: p2^2 * p4 - p3^2 [designed to need degree-8]")
p2 = np.sum(data**2, axis=-1)
p3 = np.sum(data**3, axis=-1)
target_d8 = p2**2 * p4 - p3**2
result_d8 = frontier8_residual(data, target_d8)
print(f"  R^2 (forced only):  {result_d8['r_squared_forced']:.6f}")
print(f"  R^2 (with frontier): {result_d8['r_squared_frontier']:.6f}")
print(f"  Improvement:         {result_d8['frontier_improvement']:.6f}")
print(f"  Needs frontier:      {result_d8['needs_frontier']}")

# ── Quadratic model for finer resolution ─────────────────────────────────
print("\nTarget: p4 with quadratic model:")
result_p4q = frontier8_residual(data, p4, model="quadratic")
print(f"  R^2 (forced quad):   {result_p4q['r_squared_forced']:.6f}")
print(f"  R^2 (frontier quad): {result_p4q['r_squared_frontier']:.6f}")
print(f"  Improvement:         {result_p4q['frontier_improvement']:.6f}")

# ── Compare forced-pair sufficiency across N ─────────────────────────────
# NOTE: This regression test is a numerical sanity check, not a proof.
# p4 = sum(s_i^4) IS in the forced block as a theorem (it is a degree-8
# polynomial in s, but decomposes into Q_delta and H_3 terms quadratically).
# At high N with finite samples, the numerical R^2 may dip below the 0.999
# threshold — this reflects sampling noise, not a failure of the theorem.
# See spec section 6.3: "implementation sanity checks, not substitutes
# for the symbolic theorems."
print("\n--- Forced-Pair Sufficiency for p4 across N ---")
for N_test in [5, 10, 20, 30]:
    test_data = fs.dirichlet_community(n=N_test, m=500, alpha=1.0, rng=rng)
    test_p4 = np.sum(test_data**4, axis=-1)
    eff = fs.sufficient_statistic_efficiency(test_data, test_p4)
    print(f"  N={N_test:2d}: R^2_lin={eff['r_squared_linear']:.4f}, "
          f"R^2_quad={eff['r_squared_quadratic']:.6f}, "
          f"in_forced_block={eff['in_forced_block']}")

print("\n--- Frontier enrichment example complete ---")
