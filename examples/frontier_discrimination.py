"""Frontier discrimination: when degree-8 coordinates reveal hidden structure.

Constructs two synthetic populations — "duopoly" and "monopoly" — with
overlapping forced-pair (Q_delta, H_3) distributions but distinguishable
degree-8 enrichment structure.  A nearest-centroid classifier using the
forced pair alone achieves moderate accuracy; adding the E8 enrichment
coordinates from frontier8_batch improves classification substantially.

This demonstrates the practical value of degree-8 enrichment: the forced
pair captures concentration and cubic skewness, but cannot fully distinguish
weight split across two leaders from weight concentrated in one.
"""

import numpy as np

import fisher_simplex as fs
from fisher_simplex.frontier import frontier8_batch, frontier8_residual

print("=== Frontier Discrimination: When Degree-8 Matters ===\n")

rng = np.random.default_rng(42)
N = 10
M = 300

# ── Population generators ─────────────────────────────────────────────────
# Tuned so Q_delta distributions overlap (similar expected p_2).
#
# Duopoly: two components share ~80% of weight roughly evenly.
#   p_2 ≈ 2*(0.40)^2 ≈ 0.32  →  Q_delta ≈ 0.22
#
# Monopoly: one component holds ~55% of weight.
#   p_2 ≈ (0.55)^2 + small ≈ 0.32  →  Q_delta ≈ 0.22
#
# But p_4 differs:  duopoly ≈ 2*(0.40)^4 = 0.051
#                    monopoly ≈ (0.55)^4 = 0.091
# This degree-8 signal is what E8 captures.


def _make_duopoly(N, m, rng):
    """Weight split roughly evenly between two dominant components."""
    out = []
    for _ in range(m):
        s = np.zeros(N)
        idx = rng.choice(N, size=2, replace=False)
        total = rng.uniform(0.72, 0.88)
        split = rng.beta(8, 8)  # centered near 0.5
        s[idx[0]] = total * split
        s[idx[1]] = total * (1 - split)
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        s[mask] = (1 - total) * rng.dirichlet(np.ones(N - 2))
        out.append(s)
    return np.array(out)


def _make_monopoly(N, m, rng):
    """Weight concentrated in a single dominant component."""
    out = []
    for _ in range(m):
        s = np.zeros(N)
        idx = rng.choice(N)
        s[idx] = rng.uniform(0.45, 0.65)
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        s[mask] = (1 - s[idx]) * rng.dirichlet(np.ones(N - 1))
        out.append(s)
    return np.array(out)


duopoly = _make_duopoly(N, M, rng)
monopoly = _make_monopoly(N, M, rng)

print(f"Duopoly:  {M} samples, N={N}  (weight in ~2 components)")
print(f"Monopoly: {M} samples, N={N}  (weight in ~1 component)")


# ── Forced-pair overlap ───────────────────────────────────────────────────
print("\n--- Forced-Pair Distributions ---\n")

q_duo, h_duo = fs.q_delta(duopoly), fs.h3(duopoly)
q_mono, h_mono = fs.q_delta(monopoly), fs.h3(monopoly)

print(f"{'':12s}  {'Q_delta':>18s}  {'H_3':>18s}")
print(f"{'':12s}  {'mean +/- std':>18s}  {'mean +/- std':>18s}")
print("-" * 52)
print(f"{'Duopoly':12s}  {q_duo.mean():.4f} +/- {q_duo.std():.4f}"
      f"  {h_duo.mean():.2e} +/- {h_duo.std():.2e}")
print(f"{'Monopoly':12s}  {q_mono.mean():.4f} +/- {q_mono.std():.4f}"
      f"  {h_mono.mean():.2e} +/- {h_mono.std():.2e}")


# ── Frontier coordinates: effect sizes ────────────────────────────────────
print("\n--- Effect Size by Coordinate (Cohen's d) ---\n")

coords_duo = frontier8_batch(duopoly)
coords_mono = frontier8_batch(monopoly)

coord_names = ["Q_delta", "H_3", "E8_1", "E8_2"]
for i, name in enumerate(coord_names):
    d_mean, d_std = coords_duo[:, i].mean(), coords_duo[:, i].std()
    m_mean, m_std = coords_mono[:, i].mean(), coords_mono[:, i].std()
    pooled_std = np.sqrt((d_std**2 + m_std**2) / 2)
    effect = abs(d_mean - m_mean) / pooled_std if pooled_std > 1e-15 else 0.0
    tag = "large" if effect > 0.8 else "medium" if effect > 0.5 else "small"
    print(f"  {name:8s}  |d| = {effect:.2f}  ({tag})")

print("\n  Cohen's d > 0.8 is conventionally 'large'.")
print("  E8 coordinates capture the duopoly-vs-monopoly shape difference")
print("  that Q_delta and H_3 only partially reflect.")


# ── Nearest-centroid classification ───────────────────────────────────────
print("\n--- Nearest-Centroid Classification ---\n")

all_coords = np.vstack([coords_duo, coords_mono])
labels = np.array([0] * M + [1] * M)

# Deterministic split: first half of each class trains, second tests
half = M // 2
train_idx = np.concatenate([np.arange(half), np.arange(M, M + half)])
test_idx = np.concatenate([np.arange(half, M), np.arange(M + half, 2 * M)])


def _classify(features, train_idx, test_idx, labels):
    """Nearest-centroid accuracy in feature space."""
    centroids = np.array([
        features[train_idx][labels[train_idx] == c].mean(axis=0)
        for c in [0, 1]
    ])
    dists = np.array([
        np.linalg.norm(features[test_idx] - c, axis=1)
        for c in centroids
    ])  # (2, n_test)
    preds = dists.argmin(axis=0)
    return float((preds == labels[test_idx]).mean())


acc_q = _classify(all_coords[:, :1], train_idx, test_idx, labels)
acc_forced = _classify(all_coords[:, :2], train_idx, test_idx, labels)
acc_frontier = _classify(all_coords, train_idx, test_idx, labels)

print(f"  Q_delta only:       {acc_q:6.1%}")
print(f"  Forced pair (Q, H): {acc_forced:6.1%}")
print(f"  Frontier (4-dim):   {acc_frontier:6.1%}")
print(f"\n  Improvement from E8: {acc_frontier - acc_forced:+.1%} accuracy")


# ── Confirm with frontier8_residual ───────────────────────────────────────
print("\n--- Frontier Residual Diagnostic ---\n")
print("Does the class label need degree-8 enrichment to predict?\n")

all_data = np.vstack([duopoly, monopoly])
result = frontier8_residual(all_data, labels.astype(float))
print(f"  R^2 (forced only):   {result['r_squared_forced']:.4f}")
print(f"  R^2 (with frontier): {result['r_squared_frontier']:.4f}")
print(f"  Improvement:         {result['frontier_improvement']:.4f}")
print(f"  Needs frontier:      {result['needs_frontier']}")

print("\n--- Frontier discrimination example complete ---")
