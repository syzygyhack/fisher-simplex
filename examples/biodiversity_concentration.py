"""Biodiversity and concentration comparison.

Demonstrates overlap diagnostics, forced-pair invariants, and the
divergence between quadratic (Phi) and multiplicative (Psi) overlap
summaries across ecological and economic archetypes.
"""

import numpy as np

import fisher_simplex as fs

# ── Generate ecological community archetypes ─────────────────────────────
rng = np.random.default_rng(42)
n_species = 10

even_communities = fs.dirichlet_community(n=n_species, m=100, alpha=5.0, rng=rng)
uneven_communities = fs.lognormal_community(n=n_species, m=100, rng=rng)
geometric_communities = fs.geometric_series(n=n_species, m=100, rng=rng)

print("=== Biodiversity / Concentration Comparison ===\n")

# ── One-number summaries ─────────────────────────────────────────────────
for name, data in [
    ("Even (Dirichlet alpha=5)", even_communities),
    ("Uneven (lognormal)", uneven_communities),
    ("Geometric series", geometric_communities),
]:
    hhi = fs.herfindahl(data)
    shannon = fs.shannon_entropy(data)
    eff_h = fs.effective_number_herfindahl(data)
    eff_s = fs.effective_number_shannon(data)

    print(f"{name}:")
    print(f"  HHI:               {hhi.mean():.4f} +/- {hhi.std():.4f}")
    print(f"  Shannon entropy:   {shannon.mean():.4f} +/- {shannon.std():.4f}")
    print(f"  Eff # (HHI):       {eff_h.mean():.2f} +/- {eff_h.std():.2f}")
    print(f"  Eff # (Shannon):   {eff_s.mean():.2f} +/- {eff_s.std():.2f}")
    print()

# ── Overlap divergence: when do Phi and Psi disagree? ────────────────────
print("--- Overlap Divergence Analysis ---\n")

for name, data in [
    ("Even", even_communities),
    ("Uneven", uneven_communities),
    ("Geometric", geometric_communities),
]:
    div = fs.divergence_analysis(data)
    print(f"{name}:")
    print(f"  Mean divergence (Phi - Psi): {div['mean_divergence']:.6f}")
    print(f"  Max |divergence|:            {div['max_divergence']:.6f}")
    print(f"  Fraction |D| > 0.01:         {div['fraction_consequential']:.0%}")
    tau = div["ranking_disagreement"]["concordance_tau"]
    print(f"  Ranking concordance (tau-b):  {tau:.4f}")
    print(f"  {div['recommendation']}")
    print()

# NOTE: High absolute divergence does NOT imply ranking disagreement.
# Kendall tau measures rank concordance, not value agreement. A community
# archetype (e.g. geometric series) can have large |Phi - Psi| while the
# two summaries still rank compositions identically (tau = 1.0).

# ── Binary case: Phi_2 = Psi_2 always ───────────────────────────────────
print("--- Binary Case (N=2): Exact Coincidence ---")
binary = fs.dirichlet_community(n=2, m=200, rng=rng)
phi_2 = fs.phi(binary)
psi_2 = fs.psi(binary)
max_diff = np.max(np.abs(phi_2 - psi_2))
print(f"  Max |Phi_2 - Psi_2| over 200 samples: {max_diff:.2e}")
print("  (Should be ~machine epsilon: Phi_2 = Psi_2 = 4*s1*s2 exactly)\n")

# ── Forced-pair coordinates: beyond HHI ──────────────────────────────────
print("--- Forced-Pair Shape Correction ---\n")

for name, data in [
    ("Even", even_communities),
    ("Uneven", uneven_communities),
    ("Geometric", geometric_communities),
]:
    coords = fs.forced_coordinates(data)
    q, h = coords[:, 0], coords[:, 1]
    print(f"{name}:")
    print(f"  Q_delta: {q.mean():.6f} +/- {q.std():.6f}")
    print(f"  H_3:     {h.mean():.6f} +/- {h.std():.6f}")
    print()

# ── Concentration profile reports (economist-facing) ─────────────────────
print("--- Concentration Profile (Geometric Communities) ---")
print(fs.concentration_profile_report(geometric_communities))
print()

# ── Comparative index report ─────────────────────────────────────────────
print("--- Comparative Index Report (Uneven Communities) ---")
print(fs.comparative_index_report(uneven_communities))

print("\n--- Biodiversity / concentration example complete ---")
