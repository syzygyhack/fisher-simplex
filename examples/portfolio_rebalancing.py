"""Portfolio allocation geometry.

Demonstrates advantages of Fisher geometry over conventional Euclidean
metrics for comparing, interpolating, and profiling portfolio allocations.
"""

import numpy as np

import fisher_simplex as fs

print("=== Portfolio Allocation Geometry ===\n")

# -- Create portfolio archetypes -----------------------------------------
rng = np.random.default_rng(42)
n_assets = 8

equal_weight = fs.portfolio_weights(n=n_assets, m=30, style="equal", rng=rng)
concentrated = fs.portfolio_weights(
    n=n_assets, m=30, style="concentrated", rng=rng
)
diversified = fs.portfolio_weights(
    n=n_assets, m=30, style="diversified", rng=rng
)

print(f"Assets: {n_assets}")
print("Samples per style: 30")

# -- Fisher vs Euclidean distance ----------------------------------------
print("\n--- Fisher vs Euclidean Distance ---\n")

# Pick representative allocations
a_eq = equal_weight[0]
a_conc = concentrated[0]
a_div = diversified[0]

pairs = [
    ("equal vs concentrated", a_eq, a_conc),
    ("equal vs diversified", a_eq, a_div),
    ("concentrated vs diversified", a_conc, a_div),
]

print(f"{'Pair':<32s}  {'Fisher':>8s}  {'Euclidean':>10s}  {'Ratio':>6s}")
print("-" * 70)
for label, p, q in pairs:
    d_fisher = fs.fisher_distance(p, q)
    d_euclid = float(np.linalg.norm(p - q))
    ratio = d_fisher / d_euclid
    print(f"{label:<32s}  {d_fisher:8.4f}  {d_euclid:10.4f}  {ratio:6.2f}")

print(
    "\nFisher distance amplifies differences near the simplex boundary\n"
    "(small allocations), where Euclidean distance is insensitive."
)

# -- Geodesic rebalancing path -------------------------------------------
print("\n--- Geodesic Rebalancing Path ---\n")
print("Interpolating from concentrated to equal-weight:\n")

ts = np.linspace(0, 1, 6)
path = fs.geodesic_interpolate(a_conc, a_eq, ts)

print(f"{'t':>5s}  {'sum':>7s}  {'min wt':>8s}  {'max wt':>8s}  {'HHI':>8s}")
print("-" * 42)
for i, t in enumerate(ts):
    pt = path[i]
    hhi = float(fs.herfindahl(pt))
    print(
        f"{t:5.2f}  {pt.sum():7.4f}  {pt.min():8.4f}  {pt.max():8.4f}  "
        f"{hhi:8.4f}"
    )

print(
    "\nEvery intermediate allocation is a valid portfolio (nonneg, sums to 1).\n"
    "No clamping or renormalization needed — the geodesic stays on the simplex."
)

# -- Forced-pair: shape beyond HHI --------------------------------------
print("\n--- Shape Beyond HHI ---\n")

# Construct two portfolios with similar HHI but different structure:
# "barbell" — two large + small fringe
# "tilt" — one dominant + gradual decline
barbell = np.array([0.40, 0.35, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03])
tilt = np.array([0.50, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.02])

hhi_barbell = float(fs.herfindahl(barbell))
hhi_tilt = float(fs.herfindahl(tilt))

q_barbell, h_barbell = fs.forced_pair(barbell)
q_tilt, h_tilt = fs.forced_pair(tilt)

print(f"{'Portfolio':<12s}  {'HHI':>8s}  {'Q_delta':>10s}  {'H_3':>12s}")
print("-" * 48)
print(
    f"{'Barbell':<12s}  {hhi_barbell:8.4f}  {q_barbell:10.6f}  "
    f"{h_barbell:12.8f}"
)
print(
    f"{'Tilt':<12s}  {hhi_tilt:8.4f}  {q_tilt:10.6f}  "
    f"{h_tilt:12.8f}"
)
print(
    f"\nHHI difference:     {abs(hhi_barbell - hhi_tilt):.4f}"
)
print(
    f"H_3 difference:     {abs(h_barbell - h_tilt):.8f}"
)
print(
    "\nH_3 distinguishes the barbell (two large blocks) from the tilt\n"
    "(single dominant position) even when HHI is similar."
)

# -- Fisher mean of advisor panel ----------------------------------------
print("\n--- Consensus Allocation from Advisor Panel ---\n")

# Simulate 5 advisor allocations
advisors = fs.portfolio_weights(n=n_assets, m=5, style="mixed", rng=rng)

consensus = fs.fisher_mean(advisors)
naive_avg = advisors.mean(axis=0)

d_consensus = np.mean(
    [fs.fisher_distance(consensus, adv) for adv in advisors]
)
d_naive = np.mean(
    [fs.fisher_distance(naive_avg, adv) for adv in advisors]
)

print(f"Fisher mean (consensus):  {consensus}")
print(f"Arithmetic mean (naive):  {naive_avg}")
print(f"Mean Fisher dist to advisors (consensus): {d_consensus:.4f}")
print(f"Mean Fisher dist to advisors (naive avg): {d_naive:.4f}")
print(
    "\nThe Fisher mean minimizes distance in the natural geometry of\n"
    "the simplex; the arithmetic mean minimizes Euclidean distance.\n"
    "For diversified interior portfolios the gap is modest; it widens\n"
    "when allocations are sparser or more boundary-adjacent (near-zero\n"
    "positions in some assets)."
)

# -- Concentration profile -----------------------------------------------
print("\n--- Concentration Profile ---\n")
panel = np.vstack([equal_weight, concentrated, diversified])
print(fs.concentration_profile_report(panel))

print("\n--- Portfolio rebalancing example complete ---")
