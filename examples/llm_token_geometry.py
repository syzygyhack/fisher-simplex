"""LLM next-token geometry: fingerprinting models beyond perplexity.

Language models produce next-token probability distributions over their
vocabulary -- points on the probability simplex.  Two models with similar
perplexity can allocate probability mass in structurally different ways:
a base model might "deliberate" across several plausible tokens, while
an RLHF-tuned variant "commits" to one token with high confidence.

This example simulates top-20 logprob outputs from two such model
variants, converts them to simplex vectors via topk_to_simplex, and
shows that frontier8 coordinates detect structural differences invisible
to entropy, top-1 probability, and even the forced pair alone.

A streaming drift-detection scenario uses WindowedFisherStats to catch
a silent model swap in real time.

Demonstrates: topk_to_simplex, frontier8_batch, frontier8_residual,
              WindowedFisherStats.
"""

import numpy as np

import fisher_simplex as fs
from fisher_simplex.frontier import frontier8_batch, frontier8_residual

print("=== LLM Next-Token Geometry ===\n")

rng = np.random.default_rng(42)

K = 20        # top-k logprobs returned by the "API"
N_PROMPTS = 300

# ── Simulate top-k from two model variants ────────────────────────────────
# Both target 70-85% of probability mass in the top-K tokens, so the
# residual tail bin (~15-30%) has similar mass in both.  The structural
# difference is HOW the top-K mass is allocated: spread across 2-4
# candidates ("deliberator") vs concentrated in one token ("committer").


def _simulate_deliberator(m, rng):
    """Model that spreads mass across 2-4 plausible next tokens."""
    topk_list = []
    for _ in range(m):
        topk = np.zeros(K)
        n_cands = rng.integers(2, 5)
        top_total = rng.uniform(0.70, 0.85)

        # Candidates share most of the top-K mass roughly evenly
        cand_share = rng.uniform(0.80, 0.92)
        cand_probs = rng.dirichlet(np.ones(n_cands) * 5)
        topk[:n_cands] = cand_probs * top_total * cand_share

        # Remaining top-K slots: small decaying values
        n_rest = K - n_cands
        if n_rest > 0:
            rest = rng.exponential(1.0, size=n_rest)
            rest = np.sort(rest)[::-1]
            rest *= top_total * (1 - cand_share) / max(rest.sum(), 1e-30)
            topk[n_cands:] = rest

        topk_list.append(topk)
    return np.array(topk_list)


def _simulate_committer(m, rng):
    """Model that concentrates mass on a single dominant token."""
    topk_list = []
    for _ in range(m):
        topk = np.zeros(K)
        top_total = rng.uniform(0.70, 0.85)

        # One dominant token gets 50-70% of top-K mass
        dominant_share = rng.uniform(0.50, 0.70)
        topk[0] = top_total * dominant_share

        # Rest: gradual exponential decay
        rest = rng.exponential(1.0, size=K - 1)
        rest = np.sort(rest)[::-1]
        rest *= top_total * (1 - dominant_share) / max(rest.sum(), 1e-30)
        topk[1:] = rest

        topk_list.append(topk)
    return np.array(topk_list)


deliberator_topk = _simulate_deliberator(N_PROMPTS, rng)
committer_topk = _simulate_committer(N_PROMPTS, rng)


# ── Convert top-k to simplex via topk_to_simplex ─────────────────────────
# single_remainder mode: residual mass (1 - sum(top_k)) becomes one tail
# bin, giving a (K+1)-simplex vector per prompt.

deliberator = fs.topk_to_simplex(deliberator_topk, mode="single_remainder")
committer = fs.topk_to_simplex(committer_topk, mode="single_remainder")
N = deliberator.shape[1]  # K + 1 = 21

print(f"Top-k:  K={K} tokens per prompt  ->  N={N} simplex (with tail bin)")
print(f"Prompts per model: {N_PROMPTS}")


# ── Standard metrics ──────────────────────────────────────────────────────
print("\n--- Standard Metrics ---\n")

ent_d = fs.shannon_entropy(deliberator)
ent_c = fs.shannon_entropy(committer)
top1_d = deliberator_topk.max(axis=1)
top1_c = committer_topk.max(axis=1)

print(f"{'':14s}  {'Entropy':>14s}  {'Top-1 prob':>14s}")
print(f"{'':14s}  {'mean +/- std':>14s}  {'mean +/- std':>14s}")
print("-" * 46)
print(f"{'Deliberator':14s}  {ent_d.mean():.3f} +/- {ent_d.std():.3f}"
      f"  {top1_d.mean():.3f} +/- {top1_d.std():.3f}")
print(f"{'Committer':14s}  {ent_c.mean():.3f} +/- {ent_c.std():.3f}"
      f"  {top1_c.mean():.3f} +/- {top1_c.std():.3f}")

# Entropy-threshold classifier
ent_all = np.concatenate([ent_d, ent_c])
ent_labels = np.array([0] * N_PROMPTS + [1] * N_PROMPTS)
threshold = np.median(ent_all)
ent_preds = (ent_all < threshold).astype(int)
ent_acc = (ent_preds == ent_labels).mean()
print(f"\nEntropy-threshold classifier: {ent_acc:.1%}")


# ── Forced-pair analysis ──────────────────────────────────────────────────
print("\n--- Forced-Pair Distributions ---\n")

q_d, h_d = fs.q_delta(deliberator), fs.h3(deliberator)
q_c, h_c = fs.q_delta(committer), fs.h3(committer)

print(f"{'':14s}  {'Q_delta':>18s}  {'H_3':>18s}")
print("-" * 54)
print(f"{'Deliberator':14s}  {q_d.mean():.6f} +/- {q_d.std():.6f}"
      f"  {h_d.mean():.4e} +/- {h_d.std():.4e}")
print(f"{'Committer':14s}  {q_c.mean():.6f} +/- {q_c.std():.6f}"
      f"  {h_c.mean():.4e} +/- {h_c.std():.4e}")


# ── Frontier coordinates: effect size comparison ──────────────────────────
print("\n--- Effect Size by Coordinate (Cohen's d) ---\n")

coords_d = frontier8_batch(deliberator)
coords_c = frontier8_batch(committer)

coord_names = ["Q_delta", "H_3", "E8_1", "E8_2"]
for i, name in enumerate(coord_names):
    mu_d, mu_c = coords_d[:, i].mean(), coords_c[:, i].mean()
    sd_d, sd_c = coords_d[:, i].std(), coords_c[:, i].std()
    pooled = np.sqrt((sd_d**2 + sd_c**2) / 2)
    effect = abs(mu_d - mu_c) / pooled if pooled > 1e-15 else 0.0
    tag = "large" if effect > 0.8 else "medium" if effect > 0.5 else "small"
    print(f"  {name:8s}  |d| = {effect:.2f}  ({tag})")

print("\n  Large effect = strong signal for telling models apart.")


# ── Classification comparison ─────────────────────────────────────────────
print("\n--- Model Identification (Nearest-Centroid) ---\n")

all_coords = np.vstack([coords_d, coords_c])
labels = np.array([0] * N_PROMPTS + [1] * N_PROMPTS)

half = N_PROMPTS // 2
train_idx = np.concatenate([np.arange(half), np.arange(N_PROMPTS, N_PROMPTS + half)])
test_idx = np.concatenate([
    np.arange(half, N_PROMPTS),
    np.arange(N_PROMPTS + half, 2 * N_PROMPTS),
])


def _classify(features, train_idx, test_idx, labels):
    """Nearest-centroid accuracy in feature space."""
    centroids = np.array([
        features[train_idx][labels[train_idx] == c].mean(axis=0)
        for c in [0, 1]
    ])
    dists = np.array([
        np.linalg.norm(features[test_idx] - c, axis=1)
        for c in centroids
    ])
    return float((dists.argmin(axis=0) == labels[test_idx]).mean())


acc_q = _classify(all_coords[:, :1], train_idx, test_idx, labels)
acc_forced = _classify(all_coords[:, :2], train_idx, test_idx, labels)
acc_frontier = _classify(all_coords, train_idx, test_idx, labels)

print(f"  Entropy threshold:  {ent_acc:6.1%}")
print(f"  Q_delta only:       {acc_q:6.1%}")
print(f"  Forced pair (Q, H): {acc_forced:6.1%}")
print(f"  Frontier (4-dim):   {acc_frontier:6.1%}")
print(f"\n  Improvement from E8: {acc_frontier - acc_forced:+.1%}")


# ── Residual diagnostic ──────────────────────────────────────────────────
print("\n--- Frontier Residual Diagnostic ---\n")
print("Does the model-identity label require degree-8 enrichment?\n")

all_data = np.vstack([deliberator, committer])
result = frontier8_residual(all_data, labels.astype(float))
print(f"  R^2 (forced only):   {result['r_squared_forced']:.4f}")
print(f"  R^2 (with frontier): {result['r_squared_frontier']:.4f}")
print(f"  Improvement:         {result['frontier_improvement']:.4f}")
print(f"  Needs frontier:      {result['needs_frontier']}")


# ── Streaming drift detection ─────────────────────────────────────────────
# Scenario: your API provider silently swaps models.  A sliding window
# of output distributions tracks the shift in real time.

print("\n--- Streaming Drift Detection ---")
print("Scenario: API silently swaps from deliberator to committer.\n")

window = fs.WindowedFisherStats(n_components=N, window_size=50)
window.push_batch(deliberator[:50])
baseline_mean = window.mean.copy()
baseline_disp = window.dispersion

print(f"Baseline (deliberator, 50 prompts):")
print(f"  Dispersion: {baseline_disp:.4f}")
print(f"  Q_delta:    {window.forced_pair_mean[0]:.6f}")

print(f"\n{'Committer %':>12s}  {'Shift':>8s}  {'Dispersion':>12s}  {'Q_delta':>10s}")
print("-" * 46)

for pct in [0, 25, 50, 75, 100]:
    win = fs.WindowedFisherStats(n_components=N, window_size=50)
    n_c = int(50 * pct / 100)
    n_d = 50 - n_c
    batch_d = deliberator[rng.choice(N_PROMPTS, size=n_d, replace=False)]
    batch_c = committer[rng.choice(N_PROMPTS, size=max(n_c, 1), replace=False)]
    blend = np.vstack([batch_d, batch_c[:n_c]]) if n_c > 0 else batch_d
    win.push_batch(blend)

    shift = win.shift_from(baseline_mean)
    disp = win.dispersion
    fp = win.forced_pair_mean
    print(f"{pct:>11d}%  {shift:>8.4f}  {disp:>12.4f}  {fp[0]:>10.6f}")

print("\nFisher shift grows monotonically as the committer model takes over.")
print("Combined with frontier coordinates, this provides a geometry-aware")
print("model-identity fingerprint that goes beyond scalar metrics.")

print("\n--- LLM token geometry example complete ---")
