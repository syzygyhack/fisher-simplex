# Fisher-Rao Atlas of Transformer Attention: Reproduction Guide

**Goal:** Discover geometric charts in your model's attention heads, extract shared tangent modes, and validate them by replay — using `fisher-simplex` and standard PyTorch.

## Prerequisites

```bash
pip install fisher-simplex torch transformers
```

You need a HuggingFace causal LM with grouped query attention (Qwen3, Llama 3.2, Gemma 2, OLMo 2 all work).

**What comes from `fisher-simplex`:** The core module provides `fisher_lift` (amplitude lift ψ_i=√s_i), `fisher_distance`, `fisher_mean`, and canonical invariants (`q_delta`, `h3`). The `interp` module provides the full atlas pipeline: `overlap_matrix`, `mean_overlap_matrix`, `discover_charts`, `chart_stability`, `extract_shared_modes`, and `project_to_modes`. All functions are O(N), handle zeros without pseudocounts, and operate on numpy arrays.

**What you write yourself:** attention extraction (Step 1) and the replay hook (Step 4). Everything between extraction and replay is library calls.

## Step 1: Extract attention distributions

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = [...]  # 30-60 diverse prompts: factual, syntactic, code, narrative

n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
n_prompts = len(prompts)
late_layers = list(range(n_layers // 2, n_layers))  # late half
n_late = len(late_layers)

# Collect into a 3D array: (heads, prompts, vocab)
# We index heads as (layer, head) pairs flattened over late layers
head_ids = [(l, h) for l in late_layers for h in range(n_heads)]
n_total = len(head_ids)

# First pass: get sequence length and collect distributions
all_dists = []
for p_idx, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    prompt_dists = []
    for l, h in head_ids:
        # Final token's attention distribution over context
        dist = out.attentions[l][0, h, -1, :].float().cpu().numpy()
        prompt_dists.append(dist)
    all_dists.append(prompt_dists)

# Pad to common length across prompts (attention spans differ by prompt length)
max_len = max(all_dists[p][0].shape[0] for p in range(n_prompts))
X = np.zeros((n_total, n_prompts, max_len))
for p in range(n_prompts):
    for i in range(n_total):
        d = all_dists[p][i]
        X[i, p, :len(d)] = d

# X shape: (n_heads, n_prompts, seq_len) — ready for fisher_simplex.interp
```

## Step 2: Discover charts

```python
from fisher_simplex.interp import (
    mean_overlap_matrix, discover_charts, chart_stability
)

# Compute mean Bhattacharyya overlap across prompts
# Input: (entities, conditions, N) — here (heads, prompts, seq_len)
overlap = mean_overlap_matrix(X)
# overlap shape: (n_heads, n_heads)

# Discover charts via threshold clustering
tau = 0.7
charts = discover_charts(overlap, tau=tau)
chart_labels = charts["labels"]
# chart_labels: array of length n_heads, each entry is a chart ID

# Report what we found
print(f"Found {charts['n_charts']} charts at tau={tau}")
for c_id, members in charts["charts"].items():
    layers = [head_ids[i][0] for i in members]
    print(f"  Chart {c_id}: {len(members)} heads, layers {min(layers)}-{max(layers)}")

# Verify stability across prompts
stability = chart_stability(X, tau=tau, min_fraction=0.7)
# stability: dict with co_occurrence matrix and stable chart labels
print(f"Stable charts: {stability['n_stable_charts']}")
print(f"Co-occurrence (within first pair): "
      f"{stability['co_occurrence'][0, 1]:.2f}")
```

## Step 3: Extract shared modes and measure fit

```python
from fisher_simplex.interp import extract_shared_modes

# For each chart with enough members, extract shared modes
for c_id, members in charts["charts"].items():
    if len(members) < 3:
        continue

    mask = np.array(members)
    X_chart = X[mask]  # (n_chart_heads, n_prompts, seq_len)

    # extract_shared_modes does the full pipeline:
    #   1. Flatten to (M, N) and compute Fisher mean as base point
    #   2. Log map to tangent space via fisher_logmap
    #   3. Tangent PCA (top-K) for dimensionality reduction
    #   4. Reshape to (n_entities, n_conditions, K) in tangent subspace
    #   5. Remove per-entity mean (head effect)
    #   6. SVD of concatenated centered trajectories → shared modes
    #   7. Per-entity R² for the shared-mode fit

    print(f"\n  Chart {c_id} ({len(members)} heads):")

    # Progressive R² at different mode counts
    for m in [1, 2, 3]:
        result = extract_shared_modes(X_chart, n_modes=m)
        print(f"    Shared-{m} R²: mean={result['mean_r2']:.3f}, "
              f"min={result['per_entity_r2'].min():.3f}")

    # Full result at target mode count
    result = extract_shared_modes(X_chart, n_modes=3)
    print(f"    Tangent dim @90%: {result['dim_90']}")
    print(f"    Tangent dim @95%: {result['dim_95']}")

    # Target: shared-3 mean R² > 0.70
    # Target: tangent dim @90% between 3 and 7
```

## Step 4: Replay validation

This is the causal test. Replace attention distributions with their shared-mode projections and measure the effect on model output. This is the one step you write yourself — it requires PyTorch hooks into the model's forward pass.

```python
from torch.nn.functional import kl_div, log_softmax
from fisher_simplex.interp import project_to_modes

def replay_attention(model, tokenizer, text, chart_head_ids,
                     mode_result, n_modes=3):
    """
    Hook-based replay: replace attention in chart heads with
    shared-mode projection, measure KL to clean output.
    """
    inputs = tokenizer(text, return_tensors="pt")

    # Clean forward pass
    with torch.no_grad():
        clean_out = model(**inputs, output_attentions=True)
    clean_logits = log_softmax(clean_out.logits[0, -1], dim=-1)

    def make_hook(head_idx_in_chart, layer, head):
        def hook_fn(module, input, output):
            attn = output[0]
            dist = attn[0, head, -1, :].float().cpu().numpy()

            # project_to_modes handles the full round-trip:
            #   amplitude lift → log map → PCA projection →
            #   mode projection → exp map → back to simplex
            p_new = project_to_modes(
                dist,
                centroid=mode_result["centroid"],
                tangent_basis=mode_result["tangent_basis"],
                shared_modes=mode_result["shared_modes"],
                tangent_mean=mode_result["tangent_mean"],
                head_effect=mode_result["head_effects"][head_idx_in_chart],
                n_modes=n_modes,
            )

            new_attn = attn.clone()
            new_attn[0, head, -1, :len(p_new)] = torch.tensor(
                p_new, dtype=attn.dtype)
            return (new_attn,) + output[1:]
        return hook_fn

    # Register hooks
    hooks = []
    for idx, (l, h) in enumerate(chart_head_ids):
        layer_module = model.model.layers[l].self_attn
        hook = layer_module.register_forward_hook(make_hook(idx, l, h))
        hooks.append(hook)

    with torch.no_grad():
        replay_out = model(**inputs, output_attentions=True)
    replay_logits = log_softmax(replay_out.logits[0, -1], dim=-1)

    for hook in hooks:
        hook.remove()

    kl = kl_div(replay_logits, clean_logits.exp(), reduction='sum',
                log_target=False).item()
    return kl

# Run on eval corpus and compare conditions:
#   center replay (n_modes=0, project to centroid) — expect KL ~ 0.03-0.10
#   shared-1 replay                                — better than center
#   shared-2 replay                                — better still
#   shared-3 replay                                — expect KL ~ 0.03-0.04
#   full tangent replay                            — expect KL ~ 0.002-0.007
#   random-3 replay                                — worse KL than shared-3
```

## What to look for

If the framework is working on your model, you should see:

1. **Charts emerge** — late-layer heads cluster into 5-15 stable groups spanning multiple layers and KV groups.

2. **Low tangent dimension** — 90% of within-chart variance captured in 3-5 dimensions (early/mid positions) or 5-7 dimensions (late/final).

3. **Shared-3 R² > 0.70** — three shared modes capture most of the per-head prompt trajectory.

4. **Replay hierarchy** — KL improves monotonically: center > shared-1 > shared-2 > shared-3 > full tangent.

5. **Shared beats random** — shared-3 replay has lower KL than random-3, especially visible in the KL metric even when perplexity looks similar.

## Validated models

Results have been confirmed on the Qwen3 family (0.6B, 1.7B, 4B, 8B). Cross-architecture testing on Llama 3.2, Gemma 2, and OLMo 2 is in progress. Any dense GQA transformer with global attention should work.

## Key references

- Fisher-simplex library: `pip install fisher-simplex` ([GitHub](https://github.com/syzygyhack/fisher-simplex))
- The amplitude lift and Fisher-Rao metric: Campbell (1986), Čencov (1982)
- Čencov's theorem: the Fisher-Rao metric is the unique Riemannian metric on probability distributions invariant under sufficient statistics

## One-line summary

Attention heads are points on the Fisher-Rao sphere via ψ(s)=√s; cluster them by Bhattacharyya overlap and you find stable geometric charts whose prompt-induced variation lives in a 3-mode tangent subspace — replay just those 3 shared directions at all positions and Qwen3-1.7B+ loses <1% PPL, proving the modes are the operationally real coordinates and individual head identity is presentation.
