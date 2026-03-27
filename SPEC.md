# fisher-simplex — Specification v0.3

**Canonical simplex geometry via the Fisher amplitude lift, with low-complexity invariant tools**

---

## 0. Purpose

`fisher-simplex` is a scientific computing library for data on the probability simplex:

    Δ^{N−1} = { s ∈ ℝ^N_{≥0} : Σ s_i = 1 }.

It provides:

1. **Exact Fisher/Hellinger geometry** on the simplex via the square-root amplitude lift.
2. **Canonical overlap diagnostics** based on two exact overlap families.
3. **Low-complexity symmetric invariant tools** in the Fisher-lifted symmetric-even sector, centered on the pair (Q_Δ, H_3).
4. **Boundary-safe geometric workflows** for distances, means, interpolation, tangent analysis, and kernel methods.
5. **Experimental frontier tools** for the first genuinely free symmetric-even enrichment beyond the forced low-degree block.

The package is designed for researchers and engineers working with compositional data in ecology, microbiome science, economics, portfolio analysis, machine learning probability outputs, information geometry, and related areas.

This specification is implementation-facing. It distinguishes carefully between:

* **exact mathematics**,
* **engineering utilities built directly from exact mathematics**,
* **empirical heuristics validated only on synthetic benchmarks**, and
* **experimental frontier features**.

---

## 1. Mathematical basis

### 1.1 The Fisher amplitude lift

The canonical map

    s_i = ψ_i²,    ψ_i ≥ 0,    Σ ψ_i² = 1

sends the probability simplex Δ^{N−1} to the positive orthant of the unit sphere S^{N−1}.

Under this map, the Fisher information metric on the simplex pulls back to 4 times the round metric on the sphere. Equivalently, square-root-transformed simplex data carries a canonical spherical geometry.

This square-root/Hellinger embedding is known in statistics and information geometry. The library's novelty is not the existence of the lift itself, but its use as the central organizing structure for:

* exact geometric tools,
* overlap-family diagnostics,
* harmonic complexity grading,
* and low-degree forced-compression invariants.

### 1.2 Canonical overlap families

Two canonical scalar overlap families on Δ^{N−1} are:

    Φ_N(s) = (N/(N−1))(1 − Σ s_i²)
    Ψ_N(s) = N^N Π s_i

Interpretation:

* Φ_N is a normalized quadratic overlap / Gini-Simpson type statistic.
* Ψ_N is a normalized multiplicative overlap / entropy-adjacent statistic.

Exact facts:

* For N = 2: Φ_2(s) = Ψ_2(s) = 4s₁s₂.
* For N ≥ 3: Φ_N and Ψ_N are generically distinct.

This gives a complete structural answer to the specific question:

> When must quadratic and multiplicative overlap summaries diverge?

It does **not** by itself settle broader normative questions like which diversity or concentration index is "correct" in a given domain.

### 1.3 Low-degree forced compression in the Fisher-lifted symmetric-even sector

The Fisher lift induces a canonical grading of simplex observables by spherical harmonic degree on S^{N−1}. Within the **S_N-symmetric even sector**, low-degree nonconstant structure is forced through two invariants:

    Q_Δ = Σ s_i² − 1/N

    H_3 = Σ s_i³ − (3/(N+2)) Σ s_i² + 2/(N(N+2))

Engineering-facing theorem statement:

> Through degree 6 in the Fisher-lifted symmetric-even sector, the algebra of nonconstant S_N-symmetric observables factors through the pair (Q_Δ, H_3).

This pair is therefore the canonical low-complexity two-number summary **for observables in that sector**.

Important scope qualifier:

* This does **not** mean every practical low-order statistic on simplex data factors through (Q_Δ, H_3).
* It applies to the Fisher-lifted symmetric-even sector specifically.
* Applied claims about arbitrary target observables must be tested empirically using the `sufficient_statistic_efficiency` workflow, not asserted from the theorem alone.

### 1.4 H_3 normalization convention

The implementation uses the formula above, which does **not** vanish at the uniform composition. At s = (1/N, …, 1/N):

    H_3(uniform) = 2 / (N²(N+2))

This is a small positive value, not zero. The formula is the natural degree-6 harmonic in the Fisher-lifted sector; artificially centering it to vanish at the barycenter would break the harmonic interpretation. Documentation and tests must reflect this exact value.

### 1.5 First selective frontier

The first genuinely free symmetric-even enrichment appears at degree 8. Up to degree 6, the nonconstant sector is forced. At degree 8, a 2-dimensional enrichment space appears.

The 2-dimensional enrichment *subspace* is canonical (determined by the representation theory). Any specific basis for it depends on the orthogonalization convention. The library's default E8_1 and E8_2 are orthogonal to (Q_Δ, H_3) under the Dirichlet(1) (uniform) measure on the simplex and documented with explicit polynomial expressions.

This motivates the package's **experimental frontier tools**, which expose degree-8 enrichment coordinates in addition to the forced pair.

---

## 2. Design principles

1. **Exact geometry first.**
   The center of gravity of the library is the exact Fisher/Hellinger geometry of the simplex.

2. **Explicit status discipline.**
   Every feature is documented as one of:
   * exact,
   * exact-derived utility,
   * empirical heuristic,
   * experimental.

3. **Boundary-safe by construction.**
   Zeros are first-class citizens. The square-root lift handles them without singularity.

4. **No silent repair of bad data.**
   Input normalization is explicit and configurable.

5. **Lightweight scientific stack.**
   Core functionality depends only on NumPy and SciPy.

6. **Simple defaults, advanced layers optional.**
   Users should be able to do distances, means, tangent PCA, overlap diagnostics, and forced-pair summaries without learning the harmonic theory.

---

## 3. Package architecture

### 3.1 Python package: `fisher_simplex`

Modules:

* `core` — scalar invariants, overlap families, forced pair, lift, bridge statistics
* `geometry` — distances, geodesics, means, tangent maps, PCA, kernels
* `analysis` — batch diagnostics, divergence reports, forced-block analysis, concentration profiling
* `generators` — synthetic reference ensembles
* `frontier` — degree-8 enrichment coordinates and residual diagnostics (experimental)
* `harmonic` — low-degree symmetric-even harmonic tools (experimental)
* `viz` — matplotlib plotting utilities (optional, requires matplotlib)
* `utils` — validation, projection, perturbation utilities

### 3.2 Future packages (not part of v0.3 target)

* `fisher-simplex-js` — JavaScript/TypeScript for browser-native dashboards.
* `fisher-simplex-r` — R implementation for ecology and bioinformatics.

---

## 4. Module specifications

### 4.1 Module: `core`

Low-level exact scalar simplex invariants and transforms.

#### 4.1.1 Input conventions

All public functions accept:

* a single composition of shape `(N,)`, or
* a batch of compositions of shape `(M, N)`.

Validation policy is explicit:

```python
validate_simplex(x, *, axis=-1, tol=1e-12, renormalize="warn")
```

Allowed `renormalize` modes:

* `"never"` — reject if sums differ from 1 beyond tolerance.
* `"warn"` — renormalize mild sum drift and emit warning via `warnings.warn`.
* `"always"` — renormalize nonnegative vectors silently.

Negative entry handling:

* Entries in `[-tol, 0)` are clipped to zero when renormalization permits.
* Entries below `-tol` always raise `ValueError`.

#### 4.1.2 Overlap family functions

Status: **exact**.

```python
phi(s) -> ndarray             # shape (M,)
psi(s) -> ndarray             # shape (M,)
overlap_divergence(s) -> ndarray   # shape (M,)
```

Definitions:

    Φ_N(s) = (N/(N−1))(1 − Σ s_i²)
    Ψ_N(s) = N^N Π s_i
    D(s) = Φ_N(s) − Ψ_N(s)

Notes:

* Code names are lowercase ASCII; documentation uses Φ_N, Ψ_N.
* `psi` uses log-sum-exp internally for numerical stability.

#### 4.1.3 Forced-pair invariants

Status: **exact** (the functions themselves); **sector-qualified** (the theorem about their sufficiency).

```python
q_delta(s) -> ndarray          # shape (M,)
h3(s) -> ndarray               # shape (M,)
forced_pair(s) -> tuple[ndarray, ndarray]
forced_coordinates(s) -> ndarray   # shape (M, 2)
```

Definitions:

    Q_Δ = Σ s_i² − 1/N
    H_3 = Σ s_i³ − (3/(N+2)) Σ s_i² + 2/(N(N+2))

`forced_coordinates(s)` returns column-stacked `[Q_Δ, H_3]` with shape `(2,)` for single input or `(M, 2)` for batch.

#### 4.1.4 Optional heuristic diagnostics

Status: **empirical heuristic**, not canonical.

```python
qh_ratio(s, *, eps=1e-12, on_zero="nan") -> ndarray
```

Requirements:

* Docstring warns that the ratio is unstable near H_3 = 0.
* `on_zero` controls behavior when |H_3| < eps: `"nan"` (default), `"inf"`, or `"clip"`.
* Not presented as a flagship invariant in any documentation or example.

#### 4.1.5 Fisher amplitude lift

Status: **exact**.

```python
fisher_lift(s) -> ndarray      # shape (M, N), points on S^{N−1}
fisher_project(psi) -> ndarray # shape (M, N), points on Δ^{N−1}
```

Definitions:

    ψ_i = √(s_i)       (lift)
    s_i = ψ_i²          (project)

Requirements:

* `fisher_lift` maps simplex data to the nonnegative orthant of S^{N−1}.
* `fisher_project` maps nonnegative unit-norm vectors back to the simplex.
* Roundtrip accuracy at machine precision (atol 1e-14 in tests).

#### 4.1.6 Standard bridge statistics

Status: **exact-derived utility**, no novelty claims.

```python
herfindahl(s) -> ndarray
simpson_index(s) -> ndarray
gini_simpson(s) -> ndarray             # alias for simpson_index
shannon_entropy(s, *, base=None) -> ndarray
effective_number_herfindahl(s) -> ndarray   # 1 / Σ s_i²
effective_number_shannon(s, *, base=None) -> ndarray  # exp(H)
```

Purpose: interoperability with existing literature and baseline comparisons.

#### 4.1.7 Binary special-case helpers

Status: **exact**, pedagogical/testing. Not headlined.

```python
binary_overlap(x) -> ndarray       # 4x(1−x)
binary_fisher_angle(x) -> ndarray  # 2 arccos(√x)
```

---

### 4.2 Module: `geometry`

**This is a first-class module and the primary practical entry point for most users.**

#### 4.2.1 Distances and coefficients

Status: **exact**.

```python
bhattacharyya_coefficient(a, b) -> ndarray    # shape (M,) or scalar
hellinger_distance(a, b) -> ndarray           # shape (M,) or scalar
fisher_distance(a, b) -> ndarray              # shape (M,) or scalar
pairwise_fisher_distances(X) -> ndarray       # shape (M, M)
pairwise_hellinger_distances(X) -> ndarray    # shape (M, M)
```

Definitions:

    BC(a, b) = Σ_i √(a_i b_i)
    d_H(a, b) = (1/√2) ‖√a − √b‖_2
    d_F(a, b) = 2 arccos(Σ_i √(a_i b_i))

The Fisher distance is the canonical geodesic distance induced by the Fisher information metric. Hellinger distance and Bhattacharyya coefficient are exposed as established names in statistics.

Convention note: the Hellinger distance uses the normalized convention where d_H ∈ [0, 1] and d_H² = 1 − BC. The unnormalized convention (d_H² = 2(1 − BC)) is also common in the literature but is not used here. The Fisher-Hellinger relation is d_F = 2 arccos(1 − d_H²).

Pairwise functions return shape `(M, M)` symmetric distance/similarity matrices.

Numerical note: `fisher_distance` clips the argument of `arccos` to `[-1, 1]` to handle floating-point drift.

#### 4.2.2 Kernels

Status: **exact-derived utility**.

```python
fisher_kernel(a, b) -> ndarray                    # BC(a, b) = Σ √(a_i b_i)
fisher_cosine(a, b) -> ndarray                    # same value, spherical-language alias
polynomial_fisher_kernel(a, b, d) -> ndarray      # BC(a, b)^d for integer d >= 1
fisher_rbf_kernel(a, b, sigma) -> ndarray         # exp(-d_F^2 / (2*sigma^2))
kernel_matrix(X, *, kind="fisher") -> ndarray     # shape (M, M)
```

The Fisher kernel is the Bhattacharyya coefficient, which is also the cosine similarity in amplitude coordinates. `kernel_matrix` returns shape `(M, M)`.

Available `kind` values:

* `"fisher"` (default) — linear Fisher kernel B(p, q).
* `"polynomial_fisher"` — B(p, q)^d. Requires `d` kwarg (positive integer).
* `"fisher_rbf"` — exp(−d_F²/(2σ²)). Requires `sigma` kwarg.
* `"hellinger_rbf"` — exp(−d_H²/σ²). Requires `sigma` kwarg.

#### 4.2.3 Means and barycenters

Status: **exact-derived utility**.

```python
fisher_mean(X, *, weights=None) -> ndarray      # shape (N,)
fisher_barycenter(X, *, weights=None) -> ndarray # alias
```

Default implementation:

1. Compute weighted arithmetic mean in amplitude coordinates: ψ̄ = Σ w_i ψ_i.
2. Normalize to unit sphere: ψ̄ / ‖ψ̄‖.
3. Project back to simplex: s_i = ψ̄_i².

Documentation must state explicitly:

> This is the extrinsic (projected arithmetic) mean in Fisher-lift amplitude coordinates. It approximates but is not identical to the intrinsic Fréchet mean under the Fisher geodesic distance. For data concentrated in a small region of the simplex, the two coincide to first order. An iterative intrinsic Fréchet mean solver may be added in a future version.

#### 4.2.4 Geodesics and interpolation

Status: **exact**.

```python
fisher_geodesic(a, b, t) -> ndarray              # single point at parameter t
geodesic_interpolate(a, b, ts) -> ndarray         # shape (len(ts), N)
```

Implementation: spherical linear interpolation (slerp) in amplitude coordinates, projected back to simplex.

Requirements:

* Endpoints exact within tolerance: `fisher_geodesic(a, b, 0) == a`, `fisher_geodesic(a, b, 1) == b`.
* Geodesic remains on simplex for all t ∈ [0, 1] (amplitudes are nonneg throughout since both endpoints are in the positive orthant and slerp preserves this).
* Degenerate case: when a == b, return a for all t.

#### 4.2.5 Tangent-space tools

Status: **exact-derived utility**.

```python
fisher_logmap(X, *, base=None) -> ndarray     # shape (M, N) tangent vectors
fisher_expmap(V, *, base) -> ndarray           # shape (M, N) simplex points
tangent_map(X, *, base=None) -> tuple[ndarray, ndarray]  # (tangent_vecs, base)
```

The log map is the inverse exponential map on S^{N−1} at the lifted base point, restricted to the positive orthant. The exp map is its inverse.

Default behavior: if `base is None`, compute `fisher_mean(X)` and use that.

Boundary note: when the base point lies on the simplex boundary (has zero components), the effective tangent space dimension drops. `fisher_logmap` remains well-defined (the tangent vector has zeros in the corresponding components) but the tangent space is degenerate in those directions. This is documented in the docstring; the function does not raise an error but emits a warning when the base point has zeros.

#### 4.2.6 Tangent PCA

Status: **exact-derived utility**.

```python
fisher_pca(X, *, base=None, n_components=None, center=True) -> dict
```

Returned dict includes:

* `"base"` — base point on the simplex, shape `(N,)`.
* `"tangent_coords"` — data projected to tangent space, shape `(M, N)`.
* `"components"` — principal directions in tangent space, shape `(K, N)`.
* `"singular_values"` — shape `(K,)`.
* `"explained_variance"` — shape `(K,)`.
* `"explained_variance_ratio"` — shape `(K,)`.
* `"scores"` — projected coordinates, shape `(M, K)`.

Documentation must state:

> This is PCA in the tangent space of the Fisher-lifted spherical geometry at the base point, not ordinary Euclidean PCA on raw simplex coordinates.

#### 4.2.7 Utility geometry methods

Status: **engineering utility**.

```python
project_to_simplex(x) -> ndarray                        # Euclidean projection
closure(x) -> ndarray                                   # proportional normalization
sample_near(s, *, scale, geometry="fisher", rng=None) -> ndarray
perturb_simplex(s, *, eps, mode="fisher", rng=None) -> ndarray
```

These are convenience functions, not theorem-level exports.

---

### 4.3 Module: `analysis`

High-level workflows combining exact tools into interpretable reports.

#### 4.3.1 Batch diagnostics

```python
batch_diagnostic(X) -> dict
full_diagnostic(s) -> dict
```

`batch_diagnostic` returns a dict with arrays of shape `(M,)`:

* `"n_components"`, `"n_compositions"` — integers.
* `"phi"`, `"psi"`, `"divergence"` — overlap family values.
* `"q_delta"`, `"h3"` — forced pair.
* `"herfindahl"`, `"simpson"`, `"shannon"` — bridge statistics.
* `"fisher_coords"` — shape `(M, N)`, lifted coordinates.

`full_diagnostic(s)` returns the same fields as scalars for a single composition, plus `"fisher_coords"` as shape `(N,)`.

Heuristic fields (e.g. `qh_ratio`) are included **only** when explicitly requested via `include_heuristics=True`.

#### 4.3.2 Overlap divergence analysis

```python
divergence_analysis(X, *, threshold=0.10) -> dict
pairwise_ranking_disagreement(a, b) -> dict
```

`divergence_analysis` returns:

* `"n_components"` — integer.
* `"mean_divergence"`, `"max_divergence"`, `"divergence_std"` — floats.
* `"fraction_consequential"` — float, fraction where |D| > 0.01.
* `"ranking_disagreement"` — dict from `pairwise_ranking_disagreement(phi_vals, psi_vals)`.
* `"recommendation"` — string, **labeled as heuristic**.

`pairwise_ranking_disagreement` returns:

* `"disagreement_rate"` — float.
* `"concordance_tau"` — float, Kendall's τ-b.
* `"concordance_p"` — float, p-value.
* `"n_pairs"` — integer.
* `"n_disagreements"` — integer.

The threshold parameter and recommendation text must be documented as **empirical heuristics**, not theorems.

#### 4.3.3 Forced-block analysis

```python
sufficient_statistic_efficiency(X, target, *, model="linear") -> dict
forced_block_regression(X, target, *, basis="pair") -> dict
```

`sufficient_statistic_efficiency` returns:

* `"r_squared_linear"` — R² of linear fit through (Q, H).
* `"r_squared_quadratic"` — R² of quadratic fit through (Q, H, Q², QH, H²).
* `"in_forced_block"` — boolean, True if R²_quad > 0.999.
* `"residual_std"` — float.

Purpose: test whether a user-supplied target observable is well-explained by the forced pair. This is an empirical check, not a theorem-level factorization claim for arbitrary targets.

#### 4.3.4 Structure and concentration reports

```python
comparative_index_report(X) -> str
concentration_profile_report(X) -> str
```

These summarize where one-number summaries agree, where they disagree, and what additional structure (Q_Δ, H_3) captures. Designed for economist-facing output where "HHI + shape correction" is more natural than harmonic language.

#### 4.3.5 Classification helpers

Status: **empirical heuristic**.

```python
community_type_discriminant(X, *, method="pair", calibrator=None) -> dict
```

Requirements:

* Default `method="pair"` operates on the 2D forced coordinates directly, not on the Q/H ratio.
* Synthetic thresholds are optional presets (not universal defaults) available via `calibrator="synthetic_v0"`.
* Return dict includes `"labels"`, `"scores"`, `"method"`, and `"calibrator"`.

#### 4.3.6 Distributional shift

Status: **exact-derived utility**.

```python
distributional_shift(X_ref, X_test, *, summary="mean") -> dict
```

Purpose: measure geometric shift between two clouds of probability vectors (e.g. train vs test classifier outputs) under Fisher distance.

Returns:

* `"mean_distance"` — mean Fisher distance between nearest pairs.
* `"cloud_distance"` — Fisher distance between the Fisher means of the two clouds.
* `"ref_dispersion"` — mean Fisher distance from reference points to the reference cloud's Fisher mean.
* `"test_dispersion"` — mean Fisher distance from test points to the test cloud's Fisher mean.

---

### 4.4 Module: `generators`

Synthetic reference ensembles for tests, tutorials, and exploratory benchmarks.

#### 4.4.1 Required generators

```python
dirichlet_community(n, m, alpha=1.0, rng=None) -> ndarray
broken_stick(n, m, rng=None) -> ndarray
lognormal_community(n, m, *, mu=2.0, sigma=1.0, rng=None) -> ndarray
geometric_series(n, m, *, k_range=(0.3, 0.8), rng=None) -> ndarray
market_shares(n, m, *, concentration="mixed", rng=None) -> ndarray
portfolio_weights(n, m, *, style="mixed", rng=None) -> ndarray
```

Requirements:

* Every generator returns valid simplex arrays of shape `(m, n)`.
* Every generator accepts `numpy.random.Generator` for reproducibility.
* Every generator has `rng=None` default (creates a fresh default generator).

#### 4.4.2 Documentation status

All generator-based interpretations (e.g. "geometric communities have Q/H ratio ≈ 1.8") must be labeled: **synthetic benchmark**, not domain law.

---

### 4.5 Module: `frontier`

Experimental tools for the first non-forced symmetric-even enrichment.

#### 4.5.1 Degree-8 coordinates

Status: **experimental**.

```python
frontier8_coordinates(s) -> ndarray    # shape (4,) or (M, 4)
frontier8_batch(X) -> ndarray          # shape (M, 4)
```

Returns a 4-vector: `[Q_Δ, H_3, E8_1, E8_2]` where `E8_1` and `E8_2` are the two degree-8 enrichment coordinates, orthogonal to (Q_Δ, H_3) under the Dirichlet(1) measure.

Implementation guidance:

* v0.3 ships precomputed symbolic expressions for the degree-8 basis.
* Explicit polynomial expressions documented in the mathematical notes.
* Higher-degree generalization belongs in `harmonic`.

#### 4.5.2 Residual diagnostics

Status: **experimental**.

```python
frontier8_residual(X, target, *, model="linear") -> dict
```

Returns:

* `"r_squared_forced"` — R² of target against (Q_Δ, H_3) only.
* `"r_squared_frontier"` — R² of target against (Q_Δ, H_3, E8_1, E8_2).
* `"frontier_improvement"` — difference.
* `"needs_frontier"` — boolean, True if improvement > 0.01.

Purpose: quantify whether a target requires degree-8 enrichment beyond the forced pair.

---

### 4.6 Module: `harmonic` (experimental)

Low-degree symmetric-even harmonic tools in Fisher-lift coordinates.

#### 4.6.1 Scope

This module does **not** promise a fully general harmonic algebra engine.

v0.3 scope:

* Exact precomputed low-degree symmetric-even bases through degree 8.
* Exact low-degree dimension information via partition counting.
* Algorithmic basis construction for higher degrees with explicit "experimental" status.

#### 4.6.2 Functions

```python
symmetric_even_dimension(N, degree) -> int
symmetric_even_basis(N, degree, *, method="precomputed") -> list
project_to_degree(f, degree, *, N=None) -> object
filtration_decomposition(f, *, max_degree=8, N=None) -> dict
selective_frontier(N) -> dict
enrichment_space(N, degree, *, method="auto") -> list
```

`method` parameter:

* `"precomputed"` — use built-in bases (available through degree 8).
* `"algorithmic"` — compute from representation theory (experimental, any degree).
* `"auto"` — precomputed if available, algorithmic otherwise.

Documentation must distinguish exact built-ins from experimental higher-degree construction.

#### 4.6.3 Dimension formulas

The dimension of the S_N-symmetric even harmonic sector at degree 2k equals the number of partitions of k into at most N parts, minus the number of partitions of k−1 into at most N parts. This formula may be exposed as:

```python
_partition_count(k, max_parts) -> int
```

but is not required for normal workflows.

---

### 4.7 Module: `viz` (optional)

Matplotlib-based plotting utilities. Requires `matplotlib >= 3.5`.

#### 4.7.1 Plots

```python
scatter_forced_pair(X, *, color=None, ax=None) -> Axes
divergence_heatmap(n_range, *, ax=None) -> Axes
fisher_sphere(X, *, ax=None) -> Axes            # N=3 only
disagreement_curve(n_range, *, ax=None) -> Axes
forced_pair_histogram(X, *, ax=None) -> Axes
geodesic_plot(a, b, *, n_points=50, ax=None) -> Axes
```

#### 4.7.2 Design rules

* All plots accept an optional `ax` parameter and return the Axes object.
* Plotting is never required for core analysis workflows.
* No plot function imports matplotlib at module load time (lazy import).

---

### 4.8 Module: `utils`

Validation, projection, and perturbation utilities.

```python
validate_simplex(x, *, axis=-1, tol=1e-12, renormalize="warn") -> ndarray
project_to_simplex(x) -> ndarray
closure(x) -> ndarray
```

`validate_simplex` is also re-exported from `core` for convenience.

---

## 5. Immediate applications

Claims are deliberately conservative. Each application names what the library provides, what caution is warranted, and what a practical workflow looks like.

### 5.1 Information geometry and compositional data analysis

**Strongest immediate application.**

Problem: standard log-ratio methods require special handling at zeros. Many practitioners need principled distances, means, interpolation, and tangent analysis on simplex data.

Library contribution: boundary-safe Fisher/Hellinger geometry, exact Fisher distance, barycenters and geodesics, tangent PCA, canonical low-complexity symmetric summaries, and kernel methods.

Positioning: a canonical information-geometric interpretation of square-root/Hellinger methods, extended with exact spherical geometry and low-degree harmonic structure.

### 5.2 Ecology and biodiversity metrics

Exact contribution: for the specific question of when quadratic and multiplicative diversity measures must disagree, the binary case is completely resolved (Φ_2 = Ψ_2) and the N ≥ 3 case generically diverges. This is a complete structural answer to that specific comparison problem.

Caution: this is a structural clarification, not a complete resolution of all debates over which biodiversity index is normatively best.

Workflow: compute overlap diagnostics, inspect disagreement rates, supplement one-number summaries with the forced pair where relevant.

### 5.3 Economics and concentration analysis

Exact contribution: the forced-compression framework shows that one-number summaries (HHI) are insufficient to capture all low-degree symmetric-even structure beyond degree 4. The forced pair (Q_Δ, H_3) captures the next layer.

Caution: this justifies supplementing HHI with (Q_Δ, H_3); it does not by itself imply immediate policy replacement of existing thresholds.

Workflow: compute HHI and forced coordinates, detect HHI-matched but H_3-distinct cases, report a concentration profile rather than a single number.

### 5.4 Machine learning probability-output geometry

Framing: the package applies to the geometry of predicted probability vectors — clustering predictions under Fisher distance, comparing output clouds across datasets or training stages, detecting confidence concentration structure, measuring distributional shift between train and test predictions, and using (Q_Δ, H_3) as features for reliability or drift models.

Caution: metrics like ECE depend jointly on predictions and outcomes with binning choices. The package supports such analyses but does not provide theorem-level calibration claims from simplex geometry alone.

### 5.5 Microbiome and sparse compositional science

Strength: the Fisher lift handles zeros naturally and supports boundary-safe geometry. This is a direct advantage over log-ratio methods for zero-inflated OTU/ASV tables.

Positioning: same as §5.1 — canonical information-geometric interpretation of square-root/Hellinger methods.

### 5.6 Portfolio and allocation analysis

Portfolio weights live on the simplex. The package is applicable to concentration profiling, distance between portfolios, geodesic interpolation between allocations, and distinguishing HHI-matched portfolios with different higher-order structure.

### 5.7 Population genetics

**Exploratory only.** Allele-frequency vectors are simplex-valued and the library may help analyse where quadratic and multiplicative summaries part ways. However, statistics like F_ST involve between-population structure and should not be oversimplified as direct single-simplex invariants. This is a plausible application area, not a headline validation target.

---

## 6. Validation and tests

The test matrix distinguishes four categories.

### 6.1 Exact identity tests

| ID | Test | Assertion |
|----|------|-----------|
| E1 | Binary overlap coincidence | Φ_2 = Ψ_2 = 4s₁s₂ at machine precision, 1000 random binary compositions |
| E2 | Ternary generic divergence | Φ_3 ≠ Ψ_3 for > 95% of 1000 Dirichlet(1,1,1) samples |
| E3 | Q_Δ at uniform | Q_Δ(1/N, …, 1/N) = 0 for N = 3, 5, 10, 20 |
| E4 | H_3 at uniform | H_3(1/N, …, 1/N) = 2/(N²(N+2)) for N = 3, 5, 10, 20 |
| E5 | Q_Δ at vertex | Q_Δ(1, 0, …, 0) = 1 − 1/N for N = 3, 5, 10 |
| E6 | Fisher lift roundtrip | s → √s → (√s)² = s, atol 1e-14 |
| E7 | Fisher lift on sphere | ‖ψ‖² = 1, atol 1e-14 |
| E8 | Fisher lift positive | ψ_i ≥ 0 for all lifted points |

### 6.2 Exact geometry tests

| ID | Test | Assertion |
|----|------|-----------|
| G1 | Distance self-identity | fisher_distance(s, s) = 0 |
| G2 | Distance symmetry | fisher_distance(a, b) = fisher_distance(b, a) |
| G3 | Triangle inequality | d(a,c) ≤ d(a,b) + d(b,c), numerical check on 500 triples |
| G4 | Geodesic endpoints | fisher_geodesic(a, b, 0) = a, fisher_geodesic(a, b, 1) = b |
| G5 | Geodesic on simplex | fisher_geodesic(a, b, t) ∈ Δ^{N−1} for t ∈ {0, 0.25, 0.5, 0.75, 1} |
| G6 | Fisher mean on simplex | fisher_mean(X) ∈ Δ^{N−1} |
| G7 | Zero handling | fisher_distance, fisher_mean, fisher_geodesic accept compositions with zero entries without error |
| G8 | Hellinger-Fisher consistency | d_F(a,b) = 2 arccos(1 − d_H(a,b)²) within tolerance |
| G9 | Kernel-distance consistency | fisher_kernel(a,b) = cos(d_F(a,b)/2) within tolerance |
| G10 | Pairwise matrix shape | pairwise_fisher_distances(X) has shape (M, M) and is symmetric |
| G11 | Kernel matrix shape | kernel_matrix(X) has shape (M, M) and is symmetric positive semidefinite |

### 6.3 Numerical regression / factorization sanity tests

| ID | Test | Assertion |
|----|------|-----------|
| R1 | p₄ = Σs⁴ in forced block | R² > 0.999 against (Q_Δ, H_3) at N = 5, 10, 20 |
| R2 | p₅ = Σs⁵ beyond forced block | R²(p₅) < R²(p₄) at N = 30 |
| R3 | Degree-8 improvement | frontier8_residual shows R²_frontier > R²_forced for a degree-8 target |

These are implementation sanity checks, not substitutes for the symbolic theorems.

### 6.4 Empirical ensemble tests

| ID | Test | Assertion | Status |
|----|------|-----------|--------|
| S1 | Divergence growth | Mean |Φ−Ψ| increases with N under Dirichlet(1) | Distribution-dependent |
| S2 | Generator separation | Geometric and lognormal communities separate in (Q_Δ, H_3) space | Distribution-dependent |
| S3 | Binary zero divergence | divergence_analysis reports zero divergence for N=2 | Exact consequence |
| S4 | All generators valid | Every generator returns valid simplex arrays | Engineering check |

---

## 7. Documentation requirements

### 7.1 Three documentation layers

**Quick start:** load simplex data, compute Fisher distance, compute Fisher mean, compute forced coordinates, run divergence analysis. Five steps, one page.

**User guide:** geometry workflows, overlap diagnostics, forced-pair interpretation, tangent PCA, concentration profiling, distributional shift, frontier tools.

**Mathematical notes:** Fisher lift and metric pullback, overlap-family theorem, forced-compression theorem with scope qualifier, H_3 normalization convention, degree-8 frontier summary, known-vs-novel attribution.

### 7.2 Example notebooks

* Basic Fisher geometry on simplex data.
* Biodiversity / concentration comparison.
* Prediction-cloud geometry and distributional shift for classifier outputs.
* Sparse compositional data with zeros (microbiome-style).
* Experimental degree-8 enrichment demo.

---

## 8. Known vs novel

### 8.1 Known

The following are established in existing literature:

* Fisher information geometry on the simplex.
* Square-root / Hellinger embedding and its metric properties.
* Bhattacharyya coefficient and Hellinger distance.
* Spherical statistics on embedded simplex data.
* Herfindahl, Simpson, Shannon, and related diversity/concentration indices.
* General spherical harmonic theory on S^{N−1}.
* Aitchison geometry of compositional data.

### 8.2 Novel or programme-distinctive

The library's programme-distinctive mathematical content is believed to include:

* The overlap-family framing (Φ_N, Ψ_N) as canonical quadratic vs multiplicative overlap summaries.
* The binary coincidence / higher-dimensional divergence theorem as a structural organizing statement.
* The forced-compression interpretation of (Q_Δ, H_3) as the canonical low-degree resolving package in the Fisher-lifted symmetric-even sector.
* The explicit first selective frontier at degree 8 with 2-dimensional enrichment.
* Engineering exposure of frontier coordinates as practical experimental tools.

### 8.3 Borderline / adjacent

The use of the Fisher lift as an alternative to log-ratio workflows is not claimed as wholly novel. The contribution is the disciplined combination of exact geometry, boundary-safe workflows, overlap diagnostics, and harmonic low-complexity tools into a single coherent toolkit.

---

## 9. Dependencies

### Required

* Python ≥ 3.9
* NumPy ≥ 1.21
* SciPy ≥ 1.7

### Optional

* matplotlib ≥ 3.5 (viz module)
* pytest ≥ 7.0 (development)
* scikit-learn (optional adapter utilities only; not required)

No compiled extensions required.

---

## 10. Roadmap

### Phase 1 — v0.3 core release

Ship: `core`, `geometry`, `analysis`, `generators`, `utils`, basic tests, basic docs.

### Phase 2 — v0.4 analysis and visualization

Ship: `viz`, richer reports, tutorial notebooks, benchmark examples.

### Phase 3 — v0.5 frontier tools

Ship: `frontier`, degree-8 coordinates, frontier residual diagnostics.

### Phase 4 — v0.6 harmonic experimental layer

Ship: low-degree exact bases through degree 8, experimental higher-degree basis construction.

### Phase 5 — real-data validation

Run and document domain-specific validation studies against published datasets.

### Phase 6 — multi-language ports

R and JS/TS implementations.

---

## 11. Out of scope

The following do not belong in the main `fisher-simplex` package at v0.3:

* Carrier-geometry cone-law / Stokes-style modules.
* Non-simplex UFT structures.
* Infinite-dimensional research extensions.
* Policy or normative claims beyond what the current math supports.

These may become separate future packages.

---

## 12. Public positioning

One-sentence description:

> `fisher-simplex` is a boundary-safe information-geometry toolkit for probability-simplex data, built around the Fisher/Hellinger square-root embedding and extended with canonical low-complexity invariant diagnostics.

Short pitch:

> Use it when your data are normalized nonnegative vectors and you need principled distances, means, interpolation, and low-order structural summaries without ad hoc zero handling.

---

## 13. Implementation priorities

Implementation proceeds in this order:

1. `validate_simplex` and input handling
2. `fisher_lift`, `fisher_project`
3. `bhattacharyya_coefficient`, `hellinger_distance`, `fisher_distance`
4. `pairwise_fisher_distances`, `kernel_matrix`
5. `fisher_mean`, `fisher_geodesic`, `geodesic_interpolate`
6. `phi`, `psi`, `overlap_divergence`
7. `q_delta`, `h3`, `forced_pair`, `forced_coordinates`
8. `batch_diagnostic`, `divergence_analysis`, `pairwise_ranking_disagreement`
9. `fisher_logmap`, `fisher_expmap`, `fisher_pca`
10. `distributional_shift`
11. Generators
12. Reports (`comparative_index_report`, `concentration_profile_report`)
13. Visualization
14. Frontier tools
15. Harmonic experimental layer

This ordering matches mathematical certainty and likely user adoption.

---

## 14. Acceptance criteria for v0.3

v0.3 is complete when all of the following hold:

1. All exact geometry and invariant functions are implemented.
2. Input validation behavior is explicit and documented with configurable `renormalize` modes.
3. Zero-containing simplex data is supported throughout the geometry and core layers without error.
4. Pairwise distance and kernel functions return correctly shaped `(M, M)` matrices.
5. Fisher mean documentation explicitly states extrinsic vs intrinsic distinction.
6. H_3 normalization convention is documented with the exact uniform-point value 2/(N²(N+2)).
7. Docs include at least one end-to-end example using Fisher distance, Fisher mean, forced coordinates, and divergence analysis.
8. Test suite separates exact identity tests (E-series), geometry tests (G-series), regression sanity tests (R-series), and empirical ensemble tests (S-series).
9. No documentation page blurs the scope of the forced-compression theorem beyond the Fisher-lifted symmetric-even sector.
10. No flagship example relies on unstable `qh_ratio` thresholds.
11. Public README positions the package primarily as a geometry toolkit, secondarily as a low-complexity invariant toolkit.
12. `distributional_shift` is implemented and has at least one tutorial example.
13. Tangent-space boundary behavior is documented for base points with zero components.
14. Degree-8 enrichment basis is documented with orthogonality condition (Dirichlet(1) measure) and explicit polynomial expressions.

---

## 15. License and attribution

Code: MIT license unless project governance decides otherwise.

Attribution distinguishes:

* the known statistical/information-geometric background,
* from the programme-distinctive overlap and forced-compression content.

Software citation:

> `fisher-simplex`: Canonical simplex geometry via the Fisher amplitude lift, with low-complexity invariant diagnostics. Version 0.3.

Mathematical attribution:

> Fisher-Harmonic programme: overlap-family divergence, low-degree forced compression in the Fisher-lifted symmetric-even sector, and degree-8 selective frontier tools for simplex observables.
