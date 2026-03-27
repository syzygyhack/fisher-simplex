# Mathematical notes

Rigorous mathematical background for the `fisher-simplex` library. Distinguishes known results from programme-distinctive content per spec section 8.

---

## 1. Fisher lift and metric pullback

### The embedding (known)

The canonical map from the probability simplex to the positive orthant of the unit sphere:

    s_i = psi_i^2,    psi_i >= 0,    sum(psi_i^2) = 1

sends Delta^{N-1} to S^{N-1}_+ (the nonneg orthant of the unit sphere in R^N).

### Metric pullback (known)

The Fisher information metric on the simplex is:

    g_ij^F = delta_ij / (4 s_i)

Under the square-root substitution psi_i = sqrt(s_i), the pullback of the Fisher metric to the sphere gives:

    ds^2_F = sum_i (ds_i)^2 / (4 s_i)
           = sum_i (2 psi_i d(psi_i))^2 / (4 psi_i^2)
           = sum_i (d(psi_i))^2

which is 1/4 of the round metric on S^{N-1} when using the geodesic distance convention d_F = 2 * arccos(sum sqrt(a_i b_i)). Equivalently, the Fisher distance between two simplex points equals twice the great-circle angle between their amplitude representations.

This identification is standard in information geometry and statistics (Amari, Cencov, et al.). The library's contribution is not the lift itself but its use as the organizing structure for the full toolkit.

### Distance conventions (known)

**Fisher distance:**

    d_F(p, q) = 2 * arccos(B(p, q))

where `B(p, q) = sum(sqrt(p_i * q_i))` is the Bhattacharyya coefficient. This equals twice the great-circle angle between the amplitude vectors.

**Hellinger distance:**

The library uses the normalized convention:

    d_H(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2

This gives `d_H in [0, 1]` and `d_H^2 = 1 - B(p, q)`. The unnormalized convention (used in some references and in the paper) omits the `1/sqrt(2)` factor, giving `d_H^2 = 2(1 - B)`. The Fisher-Hellinger relation in the library's convention is:

    d_F = 2 * arccos(1 - d_H^2)

Both conventions yield the same Fisher distance.

### Kernels (known)

The Fisher amplitude lift induces a family of positive-definite kernels on the simplex:

- **Linear Fisher kernel:** `K_1(p, q) = B(p, q)` — the Bhattacharyya coefficient.
- **Polynomial Fisher kernel:** `K_d(p, q) = B(p, q)^d` for integer `d >= 1`.
- **Fisher RBF kernel:** `K_sigma(p, q) = exp(-d_F(p, q)^2 / (2 * sigma^2))`.

These are positive definite because they are induced from positive-definite kernels on the sphere. The polynomial kernels correspond to inner products in the degree-d tensor power of the ambient space; the RBF kernel is a standard radial construction in Fisher geodesic distance.

---

## 2. Overlap-family theorem

### Definitions (programme-distinctive framing)

Two canonical scalar overlap families on Delta^{N-1}:

    Phi_N(s) = (N/(N-1)) * (1 - sum(s_i^2))
    Psi_N(s) = N^N * prod(s_i)

Phi_N is a normalized quadratic overlap (Gini-Simpson type). Psi_N is a normalized multiplicative overlap (entropy-adjacent).

### Binary coincidence (exact)

For N = 2 with s = (x, 1-x):

    Phi_2(s) = 2(1 - x^2 - (1-x)^2) = 4x(1-x)
    Psi_2(s) = 4 * x * (1-x) = 4x(1-x)

Therefore Phi_2 = Psi_2 identically.

### Higher-dimensional divergence (exact)

For N >= 3, Phi_N and Psi_N are generically distinct functions on Delta^{N-1}. Specifically:

- Phi is a degree-2 polynomial in the simplex coordinates.
- Psi is a degree-N monomial (product form).

For N >= 3 these are algebraically independent, so the overlap divergence D(s) = Phi_N(s) - Psi_N(s) is generically nonzero. Empirically, D vanishes only on measure-zero subsets of the simplex.

This gives a complete structural answer to: "When must quadratic and multiplicative overlap summaries diverge?" It does not settle which summary is normatively preferred in a given domain.

---

## 3. Forced-compression theorem

### Statement (programme-distinctive)

**Scope qualifier:** This theorem applies to the **Fisher-lifted symmetric-even sector** specifically.

Within the S_N-symmetric even sector of spherical harmonics on S^{N-1} (pulled back from the Fisher lift), the nonconstant structure through degree 6 is forced through two invariants:

    Q_delta = sum(s_i^2) - 1/N
    H_3 = sum(s_i^3) - (3/(N+2)) * sum(s_i^2) + 2/(N*(N+2))

**Engineering-facing statement:** Through degree 6 in the Fisher-lifted symmetric-even sector, the algebra of nonconstant S_N-symmetric observables factors through the pair (Q_delta, H_3).

### Scope qualifications

This does **not** mean that every practical low-order statistic on simplex data factors through (Q_delta, H_3). The theorem applies to the Fisher-lifted symmetric-even sector specifically.

Applied claims about arbitrary target observables must be tested empirically using the `sufficient_statistic_efficiency` workflow, not asserted from the theorem alone.

### Dimension counting

The S_N-symmetric even harmonic sector at degree 2k has dimension:

    dim = P(k, N) - P(k-1, N)

where P(k, N) counts the number of partitions of k into at most N parts.

Through degree 6 (k = 1, 2, 3):
- Degree 2 (k=1): P(1, N) - P(0, N) = 1 - 1 = 0 nonconstant dimensions.
- Degree 4 (k=2): P(2, N) - P(1, N) = 2 - 1 = 1 dimension (Q_delta).
- Degree 6 (k=3): P(3, N) - P(2, N) = 3 - 2 = 1 dimension (H_3), for N >= 3.

Total nonconstant dimensions through degree 6: 2, exactly spanned by (Q_delta, H_3).

---

## 4. H_3 normalization convention

The implementation uses:

    H_3 = sum(s_i^3) - (3/(N+2)) * sum(s_i^2) + 2/(N*(N+2))

This formula does **not** vanish at the uniform composition. At s = (1/N, ..., 1/N):

    sum(s_i^3) = N * (1/N)^3 = 1/N^2
    sum(s_i^2) = N * (1/N)^2 = 1/N

    H_3(uniform) = 1/N^2 - 3/(N(N+2)) + 2/(N(N+2))
                 = 1/N^2 - 1/(N(N+2))
                 = (N+2-N) / (N^2(N+2))
                 = 2 / (N^2(N+2))

This is a small positive value, not zero. The formula is the natural degree-6 harmonic in the Fisher-lifted sector; artificially centering it to vanish at the barycenter would break the harmonic interpretation.

Verification for specific N:
- N=3: H_3(uniform) = 2/(9*5) = 2/45 ~ 0.04444
- N=5: H_3(uniform) = 2/(25*7) = 2/175 ~ 0.01143
- N=10: H_3(uniform) = 2/(100*12) = 2/1200 ~ 0.00167
- N=20: H_3(uniform) = 2/(400*22) = 2/8800 ~ 0.000227

---

## 5. Fisher mean: extrinsic vs intrinsic

The `fisher_mean` function computes the **extrinsic (projected arithmetic) mean** in Fisher-lift amplitude coordinates:

1. Lift each composition to amplitude space: psi_i = sqrt(s_i).
2. Compute the weighted arithmetic mean: psi_bar = sum(w_i * psi_i).
3. Normalize to the unit sphere: psi_bar / ||psi_bar||.
4. Project back to the simplex: s_i = psi_bar_i^2.

This is **not** the intrinsic Frechet mean, which minimizes the sum of squared geodesic distances:

    mu_F = argmin_{mu in Delta} sum_i w_i * d_F(mu, s_i)^2

The extrinsic mean coincides with the Frechet mean to first order when data are concentrated in a small region of the simplex. For widely spread data, the two may differ.

An iterative intrinsic Frechet mean solver may be added in a future version.

---

## 6. Tangent-space boundary behavior

The log map `fisher_logmap` maps simplex compositions to tangent vectors at a base point in the Fisher-lifted spherical geometry.

**Boundary behavior:** When the base point lies on the simplex boundary (has zero components), the effective tangent space dimension drops. The tangent vector has zeros in the corresponding components, and the tangent space is degenerate in those directions.

The function remains well-defined but emits a warning when the base point has zeros. The log map produces tangent vectors that are confined to the subspace where the base point has nonzero support.

Practically:
- If `base = (0.5, 0.5, 0, 0)`, the tangent space is effectively 2-dimensional even though the ambient dimension is 4.
- PCA in such a tangent space will have at most 2 nonzero singular values.
- This is mathematically correct (the boundary of the simplex maps to the boundary of the positive orthant on the sphere, which has lower-dimensional tangent spaces), but users should be aware of it when analyzing data with structural zeros.

---

## 7. Degree-8 frontier

### First selective frontier (programme-distinctive)

At degree 8, the symmetric-even harmonic sector gains its first enrichment beyond the forced pair. The dimension at degree 8 is:

    dim_8 = P(4, N) - P(3, N)

For N >= 4, this equals 2 (since P(4,4) - P(3,4) = 5 - 3 = 2). The 2-dimensional enrichment *subspace* is canonical (determined by the representation theory), but any specific basis for it depends on the orthogonalization convention. The library's E8_1 and E8_2 are one such basis, produced by Gram-Schmidt orthogonalization of {p_4, p_2^2} against the forced block under the Dirichlet(1) inner product.

### Orthogonality condition

E8_1 and E8_2 are orthogonal to the forced block {1, Q_delta, H_3} under the Dirichlet(1) (uniform) measure on the simplex:

    integral_{Delta} E8_k(s) * f(s) * Dir(s|1,...,1) ds = 0

for f in {1, Q_delta, H_3} and k in {1, 2}.

### Construction

The enrichment basis is constructed by modified Gram-Schmidt orthogonalization in the inner product space defined by the Dirichlet(1) measure. Starting from the basis of degree-4 (in s) symmetric polynomials {1, p_2, p_3, p_4, p_2^2} where p_k = sum(s_i^k):

1. The first three Gram-Schmidt directions span the forced block {1, p_2, p_3}.
2. The last two directions are orthogonal to the forced block and define E8_1, E8_2.

### Explicit polynomial expressions

E8_1 and E8_2 are linear combinations of {1, p_2, p_3, p_4, p_2^2}:

    E8_k(s) = c_k0 * 1 + c_k1 * p_2 + c_k2 * p_3 + c_k3 * p_4 + c_k4 * p_2^2

The coefficients (c_k0, ..., c_k4) depend on N and are computed from Dirichlet(1) moments via the set-partition expansion. The Gram matrix G[i,j] = E[b_i * b_j] under Dirichlet(1) is built using:

    E[p_{k1} * ... * p_{km}] = (sum over set partitions of index coincidences) / Pochhammer(N, K)

where K = sum(k_i).

The implementation caches these coefficients per N via `lru_cache`.

---

## 8. Known-vs-novel attribution

### Known (established literature)

- Fisher information geometry on the simplex (Amari, Cencov, et al.)
- Square-root / Hellinger embedding and its metric properties
- Bhattacharyya coefficient and Hellinger distance
- Spherical statistics on embedded simplex data
- Herfindahl, Simpson, Shannon, and related diversity/concentration indices
- General spherical harmonic theory on S^{N-1}
- Aitchison geometry of compositional data

### Programme-distinctive

- The overlap-family framing (Phi_N, Psi_N) as canonical quadratic vs multiplicative overlap summaries
- The binary coincidence / higher-dimensional divergence theorem as a structural organizing statement
- The forced-compression interpretation of (Q_delta, H_3) as the canonical low-degree resolving package in the Fisher-lifted symmetric-even sector
- The explicit first selective frontier at degree 8 with 2-dimensional enrichment
- Engineering exposure of frontier coordinates as practical experimental tools

### Borderline

The use of the Fisher lift as an alternative to log-ratio workflows is not claimed as wholly novel. The contribution is the disciplined combination of exact geometry, boundary-safe workflows, overlap diagnostics, and harmonic low-complexity tools into a single coherent toolkit.
