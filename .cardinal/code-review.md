# Code Review — fisher-simplex v0.3

## Summary

**17 items reviewed, 3 fixed, 4 deferred (all LOW severity).**

The codebase is well-structured, correct in its core mathematics, and comprehensively tested (248 tests, all passing). Three issues were fixed during review: one HIGH (spec-noncompliant `fisher_coords` in diagnostics), one MEDIUM (missing `sigma > 0` guard), and one MEDIUM (missing docstring content). Four LOW-severity items are documented for future improvement.

---

## Spec Compliance

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | All exact geometry/invariant functions implemented | PASS | All functions from spec §4.1–4.8 present in source: core.py, geometry.py, analysis.py, generators.py, frontier.py, harmonic.py, viz.py, utils.py |
| 2 | Input validation explicit and configurable | PASS | `validate_simplex` in utils.py:15–92 implements "never"/"warn"/"always" modes per spec §4.1.1 |
| 3 | Zero-containing data supported | PASS | Test G7 (test_geometry.py:96–115) verifies fisher_distance, fisher_mean, fisher_geodesic with zeros |
| 4 | Pairwise functions return (M,M) | PASS | Tests G10 (test_geometry.py:137–146), G11 (test_geometry.py:148–158) verify shape and symmetry |
| 5 | Fisher mean extrinsic/intrinsic documented | PASS | geometry.py:231–238 docstring contains required statement (enhanced during review) |
| 6 | H_3 convention documented with 2/(N²(N+2)) | PASS | docs/mathematical_notes.md:104–127 section 4 documents exact convention with worked examples |
| 7 | End-to-end example exists | PASS | examples/basic_geometry.py covers Fisher distance, mean, geodesic, forced coords, PCA, kernel |
| 8 | Tests separated into E/G/R/S series | PASS | test_core.py:TestESeries (E1–E8), test_geometry.py:TestGSeries (G1–G11), test_analysis.py:TestRSeries (R1–R2), TestSSeries (S1–S3) |
| 9 | Forced-compression scope not blurred | PASS | user_guide.md:101–102 correctly scopes to "Fisher-lifted sector"; mathematical_notes.md:72–85 has full qualifier |
| 10 | No flagship qh_ratio | PASS | grep of examples/, docs/, README.md finds zero mentions of qh_ratio |
| 11 | README positioned correctly | PASS | README.md:9–11: "Primarily a geometry toolkit... Secondarily a low-complexity invariant toolkit" |
| 12 | distributional_shift with tutorial | PASS | examples/distributional_shift.py (81 lines) demonstrates full workflow including no-shift baseline |
| 13 | Tangent boundary documented | PASS | geometry.py fisher_logmap docstring Notes section (added during review); mathematical_notes.md:149–161 section 6 |
| 14 | Degree-8 documented with orthogonality | PASS | mathematical_notes.md:164–201 section 7 documents orthogonality condition, construction, and polynomial expressions |

---

## Issues Found

### HIGH

- **batch_diagnostic/full_diagnostic `fisher_coords` field wrong**: analysis.py:91 — Returned `forced_coordinates(X)` (shape M,2) instead of `fisher_lift(X)` (shape M,N) as specified in spec §4.3.1. — **FIXED**
  - Changed to `fisher_lift(X)` in both `batch_diagnostic` and `full_diagnostic`
  - Updated tests in test_analysis.py to expect correct shapes (M,N) and (N,)

### MEDIUM

- **kernel_matrix sigma=0 division by zero**: geometry.py:209–214 — `kind="hellinger_rbf"` with `sigma=0` would produce NaN on diagonal (0/0). — **FIXED**
  - Added `sigma <= 0` guard raising ValueError

- **fisher_logmap missing boundary documentation**: geometry.py:366 — Docstring lacked the tangent boundary behavior note required by spec acceptance criterion 13 and spec §4.2.5. — **FIXED**
  - Added Notes section documenting boundary behavior per spec

- **fisher_mean docstring incomplete**: geometry.py:231 — Docstring mentioned extrinsic/intrinsic but lacked the full required statement from spec §4.2.3. — **FIXED** (enhanced to include full spec-required text)

### LOW

- **Redundant validation in call chains**: Multiple public functions call `_validated(s)` then call other public functions that also validate (e.g., `overlap_divergence` validates, then calls `phi` and `psi_overlap` which each validate again; `forced_pair` validates, then calls `q_delta` and `h3` which each validate). Not a correctness issue but causes ~3x overhead on validation for some call paths.

- **O(M²) Python loop in pairwise_ranking_disagreement**: analysis.py:165–170 — Uses nested Python loops for M² pairwise comparisons. For large M this is slow. Could be vectorized with NumPy broadcasting.

- **O(M²) Python loop in distributional_shift cross distances**: analysis.py:634–636 — Uses nested Python loops for cross-distance matrix. Could use vectorized pairwise computation.

- **Missing explicit status labels in some core docstrings**: Spec §2 requires every feature documented with status (exact/exact-derived/empirical heuristic/experimental). Some core.py functions (phi, psi_overlap, q_delta, h3, herfindahl, etc.) lack explicit "Status: exact" labels in their docstrings. The status is correctly stated in the module docstring and mathematical notes, but individual function docstrings are inconsistent.

---

## Code Quality Assessment

### Numerical Stability
- `psi_overlap` (core.py:98–126): Correctly uses log-space computation to avoid overflow for N^N * prod(s_i). Zero detection handles boundary correctly.
- `fisher_distance` (core.py/geometry.py:99–100): Clips arccos argument to [-1, 1]. Self-distance tolerance correctly explained in tests (arccos amplification near 1.0).
- `fisher_logmap` (geometry.py): Direction normalization guarded with `dir_norms > 1e-30` threshold.
- `shannon_entropy` (core.py:352–358): Uses `0 * log(0) = 0` convention via safe replacement.
- `qh_ratio` (core.py:234–285): Three on_zero modes correctly implemented.

### Input Handling
- All public functions in core.py and geometry.py accept both `(N,)` and `(M, N)` input. Verified by TestBatchShapes class.
- `validate_simplex` correctly handles 1D and 2D inputs with all three renormalize modes.

### Type Annotations
- All public function signatures have type annotations using `ArrayLike`, `NDArray[np.floating]`, and standard types.
- Uses `from __future__ import annotations` for deferred evaluation.

### Docstrings
- All public functions have docstrings with Parameters and Returns sections.
- Some functions lack explicit status labels (noted as LOW).

### Dead Code / TODOs
- No TODO, FIXME, HACK, or XXX comments found in source (verified via grep).
- No dead code detected.

### Security/Correctness
- `effective_number_herfindahl`: Division by `herfindahl(s)` is safe — for valid simplex data, HHI >= 1/N > 0.
- `closure`: No guard against zero-sum input, but this is expected behavior for a utility function.
- `NaN propagation in shannon_entropy with zeros`: Correctly handled via safe_s replacement (core.py:354).
- `kernel_matrix sigma=0`: Now guarded (fixed during review).

### Test Coverage
- 248 tests across 10 test files covering all public functions.
- E-series (E1–E8): Foundational invariant identities.
- G-series (G1–G11): Geometry properties (metric axioms, consistency relations).
- R-series (R1–R2): Regression sanity checks for forced-block membership.
- S-series (S1–S4): Statistical properties and generator validity.
- Edge cases tested: zero entries, vertices, uniform compositions, batch vs single input.
- Viz module has smoke tests with lazy-import verification.

---

## Remaining Items

| Issue | Severity | Disposition |
|-------|----------|-------------|
| Redundant validation in composed function calls | LOW | Defer — correctness preserved, optimize if profiling shows need |
| O(M²) Python loops in pairwise_ranking_disagreement | LOW | Defer — acceptable for typical M < 1000; vectorize if perf needed |
| O(M²) Python loops in distributional_shift | LOW | Defer — acceptable for typical cloud sizes; vectorize if perf needed |
| Missing status labels in individual core docstrings | LOW | Defer — status is documented in module docstring and math notes |
