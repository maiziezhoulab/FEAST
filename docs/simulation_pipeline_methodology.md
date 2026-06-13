# FEAST Single-Slice Simulation Pipeline

This document describes the production simulation pipeline implemented in
`src/FEAST/FEAST_core/`.  The pipeline takes a spatial transcriptomics count
matrix as input and produces a simulated count matrix with matched gene-level
statistical properties.

---

## 1. Overview

```
Input: AnnData (N spots × G genes)
│
├─ 1. Compute per-gene statistics (μ_g, σ²_g, π_g)
│
├─ 2. Fit marginal distributions
│      μ  → Student's T mixture (max 15 components, log10 scale)
│      σ² → Student's T mixture (max 15 components, log10 scale)
│      π  → Beta mixture (max 8 components, raw scale)
│
├─ 3. Fit vine copula on pseudo-observations (BIC-selected structure)
│
├─ 4. Sample αG synthetic (μ̃, σ̃², π̃) from copula + PPF
│
├─ 5. Assign synthetic → real genes via hybrid OT
│      cost = α · cost_log + (1−α) · cost_raw
│
├─ 6. Convert assigned (μ̂, σ̂², π̂) → count model params
│      Model selection via excess-zero heuristic
│      Parameter estimation via log10 moment-matching (L-BFGS-B)
│
├─ 7. PPF count decoding (inverse CDF of fitted count distribution)
│
Output: AnnData (N spots × G genes, simulated counts)
```

**Production defaults:**

| Parameter | Value |
|-----------|-------|
| Assignment weights | `{'mean': 3.0, 'variance': 1.0, 'zero_prop': 1.0}` |
| Overgeneration factor | 2.0 |
| Hybrid alpha (log weight) | 0.2 |
| Boundary multiplier | 1.1 |
| Clip overshoot factor | 0.0 |

---

## 2. Parameter Cloud Construction

### 2.1 Gene Statistics

For each gene *g* in the count matrix **X** ∈ ℝ^(N×G):

```
μ_g = (1/N) Σᵢ X_ig
σ²_g = (1/N) Σᵢ (X_ig − μ_g)²
π_g = (1/N) Σᵢ 1(X_ig = 0)
```

The resulting cloud {p_g = (μ_g, σ²_g, π_g)}^G_{g=1} forms the empirical
joint distribution over gene-level statistics.

→ `GeneParameterSimulator.fit_statistics_only()` — parameter_cloud.py:168

### 2.2 Marginal Models

Three mixture models are fitted independently:

| Parameter | Model | Max Components | Transform |
|-----------|-------|---------------|-----------|
| mean (μ) | Student's T mixture | 15 | log10 |
| variance (σ²) | Student's T mixture | 15 | log10 |
| zero_prop (π) | Beta mixture | 8 | none |

For mean and variance, the log10 transform stabilizes the heavy right tail:

```
processed = log10(data + 1e-10)    # before fitting
samples   = 10^{PPF(u)}            # after inverse-CDF
```

The Student's T mixture fits *K* components via EM with BIC-based restart
selection.  Each component *k* has weight w_k, location θ_k, scale σ_k, and
degrees of freedom ν_k:

```
f(x) = Σ_k w_k · StudentT(x | θ_k, σ_k, ν_k)
```

→ `StudentTMixtureMarginalModeler` — StudentT_mixture_model.py:43

### 2.3 Vine Copula Dependency

Pseudo-observations (rank-based uniforms) are computed from the marginal CDFs:

```
U₁ = F_μ(μ_g),   U₂ = F_σ²(σ²_g),   U₃ = F_π(π_g)
```

A 3-dimensional vine copula is fitted to these uniforms using
`pyvinecopulib.Vinecop` with BIC-based family selection.

→ `DependencyModeler.fit_copula_model()` — parameter_cloud.py:120

---

## 3. Synthetic Parameter Generation

### 3.1 Copula Sampling

The fitted vine copula generates *αG* synthetic uniform triples, where
α = overgeneration_factor (default 2.0):

```
(u₁⁽ʲ⁾, u₂⁽ʲ⁾, u₃⁽ʲ⁾) ∼ Copula,   j = 1, …, αG
```

### 3.2 Inverse Marginal Transform

Each uniform is mapped back to the original scale via the marginal PPF
(percent-point function = inverse CDF):

```
μ̃⁽ʲ⁾  = F_μ^{-1}(u₁⁽ʲ⁾)     # 10^{StudentT_mixture.ppf(u₁)}
σ̃²⁽ʲ⁾ = F_σ²^{-1}(u₂⁽ʲ⁾)   # 10^{StudentT_mixture.ppf(u₂)}
π̃⁽ʲ⁾  = F_π^{-1}(u₃⁽ʲ⁾)     # Beta_mixture.ppf(u₃)
```

Values are clipped to `[min_observed, +∞)` for mean and variance, and
`[min_observed, 1.0]` for zero proportion.

→ `GeneParameterSimulator.simulate()` — parameter_cloud.py:204

---

## 4. Hybrid OT Assignment

### 4.1 Cost Construction

For each synthetic profile, two cost matrices are computed in parallel:

**Log-space cost** (captures relative-scale similarity):
```
C_log(i, j) = || z_log(real_i) × w  −  z_log(synth_j) × w ||₂
```
where z_log(x) = StandardScaler(log10(clip(x, 1e-10))).

**Raw z-score cost** (captures absolute-scale similarity, more sensitive to
high-expression genes):
```
C_raw(i, j) = || z_raw(real_i) × w  −  z_raw(synth_j) × w ||₂
```
where z_raw(x) = StandardScaler(clip(x, 0)).

**Weights**: `w = [3.0, 1.0, 1.0]` for (mean, variance, zero_prop).

### 4.2 Hybrid Combination

The two costs are normalized by their shared mean (NOT independently to [0,1])
so that the mixing parameter α meaningfully controls the trade-off:

```
C(i, j) = α · C_log(i,j)/norm  +  (1−α) · C_raw(i,j)/norm
norm = max(mean(C_log), mean(C_raw), 1e-10)
```

| α | Interpretation |
|----|---------------|
| 1.0 | Pure log-space (old default) |
| 0.2 | Production: 20% log + 80% raw |
| 0.0 | Pure raw z-score |

### 4.3 Optimal Transport

The Hungarian algorithm (linear_sum_assignment) solves the G × αG cost
matrix.  All *αG* candidates are used — the overgenerated pool is *not*
subsetted before OT.

→ `GeneParameterSimulator.assign_to_genes()` — parameter_cloud.py:226

---

## 5. Count Model Conversion

For each gene with assigned target (μ̂, σ̂², π̂):

### 5.1 Model Selection via Excess-Zero Heuristic

```
if μ̂ ≤ 1e-8:                           → Poisson
if not overdispersed (σ̂²/μ̂ ≤ 1.5):
    if π̂ > Poisson_zero + tolerance:     → ZIP
    else:                                 → Poisson
if overdispersed (σ̂²/μ̂ > 1.5):
    if π̂ > NB_zero + tolerance:          → ZINB
    else:                                 → NB
```

where `Poisson_zero = exp(-μ̂)`, `NB_zero = (r/(r+μ̂))^r` with
`r = μ̂²/(σ̂² − μ̂)`, and `tolerance = 0.05`.

This replaces the old raw-zero-proportion threshold (π̂ > 0.3), which
misclassified low-expression Poisson genes as zero-inflated.

→ `_select_model_with_heuristic()` — parameter_cloud.py:719

### 5.2 Parameter Estimation

**Poisson**: `λ = μ̂`

**NB**: Direct moment matching.
```
μ = μ̂
r = μ̂² / (σ̂² − μ̂)        # size parameter (= 1/dispersion)
```
Returns `{'mu': μ, 'r': r}`.

**ZIP(π, λ)**: Numerical optimization minimizing the log10-scale error on all
three moments:
```
min  (log10 E[X]     − log10 μ̂)²
   + (log10 Var(X)   − log10 σ̂²)²
   + (log10 P(X = 0) − log10 π̂)²
```
where:
```
E[X]     = (1 − π)λ
Var(X)   = (1 − π)(λ + πλ²)
P(X = 0) = π + (1 − π)e^{−λ}
```

**ZINB(π, μ, r)**: Same objective, where *r* is the NB size parameter
(r = 1/α in the paper's notation):
```
E[X]     = (1 − π)μ
Var(X)   = (1 − π)(μ + μ²/r + πμ²)
P(X = 0) = π + (1 − π)·[r/(r + μ)]^r
```

Bounds: π ∈ [1e-6, 1−1e-6], μ ∈ [1e-6, ∞), r ∈ [1e-6, ∞).
Optimizer: L-BFGS-B, max 5000 iterations.
Fallback: method-of-moments initial guess when `result.fun ≥ 1e-4`.

→ `_calculate_zip_theoretical_stats()` — parameter_cloud.py:531
→ `_calculate_zinb_theoretical_stats()` — parameter_cloud.py:539
→ `_moment_objective_function_log_scale()` — parameter_cloud.py:556

### 5.3 Parameter Storage

All distributions store 3-element arrays in `marginal_param1`:
```
[pi0, r_or_inf, mean_param]
```

| Distribution | pi0 | r_or_inf | mean_param |
|-------------|------|----------|------------|
| Poisson | 0.0 | np.inf | λ |
| NB | 0.0 | r (size) | μ |
| ZIP | π | np.inf | λ |
| ZINB | π | r (size) | μ |

→ `convert_params_for_new_simulator()` — parameter_cloud.py:729

---

## 6. Count Decoding (PPF)

### 6.1 Quantile Generation

| Calibration | Quantile Source |
|------------|----------------|
| `raw` (FEAST_OT) | q_ig ∼ Uniform(0, 1) i.i.d. |
| `reference_rank` (FEAST_Rank) | q_ig = X_ref[i, g], decoder calibration = 'rank' |

### 6.2 PPF Decoding

For each gene, quantiles are mapped through the inverse CDF of the fitted
count distribution. Quantiles are clipped to [1e-6, 1−1e-6] for numerical
stability.

**Poisson**: `counts = Poisson.ppf(q | λ = mean_param)`

**NB**: `counts = NegBinomial.ppf(q | n = r, p = r/(r + mean_param))`

**ZIP**: Non-zero component via adjusted quantile:
```
q_adj = clip((q − π)/(1 − π), 0, 1)
counts = Poisson.ppf(q_adj | λ = mean_param)
counts[q ≤ π] = 0
```

**ZINB**: Same adjustment with NB non-zero component:
```
q_adj = clip((q − π)/(1 − π), 0, 1)
counts = NegBinomial.ppf(q_adj | n = r, p = r/(r + mean_param))
counts[q ≤ π] = 0
```

### 6.3 Boundary Clipping

```
boundary_g = max_spots(X_ref[:, g]) × boundary_multiplier
counts[:, g] = min(counts[:, g], boundary_g)
```

For genes where `boundary_g < 1.0` and fitted mean > 1e-6, the boundary
is relaxed to infinity (no clip).

→ `decode_counts_from_quantiles()` — count_decoding.py:124
→ `_boundary_per_gene()` — count_decoding.py:58

---

## 7. Key Changes from Previous Version

| Component | Old | New |
|-----------|-----|-----|
| Assignment | Copula-rank OT, weights [1,1,1] | Hybrid OT (log + raw z-score), weights [3,1,1] |
| Overgeneration | 1.1×, discarded before OT | 2.0×, all candidates reach Hungarian |
| Cost normalization | Independent [0,1] per matrix | Shared mean normalization |
| Model selection | Raw zp > 0.3 → ZIP/ZINB | Excess-zero vs Poisson/NB expectation |
| Heuristic search | Warned-ignored | Removed (OT handles it) |
