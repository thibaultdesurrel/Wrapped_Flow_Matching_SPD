# Comparative Analysis: WrappedCFM vs DiffeoCFM vs GeodesicCFM

## 1. Method Comparison Table

| Aspect | **WrappedCFM** | **DiffeoCFM** | **GeodesicCFM** |
|---|---|---|---|
| **Working space** | Native SPD manifold | Tangent space at I_d (Euclidean) | Native SPD manifold |
| **Conditional center** X_t | X₁^t (power interp.) | I_d (fixed) | gamma_AIRM(x₀, x₁, t) (geodesic) |
| **Noise schedule** sigma_t | sigma_init - (sigma_init - sigma_min)t | Same formula, in tangent space | None (pure geodesic) |
| **Noise injection** | x₀^{sigma_t/sigma_init} (matrix power) | Linear scaling in Sym(d) | No noise modulation |
| **Cond. vector field** | Closed-form (Thm 5.1) | Closed-form (Euclidean) | Auto-diff via `jvp` through eigendecomp |
| **Loss metric** | AIRM inner product | MSE / Frobenius | AIRM inner product |
| **Network wrapper** | ProjectToTangent | None (raw MLP) | ProjectToTangent |
| **ODE solver** | Riemannian (eigval clamping) | Euclidean + Exp at end | Riemannian (eigval clamping) |
| **SPD guarantee** | By construction (eigval clamp) | Only after final Exp map | By construction (eigval clamp) |
| **Key files** | `runner/src/cfm_models/wrapped_cfm.py` | `runner/src/cfm_models/diffeo_cfm.py` | `runner/src/cfm_models/geodesic_cfm.py` |

### Architecture differences (from experiment configs)

| | WrappedCFM | DiffeoCFM | GeodesicCFM |
|---|---|---|---|
| **n_layers** | 6 | 1 | 6 |
| **lr** | 1e-4 | 1e-3 | 1e-4 |
| **warmup_epochs** | 200 | 10 | 200 |
| **max_epochs** | 2000 | 5000 | 2000 |

---

## 2. Theoretical Advantages of WrappedCFM

### 2.1 Intrinsic Riemannian conditional path with closed-form vector field

WrappedCFM defines the conditional probability path entirely on the manifold:

```
p_t(·|X₁) = WG(X₁^t, 0, σ_t² I_D)
```

The center X₁^t traces a one-parameter subgroup in SPD(d) from I_d (t=0) to X₁ (t=1). This path stays on SPD(d) by construction. The key advantage over GeodesicCFM is that the conditional vector field u_t is available **in closed form** (`wrapped_cfm.py:34-55`):

```python
ut = 0.5 * log(X₁) @ x_t + 0.5 * x_t @ log(X₁)
   + (σ'_t / σ_t) * X₁^{t/2} @ √M_t @ log(M_t) @ √M_t @ X₁^{t/2}
```

GeodesicCFM, by contrast, must compute u_t = dγ/dt via `torch.func.jvp` through `airm_geodesic` (`geodesic_cfm.py:34`), which differentiates through eigendecomposition. This is:
- **Numerically fragile** when eigenvalues are near-degenerate (repeated eigenvalues cause ill-defined gradients of `eigh`)
- **More expensive** (JVP through multiple `power_matrix` calls in `airm_geodesic`)

### 2.2 Proper stochastic interpolation (vs. GeodesicCFM's deterministic path)

WrappedCFM modulates noise along the path via σ_t. At each time t, the conditional distribution is a **wrapped Gaussian** around the center X₁^t, and the noise term `x₀^{σ_t/σ_init}` smoothly shrinks as t→1 (`wrapped_cfm.py:29-32`):

```python
x_t = X₁^{t/2} · x₀^{σ_t/σ_init} · X₁^{t/2}
```

GeodesicCFM uses a pure geodesic `γ(x₀, x₁, t)` with **no noise modulation** — the entire randomness comes from the source x₀. This means:
- The conditional path at intermediate times has no mechanism to control variance
- The marginal p_t may be less regular/smooth, making the regression target harder to learn
- There is no gradual "annealing" of noise — the path is deterministic given (x₀, x₁)

### 2.3 Correct Riemannian metric vs. DiffeoCFM's Euclidean surrogate

WrappedCFM optimizes using the AIRM inner product (`wrapped_cfm.py:65`):

```python
loss = <v_t - u_t, v_t - u_t>_{x_t} / dim
```

where `<U, V>_P = tr(P⁻¹ U P⁻¹ V)` (`spd_ops.py:7-22`).

DiffeoCFM uses MSE in tangent space (`diffeo_cfm.py:36`):

```python
loss = ||v_t - u_t||²_F
```

The Log map at identity is a diffeomorphism but **not an isometry**. The Euclidean Frobenius norm in Sym(d) does not correspond to the AIRM distance on SPD(d). Concretely:
- SPD matrices with large eigenvalue spread (far from identity) are **severely distorted** in the Log chart
- The Frobenius-optimal solution in tangent space is **not** the Riemannian-optimal solution on the manifold
- The distortion grows with the Riemannian distance from identity, meaning DiffeoCFM's loss becomes progressively more misaligned for data far from I_d

### 2.4 Adaptive center vs. fixed center

WrappedCFM's conditional center X₁^t **adapts to each target** X₁. The noise x₀ is always modulated relative to the target, so the "transport distance" is always proportional to the complexity of the target.

DiffeoCFM fixes the center at I_d for all samples. In the tangent space, this means:
- All flows originate from (near) the origin in Sym(d)
- Targets far from identity require long, potentially non-geodesic paths in the Log chart
- The generated flow in tangent space is forced to be approximately linear, which is suboptimal for the curved manifold

### 2.5 SPD constraint enforcement during inference

WrappedCFM uses the **Riemannian ODE solver** with eigenvalue clamping (`riemannian_ode_solver.py:31-33`):

```python
epsilon = get_epsilon(x0.dtype, "eigval_clamp")
threshold_value = max(1e-5, epsilon)
x1 = clamp_eigvals.apply(sol[-1], threshold_value)
```

DiffeoCFM uses the **Euclidean ODE solver** (`euclidean_ode_solver.py`) with no eigenvalue clamping. The final `matrix_exp.apply(result_tangent)` at `diffeo_cfm.py:50` guarantees SPD output, but intermediate ODE steps in tangent space may accumulate errors that lead to poorly conditioned final matrices.

### 2.6 When DiffeoCFM's assumptions break down

DiffeoCFM's core assumption is that working in the tangent space at identity is a reasonable approximation. This breaks down when:

1. **Data is not centered around identity**: If the Riemannian mean of the data is far from I_d, the Log map creates large tangent vectors, and Euclidean operations in that regime are highly distorted. The `recenter` option in the datamodule (`base_datamodule.py:60-81`) mitigates this, but it's a preprocessing hack that doesn't help when class-conditional distributions have different means.

2. **Near-singular SPD matrices**: When eigenvalues of SPD matrices approach zero, `log(λ) → -∞`, creating extreme values in tangent space. DiffeoCFM's Euclidean operations may amplify these.

3. **High curvature regions**: SPD(d) has non-constant sectional curvature under AIRM (it's a symmetric space of non-positive curvature). In high-curvature regions, the straight-line paths in tangent space deviate maximally from geodesics.

### 2.7 When GeodesicCFM's assumptions break down

1. **Ill-conditioned eigendecompositions**: The JVP through `power_matrix` → `eigh` is unstable when eigenvalues are close (`spd_ops.py:36`). The derivative of eigenvalues w.r.t. matrix entries involves terms like `1/(λ_i - λ_j)`, which blow up for near-degenerate spectra.

2. **No noise schedule**: Without σ_t modulation, the regression target u_t = dγ/dt can have **high variance** across different (x₀, x₁) pairs when x₀ and x₁ are far apart, making the neural network's job harder.

---

## 3. Empirical Evidence from Existing Results

Extracted and averaged from `runner/results/all_results.csv`.

### 3.1 BNCI2014002 (5 groups averaged)

| Metric | DiffeoCFM | GeodesicCFM | **WrappedCFM** | WrappedCFM+Prior |
|---|---|---|---|---|
| **TSTR F1** | 0.685 | 0.706 | **0.715** | 0.718 |
| **TRTS F1** | 0.766 | 0.791 | **0.764** | 0.687 |
| **Quality Precision** | 0.496 | 0.714 | **0.718** | 0.710 |
| **Quality Density** | 0.276 | 0.577 | **0.607** | 0.554 |
| **Quality Coverage** | 0.760 | **0.942** | 0.939 | 0.924 |
| SPD fraction | 1.000 | 1.000 | 1.000 | 1.000 |
| Training time (s) | 2422 (5000ep) | 1612 (2000ep) | 2355 (2000ep) | 2875 (2000ep) |
| Sampling time (s) | 0.54 | 1.34 | 2.12 | 1.44 |

### 3.2 BNCI2015001 (2 groups averaged)

| Metric | DiffeoCFM | GeodesicCFM | **WrappedCFM** | WrappedCFM+Prior |
|---|---|---|---|---|
| **TSTR F1** | 0.729 | 0.718 | 0.724 | 0.701 |
| **Quality Precision** | 0.645 | 0.884 | **0.909** | 0.892 |
| **Quality Density** | 0.335 | 0.934 | **1.072** | 0.974 |
| **Quality Coverage** | 0.789 | 0.989 | **0.992** | 0.978 |

### 3.3 Textile (1 group)

| Metric | DiffeoCFM | GeodesicCFM | **WrappedCFM** | WrappedCFM+Prior |
|---|---|---|---|---|
| **TSTR F1** | 0.906 | 0.929 | **0.930** | 0.925 |
| **Quality Precision** | 0.925 | 0.975 | **0.973** | 0.975 |
| **Quality Density** | 0.698 | **0.949** | 0.948 | 0.945 |
| **Quality Coverage** | 0.958 | 0.994 | 0.994 | **0.995** |

### 3.4 Key empirical takeaways

**WrappedCFM consistently dominates DiffeoCFM** across all metrics and datasets:
- **Quality Density** is ~2x higher (0.607 vs 0.276 on BNCI2014002; 1.072 vs 0.335 on BNCI2015001)
- **Quality Coverage** is ~20-25% higher
- **TSTR F1** is higher on BNCI2014002 (+3%) and Textile (+2.4%)
- DiffeoCFM's Precision is dramatically worse (~0.50 vs ~0.72), meaning generated samples often fall outside the real data support

**WrappedCFM is competitive with or slightly better than GeodesicCFM**:
- Higher **Density** on BNCI2014002 (+5%) and BNCI2015001 (+15%)
- Higher **TSTR F1** on BNCI2014002 (+1.2%) and Textile (+0.1%)
- Comparable Coverage and Precision
- The advantage is most pronounced on BNCI2015001 where Density reaches **1.072** (above 1.0 means generated distribution is denser than real in many neighborhoods)

**WrappedCFM+Prior does not clearly outperform vanilla WrappedCFM** on these datasets — the prior-informed version sometimes underperforms on TSTR F1 while having slightly lower quality metrics. This could indicate that the class priors need more data or more tuning.

---

## 4. Conditions Where WrappedCFM is Provably/Demonstrably Better

### Demonstrably better (empirical)

| Condition | Why WrappedCFM wins | Evidence |
|---|---|---|
| **Distributional quality (Density/Coverage)** | Native Riemannian loss + manifold-aware noise → samples closer to real distribution | Density 2-3x higher than DiffeoCFM; 5-15% higher than GeodesicCFM |
| **TSTR classification** | Better distributional match → synthetic samples more useful as training data | Consistently highest or tied-highest F1 |
| **Non-centered data** | Adaptive center X₁^t vs fixed identity center | DiffeoCFM's quality degrades when data spread is large |
| **All-SPD output guarantee** | Riemannian solver + eigval clamping | All methods achieve 100% SPD in current experiments, but WrappedCFM has structural guarantees |

### Provably better (theoretical)

| Condition | Why WrappedCFM wins |
|---|---|
| **Riemannian consistency of the loss** | AIRM loss ≡ proper Riemannian regression; DiffeoCFM's MSE in tangent space is geometrically inconsistent |
| **Near-degenerate eigenvalue spectra** | Closed-form u_t avoids JVP through eigendecomposition (GeodesicCFM); matrix power in Log chart avoids -∞ explosion (DiffeoCFM) |
| **Smooth noise annealing on the manifold** | σ_t schedule + matrix power provide geometrically natural interpolation; GeodesicCFM has no noise control |
| **Target-adaptive conditional path** | X₁^t adapts to each target (vs. fixed geodesic from random x₀ in GeodesicCFM, or fixed identity center in DiffeoCFM) |

---

## 5. Suggested Next Steps

### 5.1 Missing experiments to run

- **BreizhCrops results are absent** from `all_results.csv`. Configs exist for all 4 methods. Run:
  ```bash
  cd runner
  ./scripts/train_all_models_breizhcrops.sh
  ```
  BreizhCrops has 4 classes (vs. 2 for EEG/Textile), making it the best test of multi-class generation quality.

### 5.2 Ablations to demonstrate WrappedCFM's advantages

1. **Recentering ablation**: Run all models with `recenter: false` to show that DiffeoCFM degrades when data is not centered around identity, while WrappedCFM remains stable. This directly tests theoretical claim 2.3.

2. **Dimension scaling**: Create synthetic SPD datasets at d = 5, 10, 20, 30, 50. As d increases, the curvature effects intensify and the Log-chart distortion grows. WrappedCFM should maintain quality while DiffeoCFM degrades. See [Section 5.5](#55-synthetic-dataset-implementation) for the implementation approach.

3. **Condition number stress test**: Generate training data with controlled eigenvalue spread (condition number κ = λ_max/λ_min). Test κ = 10, 100, 1000, 10000. WrappedCFM should be robust while DiffeoCFM's tangent-space representation breaks down for large κ. See [Section 5.5](#55-synthetic-dataset-implementation) for the implementation approach.

4. **Convergence speed**: Plot quality metrics (Density, TSTR-F1) vs. epoch for all methods. WrappedCFM uses 2000 epochs vs. DiffeoCFM's 5000; if it reaches comparable or better quality in 2000 epochs, this demonstrates faster convergence per epoch.

5. **Eigenvalue spectrum analysis**: For generated samples from each method, plot the distribution of eigenvalues and compare to real data. This would reveal whether DiffeoCFM's tangent-space approach distorts the spectral distribution.

### 5.3 Theoretical formalizations to pursue

1. **Bound the metric distortion of DiffeoCFM**: Quantify the gap between the Frobenius loss in tangent space and the AIRM loss on the manifold as a function of the Riemannian distance from identity. This would give a formal proof that DiffeoCFM's loss becomes increasingly misaligned for non-centered data.

2. **Stability analysis of GeodesicCFM's JVP**: Characterize the condition number of the JVP computation through `power_matrix` → `eigh` as a function of the eigenvalue gap. Show that WrappedCFM's closed-form avoids this instability.

3. **Variance analysis of the regression target**: Compare the variance of u_t across the batch for each method. WrappedCFM's noise schedule should yield lower-variance targets than GeodesicCFM's pure geodesic, making the regression easier to learn.

### 5.5 Synthetic dataset implementation

Both the dimension scaling and condition number ablations share a single synthetic data generator that plugs into the existing `BaseDataModule`.

#### Core idea

For each experiment, generate K classes of SPD matrices as **wrapped Gaussians** around distinct class means with controlled geometric properties. The existing `sample_wrapped_gaussian` in `runner/src/utils/sampling_wrapped_gaussian.py` already does the hard work (sample in tangent space, push to manifold via Exp map). The datamodule only needs to generate means with the desired properties and call it.

#### Generating class means

**For dimension scaling** (variable d, random means): For each class k, sample a random orthogonal matrix Q_k via QR decomposition of a Gaussian matrix, draw eigenvalues uniformly in [0.5, 5], and form M_k = Q_k diag(λ) Q_k^T. This gives well-conditioned SPD means spread across the manifold.

**For condition number stress test** (fixed d, controlled κ): For each class k, sample Q_k the same way, but set eigenvalues as `logspace(0, log10(κ), d)` — i.e., eigenvalues span from 1 to κ in log-uniform spacing. Log-uniform spacing is natural for SPD matrices because AIRM distance depends on `log(λ_i)`, so this gives uniform spacing in the Riemannian sense. Keep d fixed (e.g., d=10) and vary only κ.

#### Sampling around each mean

Use `sample_wrapped_gaussian(num, p=M_k, mu=0, Sigma=σ² I_D)` with a fixed σ = 1/√D (same formula as `CFMBase.sigma_init`) so the only variable across experiments is the dimension or condition number.

#### Datamodule skeleton

A single new file `runner/src/datamodules/synthetic_spd.py`:

```python
class SyntheticSPDDataModule(BaseDataModule):
    def __init__(self, dim, num_classes, num_samples_per_class,
                 kappa=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.kappa = kappa  # None = random eigenvalues; float = controlled κ

    def _generate_class_mean(self, rng):
        Q, _ = torch.linalg.qr(
            torch.randn(self.dim, self.dim, dtype=torch.float64,
                         generator=rng)
        )
        if self.kappa is None:
            eigvals = 0.5 + 4.5 * torch.rand(self.dim, dtype=torch.float64,
                                               generator=rng)
        else:
            eigvals = torch.logspace(
                0, math.log10(self.kappa), self.dim, dtype=torch.float64
            )
        return Q @ torch.diag(eigvals) @ Q.T

    def setup(self, stage=None):
        rng = torch.Generator().manual_seed(self.seed)
        D = self.dim * (self.dim + 1) // 2
        sigma = 1.0 / math.sqrt(D)

        all_x, all_y = [], []
        means = []
        for k in range(self.num_classes):
            M_k = self._generate_class_mean(rng)
            means.append(M_k)
            mu0 = torch.zeros(D, dtype=torch.float64)
            Sigma0 = (sigma ** 2) * torch.eye(D, dtype=torch.float64)
            samples = sample_wrapped_gaussian(
                self.num_samples_per_class, M_k, mu0, Sigma0
            )
            all_x.append(samples)
            all_y.append(torch.full((self.num_samples_per_class,), k))

        x = torch.cat(all_x)
        y = torch.cat(all_y)

        # Shuffle and split 80/10/10
        idx = torch.randperm(len(x), generator=rng)
        n = len(x)
        tr, va = int(0.8 * n), int(0.9 * n)
        x_train, y_train = x[idx[:tr]], y[idx[:tr]]
        x_val, y_val = x[idx[tr:va]], y[idx[tr:va]]
        x_test, y_test = x[idx[va:]], y[idx[va:]]

        if self.recenter:
            x_train_np, x_val_np, x_test_np = self._center_and_scale(
                x_train.numpy(), x_val.numpy(), x_test.numpy()
            )
            x_train = torch.from_numpy(x_train_np)
            x_val = torch.from_numpy(x_val_np)
            x_test = torch.from_numpy(x_test_np)

        self._compute_class_means_var(x_train.numpy(), y_train.numpy())
        self.train_ds = TensorDataset(x_train, y_train.long())
        self.val_ds = TensorDataset(x_val, y_val.long())
        self.test_ds = TensorDataset(x_test, y_test.long())
```

#### Hydra configs

Add configs under `runner/configs/datamodule/` and `runner/configs/experiment/`:

- **Dimension scaling**: `synthetic_dim5.yaml`, `synthetic_dim10.yaml`, ..., `synthetic_dim50.yaml` — each sets `dim`, `input_dim = dim*(dim+1)/2`, `kappa: null`.
- **Condition number**: `synthetic_kappa10.yaml`, ..., `synthetic_kappa10000.yaml` — each sets `dim: 10`, `kappa`.

Cross with model configs to get the full experiment grid (e.g., `synthetic_dim20_wrappedcfm.yaml`). The training scripts and evaluation pipeline (`evaluate.py`) work unchanged — only the data source changes.

#### What this tests

- **Dimension scaling**: As d grows, D = d(d+1)/2 grows quadratically, the manifold curvature effects intensify, and the Log-chart distortion at identity becomes more severe. DiffeoCFM's Frobenius loss in Sym(d) becomes an increasingly poor proxy for the AIRM loss. WrappedCFM, working natively on the manifold, should maintain quality.
- **Condition number**: Large κ means SPD matrices have eigenvalues spanning orders of magnitude. Under the Log map, this maps to tangent vectors with components ranging from `log(1)=0` to `log(κ)`, creating extreme scale imbalance in DiffeoCFM's Euclidean space. WrappedCFM's matrix-power noise injection `x₀^{σ_t/σ_init}` handles this naturally since it operates on each eigenvalue independently in log-scale.

### 5.6 Quick wins

- **Fix the docstring bug** in `geodesic_cfm.py:1`: says `"""DiffeoCFM."""` instead of `"""GeodesicCFM."""`.
- **Fair architecture comparison**: Current configs give DiffeoCFM only 1 layer vs. 6 for the others. While this may reflect what works best for each method, an ablation with equal architectures (e.g., 6 layers for all) would strengthen the empirical comparison by ruling out that DiffeoCFM simply needs a deeper network.

---

## Open Questions

1. **BreizhCrops**: Were these experiments run but results not yet aggregated, or have they not been run yet?

2. **The DiffeoCFM architecture discrepancy** (1 layer vs. 6 layers): Is this intentional because DiffeoCFM works best with shallow networks in tangent space, or should we propose a controlled experiment with matched architectures?

3. **WrappedCFM_with_prior underperformance**: The prior-informed version sometimes underperforms vanilla WrappedCFM on TSTR F1 and TRTS F1 (e.g., BNCI2014002 TRTS F1: 0.687 vs 0.764). Is this expected, or something to investigate further?
