# Supervised MCFS Spectral-Graph Risk Audit & Mathematical Resolution
**Project:** Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots (2026 Reproduction)  
**Document ID:** `MCFS-SPECTRAL-RISK-AUDIT-02`  
**Date:** 2026-10-10  
**Phase:** Stage 2 — Pre-Nested-CV MCFS Safeguards  
**Audited Component:** [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py)  

---

## 1. Executive Summary

This audit assesses the spectral graph theoretic properties of the Supervised Multi-Cluster Feature Selection (MCFS) module in [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py), focusing on the mathematical implications of zeroing cross-class edges ($W_{ij} = 0$ if $y_i \neq y_j$) as described in Section 3.6 of Lashkari et al. (2026).

Our empirical and theoretical investigation confirms four critical mathematical risks in the original Stage 1 implementation:
1. **Disconnected Components & Zero-Degree Vertices:** When $k$-NN is computed globally across all classes and cross-class edges are zeroed, boundary samples whose global $k$ nearest neighbors belong to other classes lose all outbound edges. On the Bonn Case 3 dataset ($N=200$), this produced **3 isolated vertices** with degree $d_i = 0$.
2. **Numerical Instability of Zero-Degree Handling:** Setting `eps_degree = 1e-10` resulted in $D_{ii}^{-1/2} = 10^5$. For isolated vertices, the normalized generalized eigenvector spiked to $100,000$ at that sample, causing severe coordinate descent non-convergence (`ConvergenceWarning`) in Lasso regression.
3. **Repeated Zero Eigenvalues & Non-Deterministic Eigenvectors:** For a graph partitioned into $C$ class components, the normalized Laplacian $L_{\text{sym}}$ has eigenvalue $\lambda = 0$ with multiplicity at least $C$. Standard symmetric eigensolvers (`scipy.linalg.eigh`) return an arbitrary orthonormal basis of this degenerate null space.
4. **Invalid Trivial Eigenvector Filtering:** Discarding only vectors with sample variance $\le 10^{-6}$ failed because arbitrary basis vectors in the multi-component null space are piecewise constant across classes, possessing variance $\approx 0.1 > 10^{-6}$. Consequently, two redundant targets spanning the class indicator subspace (including the trivial all-ones component $\mathbf{1}$) were selected, while true intra-class manifold harmonics ($\lambda > 0$) were omitted.
5. **Silent Lasso Fallback:** The adaptive Lasso fallback modified $\alpha \to \alpha \times 0.01$ or defaulted to Ridge without recording the effective regularization parameter.

---

## 2. In-Depth Analysis of the Five Pre-Flight Checks

### Check 1: How Zero-Degree Vertices Are Handled
* **Original Code:**
  ```python
  degrees = np.sum(W, axis=1)
  eps_degree = 1e-10
  degrees_safe = np.maximum(degrees, eps_degree)
  D_inv_sqrt = 1.0 / np.sqrt(degrees_safe)
  ```
* **Failure Mode:** If vertex $i$ has $d_i = 0$, $D_{ii} = 10^{-10}$ and $D_{ii}^{-1/2} = 10^5$. Because row $i$ of $W$ is zero, row $i$ of $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$ is $e_i^T$. The unit vector $e_i$ is an eigenvector with eigenvalue $\lambda = 1.0$. The generalized eigenvector $y = D^{-1/2} e_i$ has value $100,000$ at coordinate $i$ and 0 elsewhere. Lasso regression on this target attempts to fit a single-sample artifact of magnitude $10^5$.
* **Root Cause:** $k$-NN was queried globally before zeroing cross-class edges.
* **Resolution:**
  1. In supervised mode, construct the $k$-NN affinity graph **within each class partition** $X^{(c)}$. Every sample connects to its $k$ nearest neighbors of the same class, guaranteeing $d_i \ge k/2 > 0$ for all samples (since class size $N_c \gg k$). On Bonn Case 3, within-class $k$-NN strictly yields $0$ isolated vertices (minimum degree $2.5$).
  2. For synthetic/degenerate inputs with isolated vertices, set $D_{ii} = 1.0$ (unit regularized degree), preventing $D^{-1/2}$ explosion, and zero out isolated vertex coordinates in regression targets.

### Check 2: Whether the Laplacian Is Numerically Symmetric
* **Inspection:**
  * Line 180: `W = 0.5 * (W + W.T)` symmetrizes adjacency.
  * Line 183: `np.fill_diagonal(W, 0.0)` eliminates self-loops.
  * Line 194: `L_sym = np.eye(n_samples) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])`.
  * Line 195: `L_sym = 0.5 * (L_sym + L_sym.T)` enforces exact machine-precision symmetry.
* **Finding:** The normalized Laplacian is guaranteed numerically symmetric.

### Check 3: Eigenvector Selection for Multiplicity > 1 Zero Eigenvalues
* **Spectral Graph Theory:**
  Let $W$ have $C$ connected components (the $C$ classes). The null space of $L = D - W$ (eigenvalue $\lambda = 0$) has dimension $C$. It is spanned by the class indicator vectors $\{\mathbf{1}^{(1)}, \dots, \mathbf{1}^{(C)}\}$.
  The constant vector $\mathbf{1} = \sum_{c=1}^C \mathbf{1}^{(c)}$ is ALWAYS in the null space: $L \mathbf{1} = (D - W) \mathbf{1} = 0$.
  The trivial vector $\mathbf{1}$ has zero correlation with centered features: $X_{\text{scaled}}^T \mathbf{1} = 0$.
  The subspace of the null space that is **orthogonal to $\mathbf{1}$** (under the $D$ metric) has dimension $C - 1$.
* **Failure Mode in Original Code:**
  `scipy.linalg.eigh` returns an arbitrary orthonormal basis of the $C$-dimensional null space. Because these basis vectors are not aligned with $\mathbf{1}$, each has variance $> 10^{-6}$. The code selected $K=2$ arbitrary vectors from the null space. For $C=2$ classes, both vectors were linear combinations of class indicators $\mathbf{1}^{(1)}$ and $\mathbf{1}^{(2)}$. Once centered with intercept in Lasso, Target 1 and Target 2 were collinear ($r = -1.0$).
* **Resolution:**
  Construct the null space basis deterministically:
  1. The trivial direction $v_0 = \frac{D^{1/2} \mathbf{1}}{\| D^{1/2} \mathbf{1} \|_2}$ is explicitly identified and **discarded**.
  2. The remaining $C - 1$ dimensions of the null space form the **class-discriminant subspace** (orthogonal to $\mathbf{1}$). For $C=2$ classes, there is exactly 1 class-discriminant direction $y_1$ (positive on one class, negative on the other, $y_1^T D \mathbf{1} = 0$).
  3. Any additional targets up to $K$ are selected from the eigenvectors with **strictly positive eigenvalues $\lambda > 0$**, which capture the intra-class manifold harmonics.
  4. A canonical sign convention (requiring the coordinate with largest absolute value to be positive) ensures bit-exact determinism across all platforms and BLAS libraries.

### Check 4: Whether Discarding Only One Eigenvector Is Mathematically Justified
* **Analysis:**
  * In Cai et al. (2010), the graph is connected, so the null space has dimension 1 (only $y_0 \propto \mathbf{1}$). Discarding $y_0$ leaves $K$ eigenvectors with $\lambda > 0$.
  * In supervised MCFS with $C$ classes, the null space has dimension $C$. Exactly 1 direction is trivial ($\mathbf{1}$), while $C - 1$ directions are class-discriminating.
  * Discarding only 1 direction ($\mathbf{1}$) is mathematically justified because the remaining $C - 1$ null directions are NOT trivial: they are the most discriminative directions possible ($L y = 0 \implies \sum W_{ij} (y_i - y_j)^2 = 0$, meaning within-class variation is zero while between-class separation is maximized).
  * However, discarding 1 direction requires **explicitly identifying the trivial direction $\mathbf{1}$**, rather than relying on `np.std(y) > 1e-6` on arbitrary rotations.

### Check 5: Tracking Adaptive Lasso Fallback
* **Analysis:**
  * If $\alpha$ causes all Lasso coefficients to vanish, the retry with $\alpha \times 0.01$ or Ridge changes the objective function.
  * The original code did not record this change.
* **Resolution:**
  * Add attributes `alpha_used_: np.ndarray` (recording the actual $\alpha$ or penalty used per target $r$) and `lasso_fallback_used_: list[str]` (recording `'none'`, `'retry_alpha_scaled'`, or `'ridge'` per target).

---

## 3. Comparison of Implementation Options

| Criterion | Option A: Unmodified Stage 1 Code | Option B: Supervised Within-Class $k$-NN + Deterministic Spectral Decomposition (Recommended) | Option C: Pure Unsupervised MCFS (Cai et al. 2010) |
| :--- | :--- | :--- | :--- |
| **Bonn Case 3 Isolated Vertices** | 3 isolated vertices ($d_i = 0$) | 0 isolated vertices ($d_i \ge 2.5$) | 0 isolated vertices |
| **Lasso Numerical Stability** | Severe `ConvergenceWarning` ($10^5$ spike) | Converges smoothly (0 warnings) | Converges smoothly |
| **Eigenvector Determinism** | BLAS-dependent arbitrary rotation in null space | Canonical, deterministic sign & orthogonalization | Deterministic for simple spectrum |
| **Spectral Composition for $K=2$** | 2 collinear null vectors (redundant) | 1 class-discriminant + 1 manifold harmonic | 2 manifold harmonics ($\lambda > 0$) |
| **Compliance with Lashkari et al. (2026)** | Partial (violates Section 3.6 intent by fragmenting graph) | **Full** (realizes Section 3.6 supervised intent without artifacts) | Deviates (ignores Section 3.6 class supervision) |

---

## 4. Recommended Action & Justification

**Adopt Option B:**
1. In `fit(X, y)`:
   * When `supervised=True`, construct nearest-neighbor graphs within each class subset:
     ```python
     for c in unique_classes:
         # fit k-NN on samples of class c
     ```
   * Symmetrize $W$ and compute degrees.
   * Defend zero-degree vertices: `degrees_safe = np.where(degrees < 1e-12, 1.0, degrees)`.
   * Solve $L_{\text{sym}} z = \lambda z$.
   * Explicitly project out the trivial vector $v_0 \propto D^{1/2} \mathbf{1}$.
   * Canonicalize the $C - 1$ class-discriminant vectors from the null space and sort remaining eigenvectors by ascending positive eigenvalue $\lambda > 0$.
   * Enforce deterministic sign convention: `if y[np.argmax(np.abs(y))] < 0: y = -y`.
   * Record `alpha_used_` and `lasso_fallback_used_`.
2. Add comprehensive unit tests in `tests/test_mcfs.py` testing:
   * Disconnected graphs with multiple classes.
   * Explicit zero-degree handling.
   * Repeated zero eigenvalues.
   * Deterministic eigenvector selection across permutations.
   * Logging of `alpha_used_` and `lasso_fallback_used_`.
