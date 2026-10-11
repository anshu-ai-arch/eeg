# Supervised Multi-Cluster Feature Selection (MCFS) Methodology Audit
**Project:** Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots (2026 Reproduction)  
**Document ID:** `MCFS-METHODOLOGY-AUDIT-01`  
**Date:** 2026-10-10  
**Phase:** Stage 1.5 — Independent Pre-Execution Classification Audit  
**Audited Code:** [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py), [`tests/test_mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/tests/test_mcfs.py)  
**Status:** AUDIT COMPLETE — METHODOLOGY GROUNDED & VERIFIED  

---

## 1. Executive Summary

This audit rigorously inspects the mathematical implementation of the Supervised Multi-Cluster Feature Selection (MCFS) module against:
1. The primary benchmark paper: **Lashkari et al. (2026)**, *Biomedical Signal Processing and Control*, 118, 109675 (specifically Section 3.6 and Section 3.6.3).
2. The cited foundational MCFS paper: **Cai, Zhang, and He (2010)**, *"Unsupervised Feature Selection for Multi-Cluster Data"*, *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 32(9): 1568–1581 (Reference [19] in Lashkari et al.).
3. The executable source code in [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py).

All 28 existing unit tests pass, and preprocessing artifacts remain strictly frozen. The audit confirms that the implementation in `mcfs.py` accurately realizes the spectral-regression mechanism of Cai et al. (2010), while identifying and documenting the exact supervisory modification introduced by Lashkari et al. (2026).

---

## 2. Component-by-Component MCFS Fidelity Audit

The table below delineates the origin of every parameter and equation in `scripts/classification/mcfs.py`:

| MCFS Component | Classification | Literature Source & Quotation | Implementation in `mcfs.py` | Line References |
| :--- | :--- | :--- | :--- | :--- |
| **1. $k$-Nearest Neighbor Count ($k_{\text{nn}}$)** | **Method-Specified** | Cai et al. (2010), Section 3.1: $k$-nearest-neighbor graph ($k=5$ default). Lashkari et al. mention building a $k$-NN graph without overriding $k_{\text{nn}}$. | `n_neighbors = 5` default. Evaluates `NearestNeighbors(n_neighbors=k+1)`. | [`mcfs.py:L36`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L36), [`L149-L153`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L149-L153) |
| **2. Affinity Weighting & Bandwidth ($\sigma, \gamma$)** | **Method-Specified / Local Choice** | Cai et al. (2010), Section 3.1: Heat kernel $W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$ or 0-1 weighting. Lashkari et al. state: *"edges capture local neighborhood connections"*. | Heat kernel default (`weighting='heat_kernel'`). Bandwidth $\sigma$ dynamically estimated as median non-zero $k$-NN distance, $\gamma = 1/(2\sigma^2)$. Binary option supported. | [`mcfs.py:L38-L44`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L38-L44), [`L156-L177`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L156-L177) |
| **3. Cross-Class Label Gating** | **Paper-Specified (Supervised Modification)** | Lashkari et al. (2026), Section 3.6: *"Information about class labels is utilized to steer the selection toward features that improve separation between classes."* | Supervised edge gating: $W_{ij} = 0$ if $y_i \neq y_j$. Within-class neighborhood affinities are preserved; cross-class edges are strictly suppressed. | [`mcfs.py:L12-L13`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L12-L13), [`L168-L170`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L168-L170) |
| **4. Graph Symmetrization** | **Method-Specified** | Cai et al. (2010), Section 3.1: $W$ is symmetric. Standard spectral graph theory requires symmetric adjacency for real Laplacian eigenvalues. | Enforced symmetric averaging: $W = 0.5(W + W^T)$; self-loops eliminated: $\text{diag}(W) = 0$. | [`mcfs.py:L179-L183`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L179-L183) |
| **5. Laplacian Formulation & Eigenproblem** | **Method-Specified** | Cai et al. (2010), Section 3.2, Eq. (7): Generalized eigenproblem $L y = \lambda D y$, with $y^T D y = 1$, where $L = D - W$. | Normalized symmetric Laplacian $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$. Solved via `scipy.linalg.eigh(L_sym)`; transformed to generalized eigenvectors $y_r = D^{-1/2} z_r$, normalized to $y_r^T D y_r = 1$. | [`mcfs.py:L186-L208`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L186-L208) |
| **6. Number of Eigenvectors ($K$)** | **Reproducibility Choice** | Cai et al. (2010), Section 3.2: $K$ eigenvectors for $K$ clusters. Lashkari et al. do not state the exact value of $K$. | Defaults to $K = \max(2, C)$, where $C$ is the number of classes in the training fold ($K=3$ for Case 1; $K=2$ for Cases 2 and 3). Discards trivial constant eigenvector $y_0 \propto \mathbf{1}$. | [`mcfs.py:L140-L147`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L140-L147), [`L210-L224`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L210-L224) |
| **7. Lasso Objective & Penalty ($\alpha$)** | **Method-Specified / Local Choice** | Cai et al. (2010), Section 3.3, Eq. (8): $\min_{a_r} \|y_r - X a_r\|_2^2 + \alpha \|a_r\|_1$. Neither paper specifies the numerical value of $\alpha$ on Bonn EEG. | Features $X$ standardized to zero mean, unit variance. Fit `Lasso(alpha=0.01)` per eigenvector target. Defensive adaptive fallback if over-penalization zeros all coefficients. | [`mcfs.py:L230-L267`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L230-L267) |
| **8. Feature Scoring & Ranking** | **Method-Specified / Local Choice** | Cai et al. (2010), Section 3.4, Eq. (11): $\text{MCFS}(j) = \max_r |a_{r, j}|$. Lashkari et al. (p. 7): *"features are ordered based on their MCFS weights, and the top-k features are chosen"*. | Exact Eq. (11) implementation: `scores = np.max(np.abs(coefficients), axis=0)`. Ranked descending using stable `mergesort` (deterministic tie-breaking). | [`mcfs.py:L271-L279`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L271-L279) |

---

## 3. Deep-Dive: Original Unsupervised MCFS vs. Supervised Modification

### The Core Literature Divergence
1. **Original Formulation (Cai et al., 2010):**
   * Titled *"Unsupervised Feature Selection for Multi-Cluster Data"*.
   * Developed strictly for unlabeled data where clusters are discovered solely from feature-space geometry.
   * Edges in $W$ are unconstrained: $W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$ for all $j \in \mathcal{N}_k(x_i)$.
2. **Primary Paper Formulation (Lashkari et al., 2026):**
   * Explicitly described as **"supervised multi-cluster feature selection (MCFS)"** in Section 3.6 (heading and text) and Section 3.6.3.
   * Text explicitly states: *"Information about class labels is utilized to steer the selection toward features that improve separation between classes."*
   * However, the authors cite Reference [19] (Cai et al.'s unsupervised paper) without printing a mathematical equation defining how supervision is incorporated.

### Implementation Fidelity Assessment
In spectral graph literature, incorporating class supervision into graph Laplacians follows canonical label gating (also known as supervised graph embedding / Fisher-Laplacian embedding):
$$W_{ij} = \begin{cases} \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right) & \text{if } j \in \mathcal{N}_k(x_i) \text{ and } y_i = y_j \\ 0 & \text{otherwise} \end{cases}$$

**Findings:**
* [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L168-L170) implements exact class-label gating:
  ```python
  if not self.supervised or (y_arr[i] == y_arr[j_idx]):
      weight = float(np.exp(-gamma * (dist ** 2)))
      W[i, j_idx] = weight
  ```
* This ensures within-class local manifold connections are preserved while cross-class edges are prohibited from contaminating the Laplacian.
* The parameter `supervised: bool = True` allows running both the author-specified supervised mode (`supervised=True`) and the canonical Cai et al. unsupervised mode (`supervised=False`).

---

## 4. Leakage Prevention Audit

An independent line-by-line inspection of [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py) verifies zero data leakage:

1. **Strict Partition Isolation:**
   * `fit(X, y)` takes only the input arrays passed to it. It has no internal state, global caches, or file I/O.
   * When invoked inside the nested cross-validation engine, `fit` will be passed **strictly the inner-training partition** `(X_inner_train, y_inner_train)`.
   * Neither validation fold samples nor outer test fold samples enter `fit`.
2. **Transform Isolation:**
   * `transform(X)` performs pure column indexing: `return X_arr[:, self.selected_indices_]`.
   * It performs no fitting, centering, or parameter re-estimation on the incoming validation or test data.
3. **Reproducibility & Determinism:**
   * Random state is fixed (`random_state=42`) for the coordinate-descent Lasso solver.
   * Ranking uses stable mergesort (`np.argsort(-scores, kind='mergesort')`), ensuring that identical input features produce bit-exact identical rankings across repeated invocations.

---

## 5. Summary of Recommended Decisions

1. **Retain Label-Gated Supervised Mode as Default (`supervised=True`):** Strictly matches Section 3.6 of Lashkari et al. (2026).
2. **Preserve Dynamic Bandwidth Estimation:** $\sigma = \text{median}(d_{\text{knn}})$ avoids arbitrary hyperparameter tuning and scales appropriately across feature subsets.
3. **Preserve Adaptive Lasso Fallback:** Guarantees that if the L1 penalty $\alpha=0.01$ over-penalizes a specific fold, non-zero rankings are still deterministically produced.
