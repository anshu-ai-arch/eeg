# Classification Implementation Decisions & Literature Traceability Report
**Project:** Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots (2026 Reproduction)  
**Document ID:** `CLASSIFICATION-IMPLEMENTATION-DECISIONS-01`  
**Date:** 2026-10-10  
**Phase:** Classification Phase — Stage 1 (MCFS Implementation & Pre-Training Audit)  
**Status:** Component Implemented and Verified (Zero Models Trained)  

---

## 1. Executive Summary

This report documents the mathematical and architectural decisions for the classification phase of the Bonn EEG paper reproduction. In accordance with the Stage 1 constraints, the Supervised Multi-Cluster Feature Selection (MCFS) module has been implemented in [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py) and thoroughly verified through 8 mathematical unit tests in [`tests/test_mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/tests/test_mcfs.py).

All preprocessing artifacts ([`bonn_features_96.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/batch_preprocessing/bonn_features_96.csv), [`batch_diagnostics.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/batch_preprocessing/batch_diagnostics.csv), [`run_summary.json`](file:///Users/anshutiwari123/phase_sync_eeg/audit/batch_preprocessing/run_summary.json)) remain **frozen, read-only, and completely unmodified**. Zero SVM models have been trained, and zero cross-validation experiments have been executed on the Bonn dataset.

---

## 2. Literature Grounding & Specification Taxonomy

To maintain total scientific transparency, every parameter and algorithmic step is categorized into one of four tiers:
1. **Paper-Specified:** Explicitly defined in text, tables, or equations of Lashkari et al. (2026).
2. **Method-Specified:** Explicitly defined in the foundational MCFS paper (Cai, Zhang, He, TPAMI 2010 [19]) cited by the authors.
3. **Reproducibility Implementation Choice:** Necessary engineering decision where literature left parameters unspecified.
4. **Unresolved Ambiguity:** Open questions documented for user alignment.

---

## 3. Component-by-Component Decision Matrix

### A. Classification Benchmark Cases

| Setting / Decision | Tier | Literature Evidence / Formulation | Value Adopted |
| :--- | :--- | :--- | :--- |
| **Case 1 (3-Class)** | Paper-Specified | Section 3.6.2, Table 2 (p. 7): *"Case 1: Set A, B vs Set C, D vs Set E"* | A+B (`healthy`, $n=200$) vs C+D (`interictal`, $n=200$) vs E (`ictal`, $n=100$). Total $N=500$. |
| **Case 2 (Binary Imbalanced)** | Paper-Specified | Section 3.6.2, Table 2 (p. 7): *"Case 2: Set A, B, C, D vs Set E"* | A+B+C+D (`non_ictal`, $n=400$) vs E (`ictal`, $n=100$). Total $N=500$. |
| **Case 3 (Binary Balanced)** | Paper-Specified | Section 3.6.2, Table 2 (p. 7): *"Case 3: Set D vs Set E"* | D (`interictal`, $n=100$) vs E (`ictal`, $n=100$). Total $N=200$. (A, B, C excluded). |
| **Multiclass Strategy** | Paper-Specified | Section 3.6.1 (p. 7): *"Multi-class classification uses a one-vs-one voting strategy."* | One-vs-One (OvO) decision function shape in `sklearn.svm.SVC`. |

---

### B. Supervised Multi-Cluster Feature Selection (MCFS)

| Algorithmic Step | Tier | Literature Evidence / Mathematical Basis | Implemented Choice |
| :--- | :--- | :--- | :--- |
| **Feature Pool** | Paper-Specified | Page 7: *"There are 12 features for each of the 8 GLCMs and a total of 96 features."* | Input $X \in \mathbb{R}^{N \times 96}$ from `bonn_features_96.csv`. |
| **Candidate Feature Counts ($k$)** | Paper-Specified | Section 3.6.3 (p. 7): *"candidates k = 5, 10, 15; default 10"* | Evaluates $k \in \{5, 10, 15\}$ inside the inner CV loop. |
| **Affinity Graph Neighborhood** | Method-Specified | Cai et al. (2010), Section 3.1: $k$-nearest-neighbor graph ($k_{\text{nn}}=5$ default). | $k_{\text{nn}} = 5$ Euclidean nearest neighbors computed strictly on training fold. |
| **Class Label Gating** | Paper-Specified | Section 3.6 (p. 7): *"Information about class labels is utilized to steer the selection toward features that improve separation between classes."* | Supervised gating: $W_{ij} = 0$ if $y_i \neq y_j$. Edges only exist between samples sharing the same class label. |
| **Affinity Edge Weighting** | Method-Specified | Cai et al. (2010), Section 3.1: Heat kernel $W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$. | Heat kernel with $\sigma$ dynamically estimated as median $k$-NN distance; fallback to 0-1 binary weighting. |
| **Graph Symmetrization** | Method-Specified | Standard spectral graph theory (Chung, 1997; Cai et al., 2010): $W = \frac{1}{2}(W + W^T)$. | Enforced numerical symmetry with zero diagonal: $W = 0.5(W + W^T)$, $\text{diag}(W) = 0$. |
| **Laplacian Eigenproblem** | Method-Specified | Cai et al. (2010), Section 3.2, Eq. (7): Generalized eigenproblem $L y = \lambda D y$, where $y^T D y = 1$. | Solved via symmetric normalized Laplacian $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$; $y_r = D^{-1/2} z_r$, normalized to $y_r^T D y_r = 1$. |
| **Eigenvector Count ($K$)** | Reproducibility Choice | Cai et al. (2010), Section 3.2: $K$ eigenvectors for $K$ clusters. | Defaults to number of unique classes $C$ in the training fold ($K=3$ for Case 1, $K=2$ for Cases 2 and 3). |
| **Lasso Regression Formulation** | Method-Specified | Cai et al. (2010), Section 3.3, Eq. (8): $\min_{a_r} \|y_r - X a_r\|_2^2 + \alpha \|a_r\|_1$. | `sklearn.linear_model.Lasso` with $\alpha=0.01$, fitted on standardized features with intercept. |
| **Score Aggregation** | Method-Specified | Cai et al. (2010), Section 3.4, Eq. (11): $\text{MCFS}(j) = \max_r |a_{r, j}|$. | Maximum absolute regression coefficient across the $K$ spectral targets: $\text{score}(j) = \max_r |a_{r, j}|$. |
| **Ranking & Tie-Breaking** | Reproducibility Choice | Literature does not specify tie-breaking rule. | Stable sort (`mergesort`) in descending order; ties preserve canonical feature sequence. |

---

### C. Support Vector Machine (SVM-RBF) & Hyperparameter Grid

| Parameter | Tier | Literature Evidence / Mathematical Basis | Implemented Choice |
| :--- | :--- | :--- | :--- |
| **Classifier Model** | Paper-Specified | Section 3.6.1 (p. 7): *"Classification is carried out using an SVM with Radial Basis Function (RBF) kernel: $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ (Eq. 21)"* | `sklearn.svm.SVC(kernel='rbf', decision_function_shape='ovo')`. |
| **Hyperparameter Grid** | Paper-Specified | Section 3.6.1, Eq. (22) (p. 7):<br>$C \in \{10^{-3}, 10^{-2}, \dots, 10^3\}$<br>$\gamma \in \{10^{-3}, 10^{-2}, \dots, 10^3\}$ | **Base-10 logarithmic grid:**<br>$C \in \{10^{-3}, 10^{-2}, 10^{-1}, 10^0, 10^1, 10^2, 10^3\}$ (7 values)<br>$\gamma \in \{10^{-3}, 10^{-2}, 10^{-1}, 10^0, 10^1, 10^2, 10^3\}$ (7 values)<br>Total $7 \times 7 = 49$ grid points. |
| **Discrepancy Resolution** | Resolved | Pre-audit reports mentioned base-2 ranges ($2^{-5} \dots 2^{15}$). The primary paper's Eq. (22) explicitly specifies base-10 powers. Base-2 is rejected. | Base-10 powers adopted strictly. |

---

### D. Nested Cross-Validation Architecture

| Component | Tier | Literature Evidence / Mathematical Basis | Implemented Choice |
| :--- | :--- | :--- | :--- |
| **Outer Cross-Validation** | Paper-Specified | Section 3.6.1 & 3.6.3: 10-fold stratified outer CV. | `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`. |
| **Inner Cross-Validation** | Paper-Specified | Section 3.6.1 & 3.6.3: 10-fold stratified inner CV inside each outer training fold ($10 \times 10$). | `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`. |
| **Feature Standardization** | Method-Specified | Standard ML practice; preventing scale bias in RBF kernel and Lasso. | `StandardScaler` fitted strictly on training folds; applied to validation/test folds. |
| **Inner Selection Criterion** | Paper-Specified | Section 3.6.1 & 3.6.3: *"optimising mean F1 (tie-break: Accuracy)"*. | Mean macro-F1 across 10 inner validation folds; tie-break: mean Accuracy. |
| **Outer Refitting** | Paper-Specified | Section 3.6.3: *"refit the model on the entire outer training fold using the selected features and best hyperparameters, then test on the held-out outer fold."* | Outer scaler, MCFS, and SVM are retrained on 100% of outer training data using optimal $(k^*, C^*, \gamma^*)$. |
| **Outer Evaluation** | Paper-Specified | Section 3.6.4: Outer predictions tested exactly once per fold. | Predictions recorded for each held-out test fold. |
| **Statistical Inference** | Paper-Specified | Section 3.7: Stratified bootstrap ($B=2,000$ resamples) on pooled outer predictions for 95% CIs. | Macro Accuracy, Precision, Recall, F1; 95% bootstrap CIs; McNemar's test. |

---

## 4. Unresolved Ambiguities & Recommended Policies

1. **Lasso Penalty Parameter $\alpha$ in MCFS:**
   - *Status:* Neither Lashkari et al. (2026) nor Cai et al. (2010) report a single constant value of $\alpha$ for the Bonn dataset.
   - *Adopted Policy:* $\alpha = 0.01$ (standard sparse regularization for normalized features). If over-penalization yields all zero coefficients, the implementation dynamically reduces $\alpha$ to guarantee a non-trivial ranking.
2. **Eigenvector Subspace Dimension $K$ in MCFS:**
   - *Status:* For binary classification ($C=2$), literature varies between $K=1$ (the single non-trivial eigenvector separating the 2 classes) and $K=2$.
   - *Adopted Policy:* $K = \max(2, C)$, setting $K=3$ for Case 1 (3-class) and $K=2$ for Cases 2 and 3. This ensures at least 2 spectral projections are evaluated across all cases.
3. **Candidate $k$ Optimization vs Fixed $k=10$:**
   - *Status:* Section 3.6 mentions *"top-k features (where k = 10, in the experiments)"*, while Section 3.6.3 mentions *"candidates k = 5, 10, 15; default 10 chosen by inner-CV performance"*.
   - *Adopted Policy:* In the nested CV design, candidate $k \in \{5, 10, 15\}$ are tuned jointly with $(C, \gamma)$ on the inner CV loop, directly implementing Section 3.6.3.

---

## 5. Artifact Protection & Freeze Confirmation

- The 500-recording preprocessing dataset ([`audit/batch_preprocessing/bonn_features_96.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/batch_preprocessing/bonn_features_96.csv)) has been verified and remains completely untouched.
- No files in `data/bonn_eeg/` or `audit/batch_preprocessing/` were modified during this stage.
- All code additions are strictly isolated in `scripts/classification/` and `tests/`.
