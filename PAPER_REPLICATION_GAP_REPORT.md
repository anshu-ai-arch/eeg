# Paper Replication Gap Report: Lashkari et al. (2026)

**Target Paper:** Saleh Lashkari et al. (2026), *“Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots,”* Biomedical Signal Processing and Control, 118, 109675.  
**Audited Codebase:** `phase_sync_eeg`  
**Date:** 2026-10-10  
**Scope:** Forensic comparison of existing implementation and saved artifacts against the complete paper specification.

---

## 1. Already Implemented and Matching the Paper

The core proposed GRP–GLCM feature extraction and nested cross-validation classification pipeline is fully implemented and operational:

| Component | Paper Specification | Implemented Code Location | Exact Functions & Lines |
| :--- | :--- | :--- | :--- |
| **Z-score Normalization** | Section 3.1: Zero-mean, unit-variance signal normalization. | [`scripts/preprocessing/normalization.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/normalization.py) | `zscore_normalize()` [L14–L52] |
| **AMI Time Delay ($\tau$)** | Section 3.2: First local minimum of Average Mutual Information. | [`scripts/preprocessing/embedding.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/embedding.py) | `compute_ami()` [L13–L99] |
| **FNN Dimension ($m$)** | Section 3.2: False Nearest Neighbours stopping when ratio $< 1\%$. | [`scripts/preprocessing/embedding.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/embedding.py) | `compute_fnn()` [L102–L223] |
| **Takens Embedding** | Section 3.2, Eq. (1): $X(t) = [x(t), x(t+\tau), \dots, x(t+(m-1)\tau)]$. | [`scripts/preprocessing/grp.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/grp.py) | `reconstruct_phase_space()` [L14–L43] |
| **Continuous GRP Matrix** | Section 3.3, Eq. (2): Pairwise Euclidean distances $\|X(i) - X(j)\|$. | [`scripts/preprocessing/grp.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/grp.py) | `compute_grp()` [L45–L66] |
| **8 GLCM Configurations** | Section 3.5: 2 angles ($\theta \in \{0^\circ, 45^\circ\}$) $\times$ 4 offsets ($d \in \{1, 2, 3, 4\}$). | [`scripts/preprocessing/glcm.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/glcm.py) | `compute_single_glcm()` [L30–L95] |
| **12 Haralick Features (96 total)** | Section 3.3.4–3.3.15, Eqs. (6)–(19): 12 features across 8 GLCMs = 96 features. | [`scripts/preprocessing/glcm.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/glcm.py) | `extract_haralick_12()` [L97–L223] |
| **Frozen Batch Features** | Section 3.6.2: 500 recordings across subsets A–E processed into feature table. | [`audit/batch_preprocessing/bonn_features_96.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/batch_preprocessing/bonn_features_96.csv) | Shape: $500 \times 103$ (96 features + 7 metadata) |
| **MCFS Feature Selection** | Section 3.6 & [19]: Spectral Laplacian eigenproblem + Lasso regressions. | [`scripts/classification/mcfs.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py) | `MCFS.fit()` [L96–L381], `MCFS.transform()` [L383–L409] |
| **RBF-SVM Classifier** | Section 3.6.1, Eq. (21): Radial Basis Function $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$. | [`scripts/classification/nested_cv.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/nested_cv.py) | `SVC(kernel='rbf', ...)` [L217–L224, L383–L390] |
| **Base-10 Hyperparameter Grid** | Section 3.6.1, Eq. (22): $C, \gamma \in \{10^{-3}, 10^{-2}, \dots, 10^3\}$ (49 pairs). | [`scripts/classification/nested_cv.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/nested_cv.py) | `CANDIDATE_C`, `CANDIDATE_GAMMA` [L60–L61] |
| **Multiclass OvO Strategy** | Section 3.6.1: One-vs-One pairwise voting for multi-class Case 1. | [`scripts/classification/nested_cv.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/nested_cv.py) | `decision_function_shape='ovo'` [L221, L387] |
| **Nested 10×10 CV Architecture** | Section 3.6.1 & 3.6.3: Stratified 10-fold outer $\times$ Stratified 10-fold inner CV. | [`scripts/classification/nested_cv.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/nested_cv.py) | `tune_inner_cv()` [L144–L276], `run_nested_cross_validation()` [L278–L481] |
| **Inner Tuning Criterion** | Section 3.6.1: Primary inner objective maximizing macro-F1 (tie: accuracy). | [`scripts/classification/nested_cv.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/nested_cv.py) | `tiebreak_key()` [L256–L263] |
| **Case 1 Runner** | Section 3.6.2, Table 2: A+B vs C+D vs E ($N=500$, 3-class). | [`scripts/classification/run_cases12.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/run_cases12.py) | `run_case()` [L53–L169], `main()` [L192–L211] |
| **Case 2 Runner** | Section 3.6.2, Table 2: A+B+C+D vs E ($N=500$, binary, E positive). | [`scripts/classification/run_cases12.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/run_cases12.py) | `run_case()` [L53–L169], `main()` [L213–L231] |
| **Case 3 Runner** | Section 3.6.2, Table 2: D vs E ($N=200$, binary, E positive). | [`scripts/classification/run_case3_pilot.py`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/run_case3_pilot.py) | `main()` [L73–L232] |

---

## 2. Implemented but Different from the Paper

The following components are implemented in code but deviate from the literal text or equations of the paper:

1. **Quantization Formula Scaling ($K-1$ vs $K$):**
   * *Paper Procedure:* Eq. (20) specifies $\text{RP}_{\text{quantized}} = \text{round}\left( \frac{\text{RP}}{\max(\text{RP})} \cdot K \right)$. Multiplying by $K=16$ and rounding produces values in $\{0, 1, \dots, K\}$, resulting in $K+1 = 17$ discrete levels.
   * *Code Difference:* [`scripts/preprocessing/grp.py:L72–L79, L103`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/grp.py#L72-L79) multiplies by $(K - 1) = 15$, producing exactly 16 discrete levels ($\{0, 1, \dots, 15\}$) to construct a standard $16 \times 16$ GLCM.

2. **FNN Dual Criterion vs Criterion 1 Only:**
   * *Paper Procedure:* Section 3.2 cites Kennel et al. (1992), which defines two criteria: relative distance ratio increase ($R_{\text{tol}}$) and attractor size relative to standard deviation ($A_{\text{tol}}$).
   * *Code Difference:* [`scripts/preprocessing/embedding.py:L116–L121, L187–L190`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/embedding.py#L116-L121) applies only Criterion 1 ($R_{\text{tol}}=15.0$). Parameter `a_tol` is explicitly unused to prevent false high-dimensional inflation on short noisy EEG segments.

3. **GLCM Symmetrization:**
   * *Paper Procedure:* Section 3.3.2 & 3.5 describe counting co-occurrences at angles $0^\circ, 45^\circ$, but omit whether symmetric pairing $P(i, j) + P(j, i)$ is computed.
   * *Code Difference:* [`scripts/preprocessing/glcm.py:L82–L83`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/preprocessing/glcm.py#L82-L83) enforces matrix symmetrization ($P = P + P^T$) prior to probability normalization.

4. **MCFS Graph Supervision Mechanism:**
   * *Paper Procedure:* Section 3.6 states qualitatively: *"Information about class labels is utilized to steer the selection toward features that improve separation between classes."* No equations are provided.
   * *Code Difference:* [`scripts/classification/mcfs.py:L155–L191`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L155-L191) implements supervision by zeroing all cross-class edges ($W_{ij} = 0$ if $y_i \neq y_j$). Because this disconnects the graph into $C$ components, [`mcfs.py:L251–L281`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/mcfs.py#L251-L281) adds a QR-orthogonalization step on the multi-component Laplacian nullspace to obtain non-trivial class-discriminant targets.

5. **Inner-CV Feature Subset Optimization ($k=10$ vs $\{5, 10, 15\}$):**
   * *Paper Procedure:* The paper contains an internal textual contradiction: Section 3.6 (p. 7) states *"top-k features (where k = 10, in the experiments) are chosen"*, whereas Section 3.6.3 (p. 7) states *"candidates k = 5, 10, 15; default 10 chosen by inner-CV performance"*.
   * *Code Difference:* [`scripts/classification/nested_cv.py:L210–L246, L256–L263`](file:///Users/anshutiwari123/phase_sync_eeg/scripts/classification/nested_cv.py#L210-L246) tunes across $\{5, 10, 15\}$ in the inner CV loop, resolving ties by preferring $k=10$.

---

## 3. Missing Paper Components

The following experimental components described in the paper are entirely absent from the codebase:

1. **12-Feature RQA Baseline Comparison (Section 3.4 & Tables 7, 8, 9):**
   * *Paper Description:* The primary paper benchmarks the proposed GRP–GLCM method against classical Recurrence Quantification Analysis (RQA) using 12 features (Recurrence Rate, Determinism, Laminarity, Trapping Time, Length of longest diagonal line, Entropy of diagonal lines, etc.) evaluated under the exact same nested 10×10 CV protocol.
   * *Implementation Status:* **MISSING**. No RQA feature extraction module or baseline comparison scripts exist.

2. **Statistical Significance Testing (Section 3.7):**
   * *Paper Description:* Section 3.7 specifies:
     - Stratified Bootstrap 95% Confidence Intervals with $B = 2,000$ resamples on pooled outer predictions.
     - Paired McNemar tests with Holm-Bonferroni correction between GRP-GLCM and baseline models.
     - Wilcoxon signed-rank tests across outer-fold metric distributions.
   * *Implementation Status:* **MISSING**. Outer predictions and fold metrics are saved, but no statistical inference scripts exist.

3. **Benchmark Chaotic Attractor Experiments (Section 2 & Section 4):**
   * *Paper Description:* The paper evaluates GRP and texture analysis on 6 continuous/discrete non-linear dynamical systems:
     1. Lorenz attractor
     2. Rössler attractor
     3. Colpitts oscillator
     4. Chua circuit
     5. Autonomous 4D hyperchaotic system
     6. RC Nonlinear oscillator
   * *Implementation Status:* **MISSING**. No chaotic ODE generators, simulation scripts, or benchmark data files exist.

4. **Gaussian-Noise / SNR Robustness Experiments (Section 4, Figures & Tables):**
   * *Paper Description:* Evaluates classification degradation under additive white Gaussian noise across varying SNRs (e.g., 0 dB to 30 dB) to demonstrate noise robustness of GRP–GLCM relative to RQA.
   * *Implementation Status:* **MISSING**. No noise injection routines or SNR sweep runners exist.

---

## 4. Results Not Matching the Paper (Comparison Against Tables 7, 8, and 9)

A direct numerical comparison between our saved leak-free nested CV results and the paper's reported values:

### Case 1: A+B (healthy) vs C+D (interictal) vs E (ictal) — 3-class ($N=500$)

| Metric | Paper Table 7 Reported Value | Saved Result (`case1_run_summary.json`) | Recomputed Value | Discrepancy ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **99.40%** (or 99.6%) | **93.80%** (469 / 500) | **93.80%** | **-5.60%** |
| **Macro-F1** | **99.40%** | **93.29%** | **93.29%** | **-6.11%** |
| **Macro-Precision** | **99.50%** | **93.61%** | **93.61%** | **-5.89%** |
| **Macro-Recall** | **99.30%** | **93.00%** | **93.00%** | **-6.30%** |
| **Mean Fold Accuracy** | Not reported with std | **93.80% ± 3.84%** | **93.80% ± 3.84%** | — |

*Confusion Matrix (Our Saved Result):*
$$\begin{pmatrix} 191 & 7 & 2 \\ 6 & 189 & 5 \\ 6 & 5 & 89 \end{pmatrix}$$
*31 misclassifications out of 500 recordings.*

---

### Case 2: A+B+C+D (non-ictal) vs E (ictal) — Binary ($N=500$)

| Metric | Paper Table 8 Reported Value | Saved Result (`case2_run_summary.json`) | Recomputed Value | Discrepancy ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **100.0%** (or 99.8%) | **96.80%** (484 / 500) | **96.80%** | **-3.20%** |
| **Sensitivity (Ictal)** | **100.0%** | **86.00%** (86 / 100) | **86.00%** | **-14.00%** |
| **Specificity (Non-ictal)** | **100.0%** | **99.50%** (398 / 400) | **99.50%** | **-0.50%** |
| **Macro-F1** | **100.0%** | **94.76%** | **94.76%** | **-5.24%** |
| **Mean Fold Accuracy** | Not reported with std | **96.80% ± 1.83%** | **96.80% ± 1.83%** | — |

*Confusion Matrix (Our Saved Result):*
$$\begin{pmatrix} 398 & 2 \\ 14 & 86 \end{pmatrix}$$
*16 misclassifications out of 500 recordings (14 false negatives, 2 false positives).*

---

### Case 3: D (interictal) vs E (ictal) — Binary ($N=200$)

| Metric | Paper Table 9 Reported Value | Saved Result (`case3_run_summary.json`) | Recomputed Value | Discrepancy ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **100.0%** | **95.00%** (190 / 200) | **95.00%** | **-5.00%** |
| **Sensitivity (Ictal)** | **100.0%** | **94.00%** (94 / 100) | **94.00%** | **-6.00%** |
| **Specificity (Interictal)**| **100.0%** | **96.00%** (96 / 100) | **96.00%** | **-4.00%** |
| **Macro-F1** | **100.0%** | **95.00%** | **95.00%** | **-5.00%** |
| **Mean Fold Accuracy** | Not reported with std | **95.00% ± 5.00%** | **95.00% ± 5.00%** | — |

*Confusion Matrix (Our Saved Result):*
$$\begin{pmatrix} 96 & 4 \\ 6 & 94 \end{pmatrix}$$
*10 misclassifications out of 200 recordings (6 false negatives, 4 false positives).*

---

## 5. Artifact Verification & Immutability Record

No source files or experimental results were modified or recomputed during this audit:
* `audit/batch_preprocessing/bonn_features_96.csv`: SHA-256 = `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1` (Unchanged)
* `audit/classification/case1_outer_predictions.csv`: SHA-256 = `4c00db9300a61db8378d73b35eb2a5064d48aafe7d5dfd8d828b6fafb9d7d964` (Unchanged)
* `audit/classification/case2_outer_predictions.csv`: SHA-256 = `ba85285c726ec9e63e318a5b22401ad028ec21ec485eb1d5191821b3e68aad20` (Unchanged)
* `audit/classification/case3_outer_predictions.csv`: SHA-256 = `e15f680afd3956e3b04c9458f2843e78ff8692876948c23e38cb38e4d51d06b1` (Unchanged)
