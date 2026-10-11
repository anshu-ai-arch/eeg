# Bonn EEG Replication Audit Fixes & Verification Report

**Date:** October 2026  
**Repository:** `anshu-ai-arch/eeg`  
**Paper Reference:**  
Lashkari et al., *“Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots,”* Biomedical Signal Processing and Control, 118 (2026), 109675.

---

## 1. Summary of Confirmed Issues and Fixes

### A. Fixed-k Configuration vs. Standard Tuned-k Search
* **Issue:**  
  The audit noted that `scripts/classification/run_kaggle_k5.py` was hard-locked to candidate feature count `candidate_k=[5]`. The paper methodology specifies candidate values $k \in \{5, 10, 15\}$ chosen via inner 10-fold cross-validation.
* **Resolution:**  
  1. Standard nested cross-validation configuration (`NestedCVConfig`) in `scripts/classification/nested_cv.py` now defaults to `candidate_k=[5, 10, 15]`.
  2. Preserved the dedicated fixed-$k=5$ runner `scripts/classification/run_kaggle_k5.py` for exact reproducibility of the fixed-$k$ run under output directory `experiments/fixed_k5` and `experiment_id="experiment_k5"`.
  3. Created `scripts/classification/run_kaggle_tuned.py` for the standard paper candidate search $k \in \{5, 10, 15\}$ under output directory `experiments/tuned_k` and `experiment_id="experiment_tuned_k"`.
  4. Run integrity signatures in `nested_cv.py` explicitly incorporate `k{config.candidate_k}` so checkpoints from a fixed-$k$ run cannot be mistakenly loaded into a tuned-$k$ run (signature mismatch triggers an explicit error).

### B. GRP Quantization Discrepancy ($K-1=15$ vs Literal $K=16$)
* **Issue:**  
  `scripts/preprocessing/grp.py` used $\text{round}(\frac{\text{RP}}{\max(\text{RP})} \times 15)$ with values clipped to $0..15$ (16 discrete levels), whereas Eq. (20) in Lashkari et al. displays a literal multiplier of $K=16$.
* **Analysis & Resolution:**  
  - If a continuous distance matrix is multiplied literally by $K=16$ and rounded, it produces values from $0$ to $16$ ($17$ discrete gray levels), which yields a $17 \times 17$ GLCM rather than standard $16 \times 16$.
  - In literature, $K=16$ often denotes the gray-level count $N_g=16$, where 0-indexed levels are $0 \dots K-1=15$.
  - **Scientifically defensible fix:** Implemented explicit dual-mode support in `quantize_grp()`:
    - `mode="k_minus_1"` (default): $\text{round}(\frac{\text{RP}}{\max(\text{RP})} \times (K - 1))$ producing levels $0..15$ (16 discrete levels, $16 \times 16$ GLCM).
    - `mode="literal_k"` (opt-in): $\text{round}(\frac{\text{RP}}{\max(\text{RP})} \times K)$ producing levels $0..16$ (17 discrete levels, $17 \times 17$ GLCM).
  - Both modes include input validation, non-finite value protection (`np.nan_to_num`), zero/constant maximum guards, and metadata tracking (`get_quantization_metadata()`).
  - **Dataset Invariance:** The frozen feature CSV `audit/batch_preprocessing/bonn_features_96.csv` remains strictly untouched and reflects its original quantization mode (`k_minus_1`). Any experiment testing `literal_k` must regenerate features to a new versioned path.

### C. MCFS Reliability and Methodological Transparency
* **Issue:**  
  MCFS in `scripts/classification/mcfs.py` uses class-dependent edge zeroing and custom QR orthogonalization around the graph Laplacian null space.
* **Resolution:**  
  - Identified and documented in `scripts/classification/mcfs.py` docstrings that Deng Cai et al. (2010) originally formulated MCFS as an unsupervised manifold learning algorithm. Zeroing inter-class edges disconnects the graph into $C$ components, producing a $C$-dimensional null space ($\lambda = 0$).
  - Documented that the deterministic QR decomposition used to extract class-discriminant basis directions orthogonal to the constant eigenvector $v_0$ is an implementation choice for supervised adaptation, as the paper does not specify the exact null-space decomposition formulas.
  - Hardened input validation, verified Lasso $\to$ retry $\to$ Ridge fallback tracking, and ensured stable rank tie-breaking with mergesort.

### D. Nested-CV Isolation, Metric Accounting, and Timing
* **Issue:**  
  The audit found `total_runtime_seconds` reported $\approx 0.002$s when all 10 folds were restored from checkpoint because it only measured the wall time of the fold iteration loop.
* **Resolution:**  
  - Added `cumulative_fold_runtime_seconds` in `run_nested_cross_validation()` return dictionary, which sums the individual recorded fold durations across all completed/restored folds.
  - Preserved `total_runtime_seconds` for the active session execution time.
  - Verified that `StandardScaler` and `MCFS` are strictly fitted on training partitions only.
  - Verified binary sensitivity and specificity correctly evaluate the intended positive class (e.g. seizure ictal class for Case 2 and Case 3).

---

## 2. Unresolved Paper Ambiguities

1. **GRP Quantization ($K=16$ Multiplier vs Gray Levels):**  
   The paper text states that gray-scale recurrence plots use 16 gray levels, but Eq. (20) writes $K \times \frac{RP}{\max(RP)}$ with $K=16$. The paper does not clarify whether GLCM matrices are $16 \times 16$ or $17 \times 17$. We support both modes with clear metadata tracking.
2. **Supervised vs Unsupervised MCFS Details:**  
   The paper cites Cai et al. (2010) (which is unsupervised) while mentioning that classes are used. Exact mathematical details of inter-class graph pruning and null-space eigenvector selection are omitted in the paper.
3. **Inner CV Tie-Breaking Rules:**  
   When multiple $(k, C, \gamma)$ triples achieve identical inner macro-F1, the exact tie-breaking order is not stated in the paper. We use: Macro-F1 $\to$ Mean Accuracy $\to$ prefer $k=10 > k=5 > k=15 \to \min(C) \to \min(\gamma)$.

---

## 3. Configuration Comparison

| Setting | Fixed-k Experiment (`run_kaggle_k5.py`) | Standard Tuned-k Experiment (`run_kaggle_tuned.py`) |
|---|---|---|
| **Entry Point** | `scripts/classification/run_kaggle_k5.py` | `scripts/classification/run_kaggle_tuned.py` |
| **Experiment ID** | `experiment_k5` | `experiment_tuned_k` |
| **Output Directory** | `/kaggle/working/experiments/fixed_k5` | `/kaggle/working/experiments/tuned_k` |
| **Candidate $k$** | `[5]` | `[5, 10, 15]` |
| **Candidate $C$** | $\{10^{-3}, \dots, 10^3\}$ | $\{10^{-3}, \dots, 10^3\}$ |
| **Candidate $\gamma$** | $\{10^{-3}, \dots, 10^3\}$ | $\{10^{-3}, \dots, 10^3\}$ |
| **Inner Selection** | Mean macro-F1 (10 inner folds) | Mean macro-F1 (10 inner folds) |
| **Outer Refitting** | Refits MCFS with $k=5$ | Refits MCFS with selected $k^* \in \{5, 10, 15\}$ |
| **Checkpoint Integrity** | SHA-256 state string contains `k[5]` | SHA-256 state string contains `k[5, 10, 15]` |

---

## 4. Frozen Data Integrity & Quantization Provenance

* **Frozen CSV:** `audit/batch_preprocessing/bonn_features_96.csv`
  - Shape: $(500, 102)$
  - Features: 96 GLCM texture features
  - Quantization Mode: `k_minus_1` ($K=16$, levels $0..15$)
  - Status: **UNTOUCHED and PRESERVED**.
* **Quantization Policy:**  
  Do NOT modify or overwrite `audit/batch_preprocessing/bonn_features_96.csv`. If evaluating `literal_k` ($17$ levels), features must be regenerated to a new, versioned file path such as `bonn_features_96_literal_k.csv`.

---

## 5. Verification & Test Suite Results

All 61 automated unit and integration tests passed cleanly:
```
============================== 61 passed in 4.77s ==============================
```

### Key Test Coverage:
1. `tests/test_audit_fixes.py`:
   - `test_standard_config_contains_candidate_k`: PASSED
   - `test_fixed_k_config_preserves_explicit_5`: PASSED
   - `test_run_signatures_differ_between_fixed_and_tuned`: PASSED
   - `test_checkpoint_signature_mismatch_detection`: PASSED
   - `test_grp_quantization_k_minus_1_mode`: PASSED
   - `test_grp_quantization_literal_k_mode`: PASSED
   - `test_grp_quantization_zero_and_constant`: PASSED
   - `test_grp_quantization_non_finite_values`: PASSED
   - `test_grp_quantization_metadata`: PASSED
   - `test_mcfs_bounds_and_ranking`: PASSED
   - `test_mcfs_invalid_feature_count`: PASSED
   - `test_mcfs_degenerate_near_zero_variance`: PASSED
   - `test_nested_cv_timing_and_metrics`: PASSED
2. `tests/test_grp.py`: All 8 tests PASSED (including discrete levels and invalid mode guards).
3. `tests/test_mcfs.py`: All 12 tests PASSED (supervision, Laplacian, Lasso fallbacks).
4. `tests/test_nested_cv.py`: All 4 tests PASSED (isolation, coverage, determinism).
5. `tests/test_checkpointing.py` & `test_kaggle_backup.py`: All 9 tests PASSED (checkpoint recovery, JSON serialization, backup hashes).

---

## 6. Execution Commands for Next Kaggle Experiments

### Option A: Standard Tuned-$k$ Experiment ($k \in [5, 10, 15]$)
```bash
python scripts/classification/run_kaggle_tuned.py \
    --input /kaggle/input/bonn-eeg-features/bonn_features_96.csv \
    --output /kaggle/working/experiments/tuned_k \
    --resume \
    --remote-restore \
    --restore-dataset
```

### Option B: Reproducing the Historical Fixed-$k=5$ Experiment
```bash
python scripts/classification/run_kaggle_k5.py \
    --input /kaggle/input/bonn-eeg-features/bonn_features_96.csv \
    --output /kaggle/working/experiments/fixed_k5 \
    --resume \
    --remote-restore \
    --restore-dataset
```
