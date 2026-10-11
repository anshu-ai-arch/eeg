# Bonn EEG Paper Reproduction — Case 3 Pilot Report
**Project:** Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots (2026 Reproduction)  
**Date:** 2026-10-10T10:03:20Z  
**Benchmark:** Case 3 — Set D (`interictal`, $n=100$) vs Set E (`ictal`, $n=100$)  
**Status:** COMPLETE — 100% SUCCESS (200 Recordings Verified)  

---

## 1. Executive Summary & Verification Safeguards

This report documents the verification pilot of the Stratified 10×10 Nested Cross-Validation classification engine on **Case 3 (D vs E)** of the University of Bonn EEG benchmark.
The pilot strictly validates the complete ML pipeline, feature selection via Supervised MCFS, and hyperparameter tuning under the exact paper specifications.

* **Strict Artifact Protection:** Preprocessing features (`bonn_features_96.csv`) verified bit-exact untouched before and after execution (SHA-256: `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`).
* **Single Coverage Verification:** All 200 recordings appear **exactly once** in the pooled outer test predictions.
* **Zero Leakage:** Inner cross-validation scaling and MCFS feature selection executed strictly on inner-training folds. Outer test folds evaluated exactly once.
* **Execution Performance:** Full 10×10 nested CV completed in **175.60 seconds** with peak memory of **2.88 MB** on Apple M1.

---

## 2. Overall Performance Metrics

| Metric | Pooled Outer Value | Mean ± Std Across 10 Folds |
| :--- | :--- | :--- |
| **Accuracy** | 95.00% | 95.00% ± 4.47% |
| **Sensitivity (Recall E / Ictal)** | 94.00% | 94.00% ± 6.63% |
| **Specificity (Recall D / Interictal)** | 96.00% | 96.00% ± 6.63% |
| **Macro Precision** | 95.02% | 95.42% ± 4.13% |
| **Macro Recall** | 95.00% | 95.00% ± 4.47% |
| **Macro F1-Score** | 95.00% | 94.98% ± 4.49% |

### Pooled 2×2 Confusion Matrix (D vs E)
* True D (Interictal): Correct=96, Misclassified=4
* True E (Ictal):      Correct=94, Misclassified=6
* False Positives (D predicted as E): 4
* False Negatives (E predicted as D): 6

---

## 3. Per-Outer-Fold Breakdown

| Fold | Test N | Selected $k$ | Selected $C$ | Selected $\gamma$ | Accuracy | Sensitivity | Specificity | Macro-F1 | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 0 | 20 | 10 | 1000.0 | 0.01 | 100.00% | 100.00% | 100.00% | 100.00% | 17.58s |
| Fold 1 | 20 | 10 | 1000.0 | 0.01 | 90.00% | 90.00% | 90.00% | 90.00% | 16.61s |
| Fold 2 | 20 | 15 | 1000.0 | 0.001 | 100.00% | 100.00% | 100.00% | 100.00% | 16.47s |
| Fold 3 | 20 | 10 | 100.0 | 0.1 | 100.00% | 100.00% | 100.00% | 100.00% | 16.94s |
| Fold 4 | 20 | 10 | 100.0 | 0.1 | 90.00% | 100.00% | 80.00% | 89.90% | 18.61s |
| Fold 5 | 20 | 10 | 1000.0 | 0.001 | 90.00% | 90.00% | 90.00% | 90.00% | 17.22s |
| Fold 6 | 20 | 10 | 1000.0 | 0.01 | 95.00% | 90.00% | 100.00% | 94.99% | 17.61s |
| Fold 7 | 20 | 15 | 1000.0 | 0.01 | 90.00% | 80.00% | 100.00% | 89.90% | 17.29s |
| Fold 8 | 20 | 10 | 1000.0 | 0.001 | 95.00% | 90.00% | 100.00% | 94.99% | 17.88s |
| Fold 9 | 20 | 15 | 100.0 | 0.01 | 100.00% | 100.00% | 100.00% | 100.00% | 19.18s |


---

## 4. Hyperparameter Selection & Stability

Summary of inner-CV selected parameters across the 10 folds:
* **Selected $k$ distribution:** {10: 7, 15: 3}
* **Selected $C$ distribution:** {1000.0: 7, 100.0: 3}
* **Selected $\gamma$ distribution:** {0.01: 5, 0.001: 3, 0.1: 2}

---

## 5. Artifact Audit & Integrity Confirmation

| Artifact | File Path | Status |
| :--- | :--- | :--- |
| Pooled Predictions | [`audit/classification/case3_outer_predictions.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/classification/case3_outer_predictions.csv) | 200 rows, verified |
| Fold Metrics | [`audit/classification/case3_fold_metrics.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/classification/case3_fold_metrics.csv) | 10 folds, verified |
| Selected Parameters | [`audit/classification/case3_selected_parameters.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/classification/case3_selected_parameters.csv) | 10 folds, verified |
| Fold Splits Split-Map | [`audit/classification/fold_splits.json`](file:///Users/anshutiwari123/phase_sync_eeg/audit/classification/fold_splits.json) | Full index map saved |
| Run Summary | [`audit/classification/case3_run_summary.json`](file:///Users/anshutiwari123/phase_sync_eeg/audit/classification/case3_run_summary.json) | Complete execution metadata |
| Frozen Features Hash | `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1` | MATCH (Unmodified) |
