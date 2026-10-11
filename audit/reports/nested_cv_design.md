# Leakage-Safe Nested Cross-Validation Engine Design
**Project:** Improved Detection of Epileptic Seizures via EEG Signals and Texture Analysis of Recurrence Plots (2026 Reproduction)  
**Document ID:** `NESTED-CV-DESIGN-01`  
**Date:** 2026-10-10  
**Target Module:** `scripts/classification/nested_cv.py`  
**Status:** Architectural Design Complete (Execution Awaiting User Sign-Off)  

---

## 1. Objective and Core Principles

This specification defines the exact architecture for the nested cross-validation classification engine to reproduce the results of Lashkari et al. (2026). The central goal is **absolute data isolation**: ensuring that no test sample information can influence scaling, feature selection, or hyperparameter tuning.

### The 4 Non-Negotiable Leakage Invariants:
1. **Zero Pre-Split Transformation:** No global standardization, centering, PCA, or filtering may touch the feature matrix prior to fold splitting.
2. **Inner-Loop Feature Selection:** The Supervised MCFS algorithm must be fitted strictly inside the training folds. Outer test folds are only masked with the selected feature indices.
3. **Inner-Loop Hyperparameter Tuning:** Grid search for SVM-RBF parameters ($C, \gamma$) and candidate feature count ($k \in \{5, 10, 15\}$) is evaluated exclusively on inner validation folds.
4. **Single-Shot Outer Evaluation:** Outer test folds are evaluated exactly once using the retrained model, producing unbiased out-of-fold predictions.

---

## 2. Nested Cross-Validation Architecture

The engine implements a **Stratified $10 \times 10$ Nested Cross-Validation** structure matching Section 3.6.1 and Section 3.6.3:

```mermaid
flowchart TD
    subgraph Outer["Outer CV Loop: 10 Stratified Folds (Seed 42)"]
        D["Dataset (Case 1: N=500, Case 2: N=500, Case 3: N=200)"] -->|Split 90/10| OTrain["Outer Training Fold (90%)"]
        D -->|Hold out 10%| OTest["Outer Test Fold (10%)"]
        
        subgraph Inner["Inner CV Loop: 10 Stratified Folds on Outer Train (Seed 42)"]
            OTrain -->|Split 90/10| ITrain["Inner Training Fold (81% of total)"]
            OTrain -->|Hold out 10%| IVal["Inner Validation Fold (9% of total)"]
            
            ITrain -->|Fit| ScalerIn["StandardScaler (Inner)"]
            ScalerIn -->|Transform| ITrainS["Scaled Inner Train"]
            ScalerIn -->|Transform| IValS["Scaled Inner Val"]
            
            ITrainS -->|Fit(ITrainS, y_in)| MCFSIn["MCFS Algorithm"]
            MCFSIn -->|Rank & Select k in {5, 10, 15}| FeatIn["Top-k Features"]
            
            FeatIn -->|Grid Search (C, gamma)| SVMIn["SVM-RBF Grid (49 points)"]
            SVMIn -->|Evaluate on IValS| MetricIn["Compute Macro-F1 (tie-break: Accuracy)"]
        end
        
        MetricIn -->|Average over 10 Inner Folds| BestCfg["Select Optimal (k*, C*, gamma*)"]
        
        BestCfg --> Refit["Refit on Full Outer Train (90%)"]
        OTrain -->|Fit| ScalerOut["StandardScaler (Outer)"]
        ScalerOut -->|Transform| OTrainS["Scaled Outer Train"]
        ScalerOut -->|Transform| OTestS["Scaled Outer Test"]
        
        OTrainS -->|Fit(OTrainS, y_out)| MCFSOut["MCFS(k*)"]
        MCFSOut -->|Select k* features| OTrainSel["Outer Train Features"]
        MCFSOut -->|Apply mask k*| OTestSel["Outer Test Features"]
        
        OTrainSel -->|Fit SVM(C*, gamma*)| SVMOut["Final Outer SVM Model"]
        SVMOut -->|Predict(OTestSel)| PredOut["Outer Test Predictions (Recorded Once)"]
    end
    
    PredOut --> Pool["Pool 10 Outer Predictions (500 samples)"]
    Pool --> Metrics["Compute Final Metrics & 95% CIs (B=2000 Bootstrap)"]
```

---

## 3. Step-by-Step Execution Sequence

For each benchmark case (Case 1, Case 2, Case 3):

### Step 1: Dataset Partitioning & Stratification
- Extract feature matrix $X \in \mathbb{R}^{N \times 96}$ and target vector $y \in \mathbb{R}^N$ from [`bonn_features_96.csv`](file:///Users/anshutiwari123/phase_sync_eeg/audit/batch_preprocessing/bonn_features_96.csv).
- Instantiate `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Generate and save the 10 outer fold index partitions to `audit/classification/fold_splits_{case}.json` to guarantee auditability and repeatability.

### Step 2: Outer Fold Iteration (Folds 1 to 10)
For outer fold $i \in \{1, \dots, 10\}$:
1. Partition into `X_outer_train`, `y_outer_train` (90%) and `X_outer_test`, `y_outer_test` (10%).
2. Store sample IDs of the held-out outer test fold.

### Step 3: Inner Fold Hyperparameter & Feature Tuning
Inside outer fold $i$, instantiate `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` on `X_outer_train`:
1. For each inner fold $j \in \{1, \dots, 10\}$:
   - Partition into `X_inner_train`, `y_inner_train` (90% of outer train) and `X_inner_val`, `y_inner_val` (10% of outer train).
   - **Scale:** Fit `StandardScaler` on `X_inner_train`. Transform `X_inner_train` and `X_inner_val`.
   - **MCFS:** Fit `MCFS(n_selected_features=15, n_neighbors=5, alpha=0.01)` on scaled `X_inner_train` and `y_inner_train`.
     Obtain the ranked feature sequence `ranking_`.
   - **Grid Evaluation:** For each candidate feature count $k \in \{5, 10, 15\}$:
     - Subset features: `feats_k = ranking_[:k]`.
     - For each $(C, \gamma) \in \{10^{-3}, 10^{-2}, 10^{-1}, 10^0, 10^1, 10^2, 10^3\}^2$:
       - Train `SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovo', random_state=42)` on `X_inner_train[:, feats_k]`.
       - Predict on `X_inner_val[:, feats_k]`.
       - Compute inner macro-F1 and Accuracy.
2. **Configuration Selection:** Compute the mean macro-F1 across the 10 inner validation folds for each configuration $(k, C, \gamma)$ ($3 \times 7 \times 7 = 147$ combinations).
   - Select $(k^*, C^*, \gamma^*) = \arg\max \text{mean\_macro\_F1}$.
   - Tie-breaker: If multiple configurations achieve identical macro-F1, select the one with highest mean Accuracy. If still tied, prefer smaller $k$ (parsimony), then smaller $C$ (stronger regularization).

### Step 4: Outer Model Refitting & Final Out-of-Fold Prediction
1. Fit `StandardScaler` on the entire `X_outer_train`. Transform `X_outer_train` and `X_outer_test`.
2. Fit `MCFS(n_selected_features=k*, ...)` on the full scaled `X_outer_train` and `y_outer_train`.
   Extract the learned top-$k^*$ feature indices: `top_feats_outer = mcfs.selected_indices_`.
3. Train `SVC(C=C*, gamma=gamma*, kernel='rbf', decision_function_shape='ovo')` on `X_outer_train[:, top_feats_outer]`.
4. Predict classes for the held-out `X_outer_test[:, top_feats_outer]`.
5. Store the predictions, true labels, outer test sample IDs, and the chosen $(k^*, C^*, \gamma^*)$ for fold $i$.

### Step 5: Pooled Performance Evaluation & Statistical Inference
1. Aggregate out-of-fold predictions across all 10 outer folds. Every single recording in the dataset has exactly one out-of-fold prediction.
2. Compute overall performance metrics:
   - Macro Accuracy
   - Macro Precision
   - Macro Recall (Sensitivity)
   - Macro F1-score
   - For Case 3 (D vs E), treat ictal (E) as positive class.
3. Compute fold-wise summary: Mean $\pm$ Standard Deviation across the 10 outer test folds.
4. **Bootstrap Confidence Intervals:** Perform $B=2,000$ stratified bootstrap resamples on the pooled outer test predictions to calculate empirical 95% Confidence Intervals (matching Section 3.7).
5. **Statistical Tests:** Compute McNemar's test for paired comparison against baseline feature sets.

---

## 4. Planned Output Structure for Classification Runs

Outputs will be saved in a dedicated directory `audit/classification/`:

```text
audit/classification/
├── fold_splits_case1.json           # Exact sample IDs in each of the 10 outer folds
├── fold_splits_case2.json
├── fold_splits_case3.json
├── predictions_case1.csv            # 500 rows: recording_id, true_label, pred_label, fold_id
├── predictions_case2.csv            # 500 rows: recording_id, true_label, pred_label, fold_id
├── predictions_case3.csv            # 200 rows: recording_id, true_label, pred_label, fold_id
├── outer_fold_metrics.csv           # 10 rows per case: fold_id, k*, C*, gamma*, Acc, Prec, Rec, F1
└── classification_summary.json      # Complete summary with bootstrap CIs and runtime stats
```

---

## 5. Verification Checklist Prior to Execution

- [x] Preprocessing artifacts frozen and verified (Shape $500 \times 103$, 0 NaNs).
- [x] Supervised MCFS algorithm implemented and verified with 8 passing unit tests.
- [x] SVM grid locked to primary paper's explicit Eq. (22) base-10 values ($7 \times 7 = 49$).
- [x] Inner-loop optimization criterion locked to mean macro-F1 with Accuracy tie-break.
- [x] Outer refitting sequence documented.
- [x] Zero model training performed prior to user authorization.
