# Paper Replication Baseline and Next Action

## 1. Baseline Results and Artifacts
The current implementation and its outputs have been frozen as the baseline for this replication effort.
- **Git Commit:** `0217ce8a32a9b919e9afe354832845f5192e7096`
- **Frozen Features SHA-256:** `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`

**Existing Results Summary:**
* **Case 1 (A+B vs C+D vs E):**
  - Accuracy: 93.80%
  - Macro-F1: 93.29%
  - Macro-Precision: 93.61%
  - Macro-Recall: 93.00%
  - Confusion Matrix: `[[191, 7, 2], [6, 189, 5], [6, 5, 89]]`

* **Case 2 (A+B+C+D vs E):**
  - Accuracy: 96.80%
  - Macro-F1: 94.76%
  - Sensitivity (Ictal): 86.00%
  - Specificity (Non-ictal): 99.50%
  - Confusion Matrix: `[[398, 2], [14, 86]]`

* **Case 3 (D vs E):**
  - Accuracy: 95.00%
  - Macro-F1: 95.00%
  - Sensitivity (Ictal): 94.00%
  - Specificity (Interictal): 96.00%
  - Confusion Matrix: `[[96, 4], [6, 94]]`

## 2. Paper vs. Implementation Comparison

| Case / Metric | Paper Reported | Baseline Implementation | Gap (Percentage Points) |
| :--- | :---: | :---: | :---: |
| **Case 1 Accuracy** | 98.60% | 93.80% | -4.80% |
| **Case 1 Precision** | 98.80% | 93.61% (Macro) | -5.19% |
| **Case 1 Recall** | 98.50% | 93.00% (Macro) | -5.50% |
| **Case 1 F1** | 98.60% | 93.29% (Macro) | -5.31% |
| **Case 2 Accuracy** | 99.60% | 96.80% | -2.80% |
| **Case 2 Sensitivity** | 99.50% | 86.00% | -13.50% |
| **Case 2 Specificity** | 99.70% | 99.50% | -0.20% |
| **Case 2 F1** | 99.60% | 94.76% (Macro) | -4.84% |
| **Case 3 Accuracy** | 100.00% | 95.00% | -5.00% |
| **Case 3 Sensitivity** | 100.00% | 94.00% | -6.00% |
| **Case 3 Specificity** | 100.00% | 96.00% | -4.00% |
| **Case 3 F1** | 100.00% | 95.00% (Macro) | -5.00% |

*(Note: The paper's F1 definition is not explicitly confirmed to be "macro-F1", which may account for some minor divergence in multi-class metrics, but the large accuracy and sensitivity gaps indicate underlying methodological differences.)*

## 3. Explaining the Performance Gap
The following differences were confirmed during the implementation audit:

**Confirmed Methodological Differences:**
1. **GRP Quantization ($K$ vs $K-1$):** The paper specifies multiplying by $K$ (producing 17 gray levels for $K=16$), but the codebase scales by $K-1$ to enforce exactly 16 levels.
2. **FNN Embedding Criterion:** The codebase ignores Criterion 2 ($A_{tol}$) of the False Nearest Neighbors algorithm, potentially leading to different embedding dimensions $m$ than those calculated in the paper.
3. **Inner-CV Tie-Breaking:** The codebase tests $k \in \{5, 10, 15\}$ and uses custom tie-breaking logic, whereas the paper has an unresolved contradiction stating $k=10$ was chosen.
4. **Underspecified MCFS & GLCM:** Supervised MCFS graph constraints and GLCM symmetrization are implemented according to standard conventions, but lack explicit equations in the paper.

**Hypotheses to Test:**
- Does applying the literal GRP quantization ($K$ instead of $K-1$) alter the extracted texture features enough to close the performance gap?
- Does activating FNN Criterion 2 significantly alter the Takens embedding dimension, thereby changing the underlying GRP distances?
- Does fixing $k=10$ (or changing the tie-breaking rules) prevent the selection of suboptimal feature counts during inner cross-validation?

## 4. Single Highest-Priority Next Experiment
**Next Experiment:** Test literal GRP Quantization ($K$ instead of $K-1$).

**Why this should come first:** 
GRP quantization occurs at the very beginning of the feature extraction pipeline. It directly alters the pixel values of the recurrence plots, which fundamentally changes all 96 downstream Haralick texture features. Because every subsequent step (MCFS, SVM) depends on these features, fixing the foundation is a prerequisite for correctly evaluating downstream parameters (like $k$ or $C/\gamma$).

**Expected Outputs and Success Criteria:**
- **Action:** Modify `quantize_grp` in `scripts/preprocessing/grp.py` to multiply by `num_levels` (i.e. $K$) as specified in the paper's Eq. 20.
- **Expected Outputs:** A regenerated feature set (`bonn_features_96.csv`), followed by re-running Cases 1, 2, and 3.
- **Success Criteria:** We will consider this experiment a success if the new pooled metrics (particularly Case 2 Sensitivity) show significant improvement and move measurably closer to the paper's reported values. If the metrics degrade or remain unchanged, we can rule out this discrepancy as the primary cause of the gap.
