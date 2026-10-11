# Step 2 Implementation Audit

## 1. Scope and Confirmation
This audit was conducted strictly as a read-only process. No files were modified, and no experiments were executed. Source files and saved results in the project directory were inspected to verify the actual implementation against the established paper requirements.

## 2. Actual Source Files and Functions Inspected
- `scripts/preprocessing/normalization.py`: `zscore_normalize()`
- `scripts/preprocessing/embedding.py`: `compute_ami()`, `compute_fnn()`
- `scripts/preprocessing/grp.py`: `reconstruct_phase_space()`, `compute_grp()`, `quantize_grp()`
- `scripts/preprocessing/glcm.py`: `compute_single_glcm()`, `extract_haralick_12()`, `extract_all_glcm_features()`
- `scripts/classification/mcfs.py`: `MCFS.fit()`, `MCFS.transform()`
- `scripts/classification/nested_cv.py`: `tune_inner_cv()`, `run_nested_cross_validation()`
- `scripts/classification/run_cases12.py`: `run_case()`
- `scripts/classification/run_case3_pilot.py`: `main()`

## 3. Component-by-Component Status Table

### A. Dataset and Preprocessing
| Component | Status | Source File | Line Numbers | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| 1. Dataset loading & class mapping | MATCH | `run_cases12.py`, `run_case3_pilot.py` | L250-288, L87-104 | Case 1: 3-class, Case 2: binary (E=pos), Case 3: binary (E=pos). Code correctly maps Bonn subsets to respective class indices. |
| 2. Per-record z-score normalization | MATCH | `normalization.py` | L41-52 | Applies `(x - mean) / std` using population standard deviation (ddof=0). |
| 3. AMI time delay ($\tau$) | MATCH | `embedding.py` | L54-86 | Evaluates AMI using 32 equi-width bins and selects the first local minimum. |
| 4. FNN embedding dimension ($m$) | DIFFERENCE | `embedding.py` | L188-189 | Only applies Kennel Criterion 1 ($R_{tol}=15.0$). Criterion 2 ($A_{tol}$) is ignored intentionally as noted in comments. |
| 5. Takens phase-space reconstruction | MATCH | `grp.py` | L41 | Correctly extracts delayed coordinate vectors. |
| 6. Continuous GRP construction | MATCH | `grp.py` | L64 | Uses `scipy.spatial.distance.cdist` with Euclidean metric. |
| 7. GRP quantization and levels | DIFFERENCE | `grp.py` | L102-103 | Multiplies by `(num_levels - 1)` instead of `K` to map into exactly 16 discrete levels. The paper's Eq. 20 specifies multiplication by `K`. |
| 8. GLCM construction & symmetrization | PAPER_UNDERSPECIFIED | `glcm.py` | L64-83 | Uses $\theta \in \{0, \pi/4\}$ and $d \in \{1, 2, 3, 4\}$. It explicitly forces symmetrization (`P = P + P.T`). The paper omits whether this was done. |
| 9. Haralick 12 feature extraction | MATCH | `glcm.py` | L121-175 | 1-based indexing accurately maps to all 12 Haralick equations. |
| 10. Final feature matrix | MATCH | `glcm.py` | L212-222 | Outputs 96 features (12 features $\times$ 8 GLCMs). |

### B. Classification
| Component | Status | Source File | Line Numbers | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| 1. Class definitions / Sample counts | MATCH | `run_cases12.py`, `run_case3_pilot.py` | L226, L95 | Sample counts are strictly 500 for Cases 1&2 and 200 for Case 3. |
| 2. MCFS implementation | PAPER_UNDERSPECIFIED | `mcfs.py` | L156-190 | Supervised mode disconnects graph for different classes. The paper just states labels are used to steer selection, providing no explicit equations. |
| 3. Preprocessing scaling | MATCH | `nested_cv.py` | L191-193, L366-368 | Standard scaling applied strictly to train folds prior to MCFS. |
| 4. RBF-SVM configuration | MATCH | `nested_cv.py` | L217-223, L383-389 | Uses `SVC(kernel="rbf", decision_function_shape="ovo")`. |
| 5. $C$ and $\gamma$ search grid | MATCH | `nested_cv.py` | L60-61 | Exact base-10 grids from $10^{-3}$ to $10^3$. |
| 6. Inner/Outer CV split | MATCH | `nested_cv.py` | L168-172, L323-327 | 10x10 Stratified splits implemented. |
| 7. Inner-loop metric / tie-breakers | DIFFERENCE | `nested_cv.py` | L256-263 | The paper text conflicts on $k=10$ vs $k \in \{5, 10, 15\}$. The code searches the grid and resolves ties in this custom order: macro-F1 > accuracy > k=10 > k=5 > k=15. |
| 8. Fold isolation | MATCH | `nested_cv.py` | L340-342 | Train/test index overlaps assert verified; no data leakage into outer test folds. |
| 9. Final metrics calculation | MATCH | `nested_cv.py` | L445-450, L453-461 | Evaluated correctly on pooled out-of-sample predictions. |
| 10. Saved results consistency | MATCH | `run_cases12.py` | L170-171 | The parameters in `nested_cv.py` correspond exactly to the saved metadata in the JSON files. |

## 4. Specific Findings and Source Line Numbers
- **GRP Quantization ($K-1$ vs $K$):** `scripts/preprocessing/grp.py:L102-103`. The paper explicitly provides an equation multiplying by $K$. The code multiplies by $K-1=15$ to achieve exact 16-level grayscale range.
- **FNN Dual Criterion:** `scripts/preprocessing/embedding.py:L188-189`. The code explicitly comments and implements the exclusion of Criterion 2 ($A_{tol}$).
- **GLCM Symmetrization:** `scripts/preprocessing/glcm.py:L82-83`. Matrix symmetry is enforced (`P + P.T`). This is standard practice, but the paper never strictly outlines it.
- **Inner-CV Feature Subset Optimization ($k=10$ vs grid):** `scripts/classification/nested_cv.py:L210-213`. The code explicitly tunes $k \in \{5, 10, 15\}$, resolving ties to $10$.

## 5. Actual Classification Execution Path and Fold-Isolation Analysis
The implementation in `nested_cv.py` correctly guarantees data isolation. 
- The outer loop strictly slices $X$ into `X_out_train` and `X_out_test`. 
- `tune_inner_cv` solely accepts `X_out_train`, where it runs its own `StratifiedKFold`. 
- Within `tune_inner_cv`, `StandardScaler` and `MCFS` are fitted exclusively on `X_in_train`. 
- Best parameters are passed back to the outer loop, where scaling and MCFS are refitted solely on `X_out_train` before predicting on `X_out_test`.

## 6. Saved-Results Verification
Saved output files (e.g. `audit/classification/case2_run_summary.json`) perfectly reflect this implementation, recording metric extractions accurately.
For Case 2, positive class (`1`) is clearly mapped to `ictal (E)`. The saved sensitivity (86.00%) correctly corresponds to recall on the ictal class.

## 7. Unresolved Questions Requiring Human Review
- Should we modify the GRP quantization logic to literally multiply by $K$ (even if it produces 17 levels), or keep $K-1$ as a logical correction to the paper?
- Should we activate FNN Criterion 2 ($A_{tol}$)?
- Should we override the custom tie-breaking logic for $k$?

## 8. Final Verdict
`IMPLEMENTATION AUDIT COMPLETE — DIFFERENCES FOUND`
