# GRP Quantization Experiment Results

## 1. Objective and Code Changes
The objective of this experiment was to evaluate the effect of modifying the GRP quantization logic to match the literal text of the paper's Eq. 20 (multiplying by $K=16$). 

**Code Changes Implemented:**
- In `scripts/preprocessing/grp.py:L102-L103`, changed the multiplier from `(num_levels - 1)` to `num_levels` and clipped at `num_levels` instead of `num_levels - 1`. This maps continuous values into $[0, 16]$, yielding exactly 17 discrete gray levels.
- In `scripts/preprocessing/pipeline.py:L104-L109`, passed `num_levels=config.grp_num_levels + 1` (17) to `extract_all_glcm_features` to safely accommodate the expanded gray-level dimension without crashing during the GLCM co-occurrence binning.
- Updated path constants in `scripts/run_batch_preprocessing.py`, `run_cases12.py`, and `run_case3_pilot.py` to point to `audit/experiments/grp_quantization_K/`.

## 2. Execution Blocker
The experiment was successfully configured in a separate Git branch (`experiment-grp-quantization-k`) with safely isolated artifact directories. 

However, **execution was blocked by a missing environment dependency**.
Running the preprocessing pipeline directly via `python3 scripts/run_batch_preprocessing.py` failed with the following traceback:
```
Traceback (most recent call last):
  File "/Users/anshutiwari123/phase_sync_eeg/scripts/run_batch_preprocessing.py", line 35, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```
Because the explicit instructions prohibit downloading datasets, installing dependencies, or changing unrelated code to circumvent environment configuration issues, the experiment was aborted.

## 3. Conclusion
**Status:** Inconclusive / Blocked.

The experiment logic and required codebase changes to accommodate the literal $K$ scaling were prepared properly. However, due to a broken or inactive local Python environment missing core requirements (`numpy`), the feature vectors could not be regenerated and the classification metrics could not be computed. No comparison to the baseline can be made until the environment is resolved.
