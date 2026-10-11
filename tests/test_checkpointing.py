import sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock SVC and MCFS
from unittest.mock import patch, MagicMock

import scripts.classification.nested_cv as nested_cv

def test_checkpointing():
    print("Testing Outer Fold Checkpointing and Resume Logic...")
    test_dir = PROJECT_ROOT / "audit/experiments/test_k5"
    if test_dir.exists():
        shutil.rmtree(test_dir)
        
    chk_dir = test_dir / "checkpoints"
    
    # Synthetic Data
    X = np.random.rand(40, 5) # 40 samples, 5 features
    y = np.array([0, 1] * 20)
    sample_ids = [f"S{i}" for i in range(40)]
    
    config = nested_cv.NestedCVConfig(
        n_outer_splits=2,
        n_inner_splits=2,
        random_state=42,
        candidate_k=[5],
        candidate_c=[1.0],
        candidate_gamma=[1.0]
    )
    
    # Mock MCFS to be instant
    with patch("scripts.classification.nested_cv.MCFS") as MockMCFS:
        mock_mcfs_inst = MockMCFS.return_value
        mock_mcfs_inst.fit.return_value = mock_mcfs_inst
        mock_mcfs_inst.transform.side_effect = lambda X: X
        mock_mcfs_inst.ranking_ = np.array([0, 1, 2, 3, 4])
        mock_mcfs_inst.selected_indices_ = np.array([0, 1, 2, 3, 4])
        mock_mcfs_inst.feature_scores_ = np.array([1.0]*5)
        mock_mcfs_inst.alpha_used_ = np.array([0.01])
        mock_mcfs_inst.lasso_fallback_used_ = ["none"]
        
        # Mock SVC to be instant
        with patch("scripts.classification.nested_cv.SVC") as MockSVC:
            mock_svc_inst = MockSVC.return_value
            mock_svc_inst.fit.return_value = mock_svc_inst
            # Mock predict to just return zeros
            mock_svc_inst.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)

            # RUN 1: Complete Fold 0, crash on Fold 1
            # We patch time.perf_counter to raise an Exception on the second fold to simulate crash
            call_count = [0]
            orig_perf_counter = nested_cv.time.perf_counter
            
            def crash_on_fold_1():
                if call_count[0] >= 3: # 3 calls per fold: start, inner end, outer end. If we hit the 4th, it's fold 1 start.
                    raise KeyboardInterrupt("Simulated Crash!")
                call_count[0] += 1
                return orig_perf_counter()
                
            with patch("scripts.classification.nested_cv.time.perf_counter", side_effect=crash_on_fold_1):
                try:
                    nested_cv.run_nested_cross_validation(X, y, sample_ids, config, pos_label=1, checkpoint_dir=chk_dir)
                    print("FAIL: Expected simulated crash.")
                except KeyboardInterrupt:
                    print("Crash simulated successfully.")
                    
            # Check fold 0 is saved
            fold0_file = chk_dir / "fold_0.json"
            if fold0_file.exists():
                print("PASS: Fold 0 was saved.")
            else:
                print("FAIL: Fold 0 not saved.")
                
            fold1_file = chk_dir / "fold_1.json"
            if fold1_file.exists():
                print("FAIL: Fold 1 should not be saved.")
            else:
                print("PASS: Fold 1 not falsely marked complete.")
                
            # RUN 2: Resume
            print("\nResuming...")
            res = nested_cv.run_nested_cross_validation(X, y, sample_ids, config, pos_label=1, checkpoint_dir=chk_dir)
            
            # Verify results
            if len(res["fold_metrics"]) == 2:
                print("PASS: Resume successfully completed and collected both folds.")
            else:
                print(f"FAIL: Expected 2 fold metrics, got {len(res['fold_metrics'])}")
                
            if len(res["outer_predictions"]) == 40:
                print("PASS: Exact 40 predictions without duplication.")
            else:
                print(f"FAIL: Expected 40 predictions, got {len(res['outer_predictions'])}")

if __name__ == "__main__":
    test_checkpointing()
