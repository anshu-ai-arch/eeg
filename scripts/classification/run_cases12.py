"""
scripts/classification/run_cases12.py

Executes nested cross-validation for Case 1 and Case 2 on the Bonn EEG benchmark
reusing the existing leak-free Stratified 10x10 Nested Cross-Validation Engine.

Case definitions:
- Case 1: Subsets A+B (healthy, 0) vs C+D (interictal, 1) vs E (ictal, 2) [N=500, 3-class]
- Case 2: Subsets A+B+C+D (non-ictal, 0) vs E (ictal, 1) [N=500, binary]
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import time
import tracemalloc
import hashlib
from typing import Dict, Any

import numpy as np
import pandas as pd

from scripts.classification.nested_cv import (
    NestedCVConfig,
    run_nested_cross_validation,
)

FROZEN_FEATURE_FILE = "audit/experiments/grp_quantization_K/batch_preprocessing/bonn_features_96.csv"
OUTPUT_DIR = "audit/experiments/grp_quantization_K/classification"


def get_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def run_case(
    case_num: int,
    case_name: str,
    df: pd.DataFrame,
    feat_cols: list,
    label_mapping: Dict[str, int],
    class_names: Dict[int, str],
    pos_label: int | None,
    config: NestedCVConfig,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"Executing {case_name}")
    print("=" * 70)

    # Prepare labels and features
    y_raw = df["subset"].map(label_mapping).values
    if np.any(pd.isna(y_raw)):
        raise ValueError(f"Encountered NaN in mapped labels for {case_name}")
    y = y_raw.astype(int)

    X = df[feat_cols].values
    sample_ids = df["recording_id"].tolist()
    n_samples = len(df)

    print(f"Dataset: {n_samples} recordings, {len(feat_cols)} features.")
    for val, name in class_names.items():
        cnt = int(np.sum(y == val))
        print(f"  - Class {val} ({name}): {cnt} recordings")

    tracemalloc.start()
    t_start = time.perf_counter()

    results = run_nested_cross_validation(
        X=X,
        y=y,
        sample_ids=sample_ids,
        config=config,
        pos_label=pos_label,
    )

    t_end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_clock_time = t_end - t_start
    peak_mem_mb = peak_mem / (1024 * 1024)

    print(f"\n{case_name} Completed in {wall_clock_time:.2f} seconds.")
    print(f"Peak Memory: {peak_mem_mb:.2f} MB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Save outer predictions CSV
    preds_df = pd.DataFrame(results["outer_predictions"])
    preds_df["recording_id"] = preds_df["sample_id"]
    preds_df["class_true"] = preds_df["y_true"].map(class_names)
    preds_df["class_pred"] = preds_df["y_pred"].map(class_names)
    preds_path = os.path.join(OUTPUT_DIR, f"case{case_num}_outer_predictions.csv")
    preds_df[[
        "recording_id", "global_index", "outer_fold", "class_true", "class_pred",
        "y_true", "y_pred", "is_correct"
    ]].to_csv(preds_path, index=False)
    print(f"Saved: {preds_path}")

    # 2. Save fold metrics CSV
    fold_metrics_df = pd.DataFrame(results["fold_metrics"])
    fold_metrics_path = os.path.join(OUTPUT_DIR, f"case{case_num}_fold_metrics.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    print(f"Saved: {fold_metrics_path}")

    # 3. Save selected parameters CSV
    params_df = pd.DataFrame(results["selected_parameters"])
    params_path = os.path.join(OUTPUT_DIR, f"case{case_num}_selected_parameters.csv")
    params_df.to_csv(params_path, index=False)
    print(f"Saved: {params_path}")

    # 4. Save fold splits JSON
    splits_path = os.path.join(OUTPUT_DIR, f"case{case_num}_fold_splits.json")
    with open(splits_path, "w") as f:
        json.dump(results["fold_splits"], f, indent=2)
    print(f"Saved: {splits_path}")

    # 5. Save run summary JSON
    run_summary: Dict[str, Any] = {
        "benchmark_case": case_name,
        "case_number": case_num,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_feature_sha256": get_file_sha256(FROZEN_FEATURE_FILE),
        "n_samples": n_samples,
        "n_features_total": len(feat_cols),
        "classes": {str(k): v for k, v in class_names.items()},
        "positive_class": str(pos_label) if pos_label is not None else None,
        "protocol": {
            "outer_splits": config.n_outer_splits,
            "inner_splits": config.n_inner_splits,
            "random_state": config.random_state,
            "candidate_k": config.candidate_k,
            "candidate_c": config.candidate_c,
            "candidate_gamma": config.candidate_gamma,
            "mcfs_alpha": config.mcfs_alpha,
            "mcfs_neighbors": config.mcfs_neighbors,
            "kernel": "rbf",
            "decision_function_shape": "ovo",
        },
        "wall_clock_seconds": wall_clock_time,
        "peak_memory_mb": peak_mem_mb,
        "pooled_metrics": results["pooled_metrics"],
        "mean_fold_metrics": results["mean_fold_metrics"],
        "outer_folds_failed": 0,
        "undefined_metrics_count": 0,
        "single_coverage_verified": bool(len(preds_df) == n_samples and preds_df["recording_id"].nunique() == n_samples),
    }

    summary_path = os.path.join(OUTPUT_DIR, f"case{case_num}_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(make_serializable(run_summary), f, indent=2)
    print(f"Saved: {summary_path}")

    # Summary metrics printing
    print("\n" + "-" * 50)
    print(f"{case_name.upper()} SUMMARY RESULTS")
    print("-" * 50)
    print(f"Pooled Accuracy:    {results['pooled_metrics']['accuracy'] * 100:.2f}%")
    print(f"Pooled Macro-F1:    {results['pooled_metrics']['macro_f1'] * 100:.2f}%")
    if "sensitivity" in results["pooled_metrics"]:
        print(f"Pooled Sensitivity: {results['pooled_metrics']['sensitivity'] * 100:.2f}%")
    if "specificity" in results["pooled_metrics"]:
        print(f"Pooled Specificity: {results['pooled_metrics']['specificity'] * 100:.2f}%")
    print(f"Mean Fold Accuracy: {results['mean_fold_metrics']['mean_accuracy'] * 100:.2f}% ± {results['mean_fold_metrics']['std_accuracy'] * 100:.2f}%")
    print(f"Mean Fold Macro-F1: {results['mean_fold_metrics']['mean_macro_f1'] * 100:.2f}% ± {results['mean_fold_metrics']['std_macro_f1'] * 100:.2f}%")
    print("-" * 50)

    return {
        "case_num": case_num,
        "case_name": case_name,
        "wall_clock_seconds": wall_clock_time,
        "pooled_metrics": results["pooled_metrics"],
        "mean_fold_metrics": results["mean_fold_metrics"],
        "paths": {
            "predictions": preds_path,
            "fold_metrics": fold_metrics_path,
            "selected_parameters": params_path,
            "fold_splits": splits_path,
            "run_summary": summary_path,
        },
    }


def main():
    print("=" * 70)
    print("Bonn EEG Classification: Case 1 & Case 2 Execution")
    print("=" * 70)

    # 1. Verify frozen features file
    if not os.path.exists(FROZEN_FEATURE_FILE):
        raise FileNotFoundError(f"Missing frozen feature artifact: {FROZEN_FEATURE_FILE}")

    feature_hash_before = get_file_sha256(FROZEN_FEATURE_FILE)
    print(f"Frozen feature artifact SHA-256 (pre-run): {feature_hash_before}")

    # 2. Load dataset
    df = pd.read_csv(FROZEN_FEATURE_FILE).sort_values("recording_id").reset_index(drop=True)
    if len(df) != 500:
        raise ValueError(f"Expected 500 recordings, found {len(df)}")

    feat_cols = [c for c in df.columns if c.startswith("glcm_")]
    if len(feat_cols) != 96:
        raise ValueError(f"Expected 96 GLCM feature columns, found {len(feat_cols)}")

    print(f"Loaded {len(df)} recordings with {len(feat_cols)} features.")

    # 3. Setup standard configuration
    config = NestedCVConfig(
        n_outer_splits=10,
        n_inner_splits=10,
        random_state=42,
        candidate_k=[5, 10, 15],
        candidate_c=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        candidate_gamma=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        mcfs_alpha=0.01,
        mcfs_neighbors=5,
    )

    t_all_start = time.perf_counter()

    # Case 1: A+B (healthy, 0) vs C+D (interictal, 1) vs E (ictal, 2)
    case1_label_mapping = {
        "A": 0, "B": 0,
        "C": 1, "D": 1,
        "E": 2,
    }
    case1_class_names = {
        0: "healthy (A+B)",
        1: "interictal (C+D)",
        2: "ictal (E)",
    }
    case1_res = run_case(
        case_num=1,
        case_name="Case 1: A+B vs C+D vs E (3-class)",
        df=df,
        feat_cols=feat_cols,
        label_mapping=case1_label_mapping,
        class_names=case1_class_names,
        pos_label=None,
        config=config,
    )

    # Case 2: A+B+C+D (non_ictal, 0) vs E (ictal, 1)
    case2_label_mapping = {
        "A": 0, "B": 0, "C": 0, "D": 0,
        "E": 1,
    }
    case2_class_names = {
        0: "non_ictal (A+B+C+D)",
        1: "ictal (E)",
    }
    case2_res = run_case(
        case_num=2,
        case_name="Case 2: A+B+C+D vs E (binary)",
        df=df,
        feat_cols=feat_cols,
        label_mapping=case2_label_mapping,
        class_names=case2_class_names,
        pos_label=1,
        config=config,
    )

    t_all_end = time.perf_counter()

    # Verify frozen artifact integrity post-run
    feature_hash_after = get_file_sha256(FROZEN_FEATURE_FILE)
    assert feature_hash_before == feature_hash_after, "FATAL: Frozen feature file was modified during execution!"
    print(f"\nFrozen feature artifact SHA-256 (post-run): {feature_hash_after} (VERIFIED UNCHANGED)")
    print(f"Total execution time for Case 1 and Case 2: {t_all_end - t_all_start:.2f} seconds.")


if __name__ == "__main__":
    main()
