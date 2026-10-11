#!/usr/bin/env python3
"""
scripts/classification/run_kaggle_tuned.py

Entry point for Kaggle Notebook environment to run the Bonn EEG 
nested cross-validation classification protocol with standard tuned k in [5, 10, 15].

Supports outer-fold checkpointing, resuming, and optional GitHub backup via Kaggle Secrets.
Distinguished from fixed_k5 runs by experiment ID, output directory, and candidate_k configuration.
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classification.nested_cv import run_nested_cross_validation, NestedCVConfig
from scripts.classification.kaggle_backup import (
    trigger_github_backup,
    restore_from_github,
    restore_preprocessed_dataset,
)


def setup_kaggle_args():
    parser = argparse.ArgumentParser(description="Run tuned k in [5, 10, 15] EEG classification on Kaggle")
    parser.add_argument(
        "--input", 
        type=str, 
        default="/kaggle/input/bonn-eeg-features/bonn_features_96.csv",
        help="Path to the preprocessed 96-feature CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/kaggle/working/experiments/tuned_k",
        help="Directory to save experiment outputs"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest outer-fold checkpoints"
    )
    parser.add_argument(
        "--remote-restore",
        action="store_true",
        help="Fetch checkpoints from GitHub into the working directory before resuming"
    )
    parser.add_argument(
        "--restore-dataset",
        action="store_true",
        help="Download and verify the preprocessed dataset from GitHub if not found locally"
    )
    return parser.parse_args()


def numpy_json_serializer(obj):
    """
    Safely serialize NumPy scalar values and arrays to native Python types.
    Raises TypeError for unsupported objects.
    """
    if isinstance(obj, np.integer):
        return obj.item()
    if isinstance(obj, np.floating):
        return obj.item()
    if isinstance(obj, np.bool_):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json_atomic(data: dict, target_file: Path, indent: int = 2):
    """
    Atomically write data to target_file using a temporary file and os.replace
    to prevent corrupt or partial JSON files if interrupted.
    """
    target_file = Path(target_file)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = target_file.parent / f"{target_file.name}.tmp"
    
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=numpy_json_serializer)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, target_file)
    finally:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading features from {csv_path}...")
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}. Make sure the dataset is attached to the Kaggle notebook.")
        
    df = pd.read_csv(csv_path)
    if len(df) != 500:
        raise ValueError(f"Expected 500 samples, got {len(df)}")
        
    feature_cols = [c for c in df.columns if c.startswith("glcm_")]
    if len(feature_cols) != 96:
        raise ValueError(f"Expected 96 GLCM features, got {len(feature_cols)}")
        
    if df[feature_cols].isna().sum().sum() > 0:
        raise ValueError("Dataset contains missing values (NaN) in features.")
        
    return df


def run_case(df: pd.DataFrame, case_name: str, label_col: str, output_dir: Path, resume: bool, remote_restore: bool):
    print(f"\n{'='*60}\nStarting {case_name} (Tuned k in [5, 10, 15])\n{'='*60}")
    
    case_out_dir = output_dir / case_name
    chk_dir = case_out_dir / "checkpoints"
    
    if resume and remote_restore:
        restore_from_github(output_dir, "experiment_tuned_k", case_name)
        
    if case_out_dir.exists() and not resume:
        if any(case_out_dir.iterdir()):
            raise RuntimeError(f"Output directory {case_out_dir} is not empty. Use --resume or specify a new --output.")
            
    case_out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir.mkdir(parents=True, exist_ok=True)
    
    df_case = df[df[label_col].notna() & (df[label_col] != "")].copy()
    if len(df_case) == 0:
        print(f"Skipping {case_name}: No valid labels found.")
        return
        
    unique_labels = sorted(df_case[label_col].unique())
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in df_case[label_col]])
    feature_cols = [c for c in df_case.columns if c.startswith("glcm_")]
    X = df_case[feature_cols].values
    sample_ids = df_case["recording_id"].tolist()
    
    pos_label = None
    if len(unique_labels) == 2:
        for lbl, idx in label_map.items():
            if "ictal" in lbl.lower() and "non" not in lbl.lower() and "inter" not in lbl.lower():
                pos_label = idx
        if pos_label is None:
            pos_label = 1
            
    def backup_cb(fold_idx, fold_metrics):
        print(f"Outer fold {fold_idx} complete. Triggering backup...")
        trigger_github_backup(output_dir, "experiment_tuned_k", case_name)
            
    config = NestedCVConfig(
        n_outer_splits=10,
        n_inner_splits=10,
        random_state=42,
        candidate_k=[5, 10, 15],
    )
    
    results = run_nested_cross_validation(
        X=X,
        y=y,
        sample_ids=sample_ids,
        config=config,
        pos_label=pos_label,
        checkpoint_dir=chk_dir,
        fold_callback=backup_cb
    )
    
    results["metadata"] = {
        "case_name": case_name,
        "experiment_id": "experiment_tuned_k",
        "class_mapping": label_map,
        "pos_label_index": pos_label,
        "candidate_k": [5, 10, 15],
        "execution_time_iso": datetime.now(timezone.utc).isoformat()
    }
    
    out_file = case_out_dir / f"{case_name}_results.json"
    save_json_atomic(results, out_file, indent=2)
        
    print(f"Completed {case_name}. Results saved to {out_file}")


def main():
    args = setup_kaggle_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    print("=== Kaggle Tuned k in [5, 10, 15] Experiment ===")
    print(f"Input Path:      {input_path}")
    print(f"Output Dir:      {output_dir}")
    print(f"Resume Mode:     {args.resume}")
    print(f"Remote Restore:  {args.remote_restore}")
    print(f"Restore Dataset: {args.restore_dataset}")
    
    if args.restore_dataset or not input_path.exists():
        if not input_path.exists():
            print(f"Input path {input_path} not found. Attempting to restore preprocessed dataset from GitHub...")
            restored = restore_preprocessed_dataset(input_path.parent)
            if restored and restored.exists():
                input_path = restored
            else:
                print("Failed to restore preprocessed dataset from GitHub.")

    df = load_and_validate_data(input_path)
    
    run_case(df, "Case1_3Class", "case1_label", output_dir, args.resume, args.remote_restore)
    run_case(df, "Case2_Binary", "case2_label", output_dir, args.resume, args.remote_restore)
    run_case(df, "Case3_Pilot", "case3_label", output_dir, args.resume, args.remote_restore)
    
    print("\nAll cases finished successfully. Outputs are available in /kaggle/working/")


if __name__ == "__main__":
    main()
