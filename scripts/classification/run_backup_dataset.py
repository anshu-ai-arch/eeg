#!/usr/bin/env python3
"""
scripts/classification/run_backup_dataset.py

CLI entry point to back up the preprocessed Bonn EEG feature dataset
to GitHub repository anshu-ai-arch/eeg.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classification.kaggle_backup import backup_preprocessed_dataset, get_kaggle_secrets

def main():
    parser = argparse.ArgumentParser(description="Upload preprocessed EEG dataset to GitHub.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "audit" / "batch_preprocessing" / "bonn_features_96.csv"),
        help="Path to bonn_features_96.csv"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Target GitHub repository (default: anshu-ai-arch/eeg or Kaggle Secret GITHUB_REPO)"
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    token, repo = get_kaggle_secrets()
    target_repo = args.repo or repo

    print("=== Preprocessed Dataset Backup CLI ===")
    print(f"Dataset path:      {csv_path}")
    print(f"Target repository: {target_repo}")

    if not csv_path.exists():
        print(f"ERROR: Dataset file not found at {csv_path}")
        sys.exit(1)

    if not token:
        print("ERROR: GITHUB_TOKEN not found in Kaggle Secrets or environment variables.")
        sys.exit(1)

    success = backup_preprocessed_dataset(csv_path=csv_path, override_repo=target_repo)
    if success:
        print("\nSUCCESS: Dataset backed up to GitHub successfully.")
        sys.exit(0)
    else:
        print("\nFAILURE: Dataset backup failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
