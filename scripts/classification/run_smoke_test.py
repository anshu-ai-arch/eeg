#!/usr/bin/env python3
"""
scripts/classification/run_smoke_test.py

Safe live smoke-test script to verify GitHub backup and restore connectivity.
Uses an isolated, synthetic test artifact that is uploaded, restored, verified,
and safely cleaned up from the remote repository.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.classification.kaggle_backup import run_live_smoke_test, get_kaggle_secrets

def main():
    parser = argparse.ArgumentParser(description="Run live GitHub backup/restore smoke test.")
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Target GitHub repository (e.g. anshu-ai-arch/eeg). Defaults to Kaggle Secret GITHUB_REPO or anshu-ai-arch/eeg."
    )
    args = parser.parse_args()

    token, repo = get_kaggle_secrets()
    target_repo = args.repo or repo

    print("=== GitHub Backup & Restore Smoke Test Preflight ===")
    print(f"Target repository: {target_repo}")
    if not token:
        print("ERROR: GITHUB_TOKEN is not set in Kaggle Secrets or environment.")
        print("To configure in Kaggle: Add 'GITHUB_TOKEN' under Add-ons -> Secrets.")
        sys.exit(1)

    print("GITHUB_TOKEN detected (length: %d chars). Running test..." % len(token))
    success = run_live_smoke_test(override_repo=target_repo)
    if success:
        print("\nSUCCESS: GitHub backup and restore are fully operational!")
        sys.exit(0)
    else:
        print("\nFAILURE: Live smoke test failed. Check permissions or network.")
        sys.exit(1)

if __name__ == "__main__":
    main()
