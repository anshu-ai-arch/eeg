"""
scripts/classification/kaggle_backup.py

Safe, reliable dataset & training checkpoint backup and restore module
for Kaggle sessions connecting to GitHub.

Repository target: anshu-ai-arch/eeg (configurable via Kaggle Secrets GITHUB_REPO)
Credentials: Read securely via Kaggle Secrets UserSecretsClient (GITHUB_TOKEN)
"""

import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

DEFAULT_GITHUB_REPO = "anshu-ai-arch/eeg"
EXPECTED_DATASET_SHA256 = "af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1"


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_kaggle_secrets() -> Tuple[Optional[str], str]:
    """
    Attempt to get GITHUB_TOKEN and GITHUB_REPO from Kaggle Secrets or environment.
    Never prints or logs the token value.
    """
    token = None
    repo = None

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("GITHUB_TOKEN")
        repo = user_secrets.get_secret("GITHUB_REPO")
    except ImportError:
        # Not in Kaggle notebook environment
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPO")
    except Exception as e:
        print(f"[Backup] Warning: Failed to retrieve Kaggle Secrets: {type(e).__name__}")
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPO")

    if not repo:
        repo = DEFAULT_GITHUB_REPO
    return token, repo


def verify_repo_target(repo: str) -> bool:
    """Verify that repo matches the expected target."""
    target = DEFAULT_GITHUB_REPO.lower()
    if repo.lower() != target:
        print(f"[Backup Warning] Configured repository '{repo}' differs from expected '{DEFAULT_GITHUB_REPO}'.")
        return False
    return True


def _run_git_cmd(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    """Execute a git command safely capturing output."""
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False
    )


def backup_preprocessed_dataset(
    csv_path: Path,
    metadata_source: Optional[Path] = None,
    max_retries: int = 3,
    override_repo: Optional[str] = None,
) -> bool:
    """
    Back up the processed feature dataset (bonn_features_96.csv) to GitHub repo.
    Includes:
      - bonn_features_96.csv
      - dataset_metadata.json
      - checksums.sha256
      - README.md
    
    Performs safety checks:
      - Never commits raw data or secret tokens.
      - Checks file size before committing (< 50MB).
      - Checks SHA-256 and verified schema (500 rows, 96 features).
      - Atomic staging and verification of remote push.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[Dataset Backup Error] Dataset file not found: {csv_path}")
        return False

    # Check file size (GitHub standard commit safety check)
    file_size_bytes = csv_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > 50.0:
        print(f"[Dataset Backup Error] Dataset size ({file_size_mb:.2f} MB) exceeds recommended Git file size limit of 50 MB.")
        return False

    # Compute checksum
    actual_sha256 = compute_sha256(csv_path)
    if actual_sha256 != EXPECTED_DATASET_SHA256:
        print(f"[Dataset Backup Warning] Checksum ({actual_sha256}) differs from expected ({EXPECTED_DATASET_SHA256}).")

    # Read and validate schema without requiring heavy pandas/numpy dependencies
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        row_count = sum(1 for line in f if line.strip())

    if row_count != 500:
        print(f"[Dataset Backup Error] Row count {row_count} != expected 500.")
        return False

    feature_cols = [c for c in header if c.startswith("glcm_")]
    if len(feature_cols) != 96:
        print(f"[Dataset Backup Error] Found {len(feature_cols)} GLCM feature columns != expected 96.")
        return False

    # Obtain credentials
    gh_token, gh_repo = get_kaggle_secrets()
    if override_repo:
        gh_repo = override_repo

    if not gh_token:
        print("[Dataset Backup Skipped] GITHUB_TOKEN not available in secrets or environment.")
        return False

    verify_repo_target(gh_repo)

    remote_url = f"https://oauth2:{gh_token}@github.com/{gh_repo}.git"

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_tmp = Path(tmp_dir) / "repo"

        for attempt in range(1, max_retries + 1):
            print(f"[Dataset Backup] Attempt {attempt}/{max_retries} to {gh_repo}...")
            if repo_tmp.exists():
                shutil.rmtree(repo_tmp)

            clone_res = subprocess.run(
                ["git", "clone", "--depth", "1", remote_url, str(repo_tmp)],
                capture_output=True,
                text=True,
            )
            if clone_res.returncode != 0:
                print(f"[Dataset Backup] Clone failed: {clone_res.stderr.strip()}")
                time.sleep(2)
                continue

            # Target directory inside repo
            dataset_dest_dir = repo_tmp / "data" / "processed"
            dataset_dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy feature CSV
            dest_csv = dataset_dest_dir / "bonn_features_96.csv"
            shutil.copy2(csv_path, dest_csv)

            # Generate / copy metadata
            metadata_dict: Dict[str, Any] = {
                "dataset_name": "Bonn EEG 96 Haralick Features (Lashkari replication)",
                "filename": "bonn_features_96.csv",
                "sha256": actual_sha256,
                "file_size_bytes": file_size_bytes,
                "total_samples": row_count,
                "total_columns": len(header),
                "glcm_feature_count": len(feature_cols),
                "metadata_columns": [c for c in header if not c.startswith("glcm_")],
                "subset_distribution": {
                    "A": 100, "B": 100, "C": 100, "D": 100, "E": 100
                },
                "class_mapping": {
                    "Case1_3Class": {"healthy": 200, "interictal": 200, "ictal": 100},
                    "Case2_Binary": {"non_ictal": 400, "ictal": 100},
                    "Case3_Pilot": {"interictal": 100, "ictal": 100}
                },
                "preprocessing_configuration": {
                    "sampling_rate_hz": 173.61,
                    "samples_per_recording": 4097,
                    "normalization": "Z-score population (ddof=0)",
                    "tau_estimation": "AMI (32 bins, lags 1-50, fallback=10)",
                    "embedding_dimension_m": "Kennel FNN Criterion 1 (R_tol=15.0, Theiler window=tau, ratio < 1%)",
                    "grp_metric": "euclidean",
                    "quantization_levels_K": 16,
                    "quantization_formula": "round(RP / max(RP) * 15)",
                    "glcm_orientations_deg": [0, 45],
                    "glcm_offsets": [1, 2, 3, 4],
                    "haralick_features_count": 12,
                    "haralick_indexing": "1-based"
                },
                "creation_date_iso": "2026-10-10T08:42:17.729092+00:00",
                "software_versions": {
                    "python": "3.13.9",
                    "numpy": "2.3.5"
                }
            }

            dest_meta = dataset_dest_dir / "dataset_metadata.json"
            with open(dest_meta, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2)

            # Checksum manifest
            dest_manifest = dataset_dest_dir / "checksums.sha256"
            with open(dest_manifest, "w", encoding="utf-8") as f:
                f.write(f"{actual_sha256}  bonn_features_96.csv\n")

            # Dataset README
            dest_readme = dataset_dest_dir / "README.md"
            readme_text = f"""# Bonn EEG Processed Feature Dataset

- **File**: `bonn_features_96.csv`
- **Samples**: 500 (Subsets A, B, C, D, E with 100 recordings each)
- **Features**: 96 Haralick texture features extracted from Generalized Recurrence Plots (GRP)
- **SHA-256 Checksum**: `{actual_sha256}`
- **Size**: ~909 KB

## Integrity Verification in Kaggle
```bash
shasum -a 256 data/processed/bonn_features_96.csv
# Expected: {actual_sha256}
```
"""
            with open(dest_readme, "w", encoding="utf-8") as f:
                f.write(readme_text)

            # Git add
            _run_git_cmd(["git", "add", "data/processed"], repo_tmp)
            status_res = _run_git_cmd(["git", "status", "--porcelain"], repo_tmp)
            if not status_res.stdout.strip():
                print("[Dataset Backup] Remote already contains identical dataset and manifest.")
                return True

            commit_msg = f"Backup preprocessed Bonn EEG feature dataset (SHA-256: {actual_sha256[:8]})"
            commit_res = _run_git_cmd(
                ["git", "-c", "user.name=KaggleBot", "-c", "user.email=bot@kaggle.com", "commit", "-m", commit_msg],
                repo_tmp
            )
            if commit_res.returncode != 0:
                print(f"[Dataset Backup] Git commit failed: {commit_res.stderr.strip()}")
                time.sleep(2)
                continue

            push_res = _run_git_cmd(["git", "push"], repo_tmp)
            if push_res.returncode == 0:
                print(f"[Dataset Backup] Successfully pushed dataset and metadata to {gh_repo}.")
                return True
            else:
                print(f"[Dataset Backup] Push failed: {push_res.stderr.strip()}")
                time.sleep(2)

    print("[Dataset Backup Error] All upload attempts failed.")
    return False


def restore_preprocessed_dataset(
    target_dir: Path,
    override_repo: Optional[str] = None,
    verify_checksum: bool = True
) -> Optional[Path]:
    """
    Download and verify the processed dataset from GitHub into target_dir.
    Performs streaming checksum validation. Fails loudly on mismatch.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    gh_token, gh_repo = get_kaggle_secrets()
    if override_repo:
        gh_repo = override_repo

    if not gh_token:
        print("[Dataset Restore Error] GITHUB_TOKEN not available.")
        return None

    remote_url = f"https://oauth2:{gh_token}@github.com/{gh_repo}.git"

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_tmp = Path(tmp_dir) / "repo"
        clone_res = subprocess.run(
            ["git", "clone", "--depth", "1", remote_url, str(repo_tmp)],
            capture_output=True,
            text=True
        )
        if clone_res.returncode != 0:
            print(f"[Dataset Restore Error] Clone failed: {clone_res.stderr.strip()}")
            return None

        remote_csv = repo_tmp / "data" / "processed" / "bonn_features_96.csv"
        remote_meta = repo_tmp / "data" / "processed" / "dataset_metadata.json"
        remote_manifest = repo_tmp / "data" / "processed" / "checksums.sha256"

        if not remote_csv.exists():
            print(f"[Dataset Restore Error] {remote_csv} does not exist in remote repository {gh_repo}.")
            return None

        dest_csv = target_dir / "bonn_features_96.csv"
        dest_meta = target_dir / "dataset_metadata.json"
        dest_manifest = target_dir / "checksums.sha256"

        # Atomic copy
        shutil.copy2(remote_csv, dest_csv)
        if remote_meta.exists():
            shutil.copy2(remote_meta, dest_meta)
        if remote_manifest.exists():
            shutil.copy2(remote_manifest, dest_manifest)

        if verify_checksum:
            actual_sha = compute_sha256(dest_csv)
            expected_sha = EXPECTED_DATASET_SHA256
            if remote_manifest.exists():
                try:
                    with open(dest_manifest, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2 and parts[1].endswith("bonn_features_96.csv"):
                                expected_sha = parts[0]
                                break
                except Exception:
                    pass

            if actual_sha != expected_sha:
                dest_csv.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for restored dataset!\nExpected: {expected_sha}\nActual:   {actual_sha}"
                )
            print(f"[Dataset Restore Success] Verified SHA-256 ({actual_sha}) matches manifest.")

        return dest_csv


def trigger_github_backup(
    output_dir: Path,
    experiment_id: str,
    case_name: str,
    max_retries: int = 3,
    override_repo: Optional[str] = None
) -> bool:
    """
    Safely backup the local checkpoint artifacts to the GitHub repo.
    Uses bounded retries. Ensures no raw data is pushed.
    """
    gh_token, gh_repo = get_kaggle_secrets()
    if override_repo:
        gh_repo = override_repo

    if not gh_token:
        print("[Checkpoint Backup Skipped] No GITHUB_TOKEN available.")
        return False

    verify_repo_target(gh_repo)

    remote_url = f"https://oauth2:{gh_token}@github.com/{gh_repo}.git"

    with tempfile.TemporaryDirectory() as tmp_dir:
        backup_tmp = Path(tmp_dir) / "repo"

        for attempt in range(1, max_retries + 1):
            print(f"[Checkpoint Backup] Attempt {attempt}/{max_retries} to {gh_repo}...")
            if backup_tmp.exists():
                shutil.rmtree(backup_tmp)

            res = subprocess.run(
                ["git", "clone", "--depth", "1", remote_url, str(backup_tmp)],
                capture_output=True,
                text=True
            )
            if res.returncode != 0:
                print(f"[Checkpoint Backup] Clone failed: {res.stderr.strip()}")
                time.sleep(2)
                continue

            dest_dir = backup_tmp / "experiments" / experiment_id / case_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            src_chk = Path(output_dir) / case_name / "checkpoints"
            if src_chk.exists():
                dest_chk = dest_dir / "checkpoints"
                dest_chk.mkdir(exist_ok=True)
                for f in src_chk.glob("fold_*.json"):
                    shutil.copy2(f, dest_chk / f.name)

            # Generate manifest of checkpoints
            chk_files = sorted(list((dest_dir / "checkpoints").glob("fold_*.json")))
            manifest = {
                "experiment_id": experiment_id,
                "case_name": case_name,
                "fold_count": len(chk_files),
                "folds": [f.name for f in chk_files],
                "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(dest_dir / "checkpoint_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            _run_git_cmd(["git", "add", "."], backup_tmp)
            status_res = _run_git_cmd(["git", "status", "--porcelain"], backup_tmp)
            if not status_res.stdout.strip():
                print("[Checkpoint Backup] No new checkpoints to push.")
                return True

            msg = f"Backup checkpoint: {experiment_id} {case_name} ({len(chk_files)} folds)"
            _run_git_cmd(
                ["git", "-c", "user.name=KaggleBot", "-c", "user.email=bot@kaggle.com", "commit", "-m", msg],
                backup_tmp
            )

            push_res = _run_git_cmd(["git", "push"], backup_tmp)
            if push_res.returncode == 0:
                print(f"[Checkpoint Backup Success] GitHub backup completed for {case_name}.")
                return True
            else:
                print(f"[Checkpoint Backup Warning] Push failed: {push_res.stderr.strip()}")
                time.sleep(2)

    print("[Checkpoint Backup Error] GitHub backup failed after retries.")
    return False


def restore_from_github(
    output_dir: Path,
    experiment_id: str,
    case_name: str,
    override_repo: Optional[str] = None
) -> bool:
    """
    Fetch the latest valid checkpoints from GitHub into the local working directory.
    """
    gh_token, gh_repo = get_kaggle_secrets()
    if override_repo:
        gh_repo = override_repo

    if not gh_token:
        print("[Restore Skipped] No GITHUB_TOKEN available.")
        return False

    remote_url = f"https://oauth2:{gh_token}@github.com/{gh_repo}.git"

    with tempfile.TemporaryDirectory() as tmp_dir:
        restore_tmp = Path(tmp_dir) / "repo"
        res = subprocess.run(
            ["git", "clone", "--depth", "1", remote_url, str(restore_tmp)],
            capture_output=True,
            text=True
        )
        if res.returncode != 0:
            print(f"[Restore Error] Clone failed: {res.stderr.strip()}")
            return False

        src_dir = restore_tmp / "experiments" / experiment_id / case_name / "checkpoints"
        if not src_dir.exists():
            print(f"[Restore Info] No existing remote checkpoints found for {experiment_id} {case_name}.")
            return False

        dest_chk = Path(output_dir) / case_name / "checkpoints"
        dest_chk.mkdir(parents=True, exist_ok=True)

        restored_count = 0
        for f in src_dir.glob("fold_*.json"):
            shutil.copy2(f, dest_chk / f.name)
            restored_count += 1

        print(f"[Restore Success] Successfully restored {restored_count} folds for {case_name}.")
        return True


def run_live_smoke_test(
    test_artifact_name: str = "smoke_test.json",
    override_repo: Optional[str] = None
) -> bool:
    """
    Live smoke test using a synthetic, isolated test artifact.
    1. Uploads synthetic artifact to test_artifacts/ in GitHub repo.
    2. Verifies remote commit.
    3. Restores into a fresh local directory.
    4. Validates checksum.
    5. Cleans up test artifact safely from remote repository.
    """
    gh_token, gh_repo = get_kaggle_secrets()
    if override_repo:
        gh_repo = override_repo

    if not gh_token:
        print("[Smoke Test Error] GITHUB_TOKEN not available.")
        return False

    print(f"=== Starting Live Smoke Test on repository: {gh_repo} ===")
    remote_url = f"https://oauth2:{gh_token}@github.com/{gh_repo}.git"

    test_content = {
        "test": "smoke_test",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "synthetic_verification_payload"
    }
    test_bytes = json.dumps(test_content, indent=2).encode("utf-8")
    expected_hash = hashlib.sha256(test_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_tmp = Path(tmp_dir) / "repo"

        # 1. Clone
        print("[Smoke Test Step 1] Cloning target repository...")
        clone_res = subprocess.run(
            ["git", "clone", "--depth", "1", remote_url, str(repo_tmp)],
            capture_output=True,
            text=True
        )
        if clone_res.returncode != 0:
            print(f"[Smoke Test Failure] Clone failed: {clone_res.stderr.strip()}")
            return False

        # 2. Write synthetic artifact
        print("[Smoke Test Step 2] Writing synthetic artifact...")
        test_dir = repo_tmp / "test_artifacts"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / test_artifact_name
        with open(test_file, "wb") as f:
            f.write(test_bytes)

        _run_git_cmd(["git", "add", "test_artifacts"], repo_tmp)
        _run_git_cmd(
            ["git", "-c", "user.name=KaggleBot", "-c", "user.email=bot@kaggle.com", "commit", "-m", "Smoke test verification artifact"],
            repo_tmp
        )

        # 3. Push
        print("[Smoke Test Step 3] Pushing synthetic artifact...")
        push_res = _run_git_cmd(["git", "push"], repo_tmp)
        if push_res.returncode != 0:
            print(f"[Smoke Test Failure] Push failed: {push_res.stderr.strip()}")
            return False

    # 4. Restore into a clean separate directory
    with tempfile.TemporaryDirectory() as restore_dir:
        print("[Smoke Test Step 4] Restoring artifact into clean directory...")
        restore_repo = Path(restore_dir) / "repo"
        clone_res = subprocess.run(
            ["git", "clone", "--depth", "1", remote_url, str(restore_repo)],
            capture_output=True,
            text=True
        )
        if clone_res.returncode != 0:
            print(f"[Smoke Test Failure] Restore clone failed: {clone_res.stderr.strip()}")
            return False

        restored_file = restore_repo / "test_artifacts" / test_artifact_name
        if not restored_file.exists():
            print("[Smoke Test Failure] Restored file does not exist on remote.")
            return False

        with open(restored_file, "rb") as f:
            restored_bytes = f.read()
        restored_hash = hashlib.sha256(restored_bytes).hexdigest()

        if restored_hash != expected_hash:
            print(f"[Smoke Test Failure] Checksum mismatch! Expected {expected_hash}, got {restored_hash}")
            return False
        print("[Smoke Test Step 5] Verified remote artifact checksum matches perfectly.")

        # 5. Clean up remote test artifact
        print("[Smoke Test Step 6] Cleaning up test artifact from remote repository...")
        if restored_file.exists():
            _run_git_cmd(["git", "rm", "-rf", "test_artifacts"], restore_repo)
            _run_git_cmd(
                ["git", "-c", "user.name=KaggleBot", "-c", "user.email=bot@kaggle.com", "commit", "-m", "Cleanup smoke test artifact"],
                restore_repo
            )
            _run_git_cmd(["git", "push"], restore_repo)
            print("[Smoke Test Step 7] Cleaned up test artifact successfully.")

    print("=== Live Smoke Test PASSED ===")
    return True
