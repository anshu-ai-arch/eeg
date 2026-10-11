import os
import sys
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import importlib.util

spec = importlib.util.spec_from_file_location("kaggle_backup", str(PROJECT_ROOT / "scripts" / "classification" / "kaggle_backup.py"))
kaggle_backup = importlib.util.module_from_spec(spec)
sys.modules["kaggle_backup"] = kaggle_backup
spec.loader.exec_module(kaggle_backup)


def test_backup_logic():
    print("=== Testing Checkpoint Backup Upload Logic (Mocked) ===")
    
    # 1. Test Failure when credentials missing
    with patch("kaggle_backup.get_kaggle_secrets", return_value=(None, "anshu-ai-arch/eeg")):
        success = kaggle_backup.trigger_github_backup(Path("/tmp/fake_out"), "exp1", "case1")
        assert not success, "Should return False when no credentials."
        print("PASS: Handled missing credentials safely.")

    # 2. Test Backup Success and Retries
    with patch("kaggle_backup.get_kaggle_secrets", return_value=("fake_token", "anshu-ai-arch/eeg")), \
         patch("subprocess.run") as mock_run:
         
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = ""
            mock_res.stderr = ""
            if "status" in cmd:
                mock_res.stdout = " M somefile.txt"
            elif "push" in cmd:
                mock_run_side_effect.push_count = getattr(mock_run_side_effect, "push_count", 0) + 1
                if mock_run_side_effect.push_count == 1:
                    mock_res.returncode = 1
                    mock_res.stderr = "Simulated network error"
            return mock_res
            
        mock_run.side_effect = mock_run_side_effect
        
        success = kaggle_backup.trigger_github_backup(Path("/tmp/fake_out"), "exp1", "case1")
        assert success, "Backup should succeed on retry."
        assert getattr(mock_run_side_effect, "push_count", 0) == 2, "Expected exactly 2 push attempts."
        print("PASS: Checkpoint backup succeeded after retry.")


def test_restore_logic():
    print("=== Testing Checkpoint Restore Logic (Mocked) ===")
    with patch("kaggle_backup.get_kaggle_secrets", return_value=("fake_token", "anshu-ai-arch/eeg")), \
         patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.glob", return_value=[Path("/tmp/fake_repo/fold_0.json")]), \
         patch("shutil.rmtree"), \
         patch("shutil.copy2") as mock_copy:
         
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_run.return_value = mock_res
        
        success = kaggle_backup.restore_from_github(Path("/tmp/fake_out"), "exp1", "case1")
        assert success, "Restore should have succeeded."
        assert mock_copy.called, "Files should have been copied."
        print("PASS: Checkpoint restore succeeded.")


def test_checksum_computation():
    print("=== Testing Checksum Computation ===")
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("dummy test data\n")
        tf.flush()
        temp_path = Path(tf.name)

    try:
        expected = hashlib.sha256(b"dummy test data\n").hexdigest()
        actual = kaggle_backup.compute_sha256(temp_path)
        assert actual == expected, f"Expected {expected}, got {actual}"
        print("PASS: Streaming SHA-256 calculation verified.")
    finally:
        temp_path.unlink()


def test_dataset_backup_and_validation():
    print("=== Testing Dataset Validation and Backup Guardrails ===")
    # 1. Missing file
    success = kaggle_backup.backup_preprocessed_dataset(Path("/nonexistent/bonn.csv"))
    assert not success, "Missing file should fail backup."
    print("PASS: Handled missing dataset file safely.")

    # 2. Corrupted shape (e.g. wrong row count)
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tf:
        tf.write("col1,col2\n1,2\n")
        tf.flush()
        bad_csv = Path(tf.name)

    try:
        success = kaggle_backup.backup_preprocessed_dataset(bad_csv)
        assert not success, "Invalid row count should fail backup."
        print("PASS: Handled invalid sample count safely.")
    finally:
        bad_csv.unlink()

    # 3. Valid shape and successful backup upload (mocked git)
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tf:
        # Create header with 7 metadata and 96 glcm features
        cols = ["id", "f", "sub", "c", "c1", "c2", "c3"] + [f"glcm_f{i}" for i in range(96)]
        tf.write(",".join(cols) + "\n")
        for i in range(500):
            tf.write(",".join(["0"] * len(cols)) + "\n")
        tf.flush()
        valid_csv = Path(tf.name)

    try:
        with patch("kaggle_backup.get_kaggle_secrets", return_value=("fake_token", "anshu-ai-arch/eeg")), \
             patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = " M data/processed/bonn_features_96.csv"
            mock_res.stderr = ""
            mock_run.return_value = mock_res

            success = kaggle_backup.backup_preprocessed_dataset(valid_csv)
            assert success, "Valid dataset should succeed backup."
            print("PASS: Valid dataset backup succeeded with git integration.")
    finally:
        valid_csv.unlink()


def test_dataset_restore_checksum_mismatch():
    print("=== Testing Dataset Restore Checksum Mismatch Detection ===")
    with tempfile.TemporaryDirectory() as target_dir:
        dest_path = Path(target_dir)

        # Mock git clone to create a modified CSV that does not match expected checksum
        def mock_clone(*args, **kwargs):
            mock_res = MagicMock()
            mock_res.returncode = 0
            # args[0] is ['git', 'clone', '--depth', '1', remote_url, str(repo_tmp)]
            repo_tmp = Path(args[0][-1])
            data_dir = repo_tmp / "data" / "processed"
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(data_dir / "bonn_features_96.csv", "w") as f:
                f.write("corrupted data\n")
            with open(data_dir / "checksums.sha256", "w") as f:
                f.write("some_different_hash  bonn_features_96.csv\n")
            return mock_res

        with patch("kaggle_backup.get_kaggle_secrets", return_value=("fake_token", "anshu-ai-arch/eeg")), \
             patch("subprocess.run", side_effect=mock_clone):
            try:
                kaggle_backup.restore_preprocessed_dataset(dest_path, verify_checksum=True)
                assert False, "Should raise ValueError on checksum mismatch"
            except ValueError as e:
                print(f"PASS: Detected corrupt dataset and raised loudly: {e}")
                # Verify corrupt file is removed
                assert not (dest_path / "bonn_features_96.csv").exists(), "Corrupt file should be unlinked."


def test_idempotent_backup_when_no_changes():
    print("=== Testing Idempotent Backup When Remote Content Unchanged ===")
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tf:
        cols = ["id", "f", "sub", "c", "c1", "c2", "c3"] + [f"glcm_f{i}" for i in range(96)]
        tf.write(",".join(cols) + "\n")
        for i in range(500):
            tf.write(",".join(["0"] * len(cols)) + "\n")
        tf.flush()
        valid_csv = Path(tf.name)

    try:
        with patch("kaggle_backup.get_kaggle_secrets", return_value=("fake_token", "anshu-ai-arch/eeg")), \
             patch("subprocess.run") as mock_run:
            def mock_side_effect(*args, **kwargs):
                cmd = args[0]
                res = MagicMock()
                res.returncode = 0
                res.stderr = ""
                if "status" in cmd:
                    res.stdout = ""  # Clean working tree -> no changes
                return res

            mock_run.side_effect = mock_side_effect
            success = kaggle_backup.backup_preprocessed_dataset(valid_csv)
            assert success, "Should return True idempotently when nothing changed."
            print("PASS: Handled unchanged remote content cleanly without redundant commit.")
    finally:
        valid_csv.unlink()


def test_repo_target_check():
    print("=== Testing Repository Destination Check ===")
    assert kaggle_backup.verify_repo_target("anshu-ai-arch/eeg") is True
    assert kaggle_backup.verify_repo_target("wrong-owner/wrong-repo") is False
    print("PASS: Target repo validation works.")


if __name__ == "__main__":
    test_backup_logic()
    test_restore_logic()
    test_checksum_computation()
    test_dataset_backup_and_validation()
    test_dataset_restore_checksum_mismatch()
    test_idempotent_backup_when_no_changes()
    test_repo_target_check()
    print("\nALL BACKUP & RESTORE UNIT TESTS PASSED!")
