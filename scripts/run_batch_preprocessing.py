#!/usr/bin/env python3
"""
run_batch_preprocessing.py

Executes sequential, memory-safe preprocessing on all 500 Bonn EEG recordings.
Preserves all approved scientific decisions:
- 4,097 samples per recording, no additional digital filtering.
- Z-score normalization with population std (ddof=0).
- AMI time-delay estimation (32 bins, lags 1-50, fallback tau=10).
- Kennel FNN embedding dimension estimation (Criterion 1, R_tol=15.0, W=tau, threshold < 1%).
- Takens phase-space trajectory reconstruction.
- Continuous Euclidean GRP distance matrix.
- Quantization to K=16 gray levels using round(RP / max(RP) * 15).
- 8 symmetric, normalized GLCMs (0° and 45°, offsets 1-4).
- 12 Haralick features using 1-based indexing, yielding exactly 96 features.

Resource Safeguards:
- Strict sequential execution (one recording at a time).
- Intermediate distance matrices are discarded immediately; never persisted to disk.
- Incremental CSV writing with immediate file flushes.
- Robust preflight and post-processing validation gates.
"""

import sys
import os
import time
import json
import csv
import gc
import hashlib
import resource
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocessing.config import PreprocessingConfig
from scripts.preprocessing.pipeline import preprocess_recording, calculate_file_sha256


# Canonical Subset and Class Mapping
SUBSET_METADATA = {
    "A": {
        "description": "Healthy volunteers, eyes open (surface EEG)",
        "class_label": "healthy",
        "case1_label": "healthy",
        "case2_label": "non_ictal",
        "case3_label": None,
    },
    "B": {
        "description": "Healthy volunteers, eyes closed (surface EEG)",
        "class_label": "healthy",
        "case1_label": "healthy",
        "case2_label": "non_ictal",
        "case3_label": None,
    },
    "C": {
        "description": "Inter-ictal, hippocampal formation (opposite hemisphere)",
        "class_label": "interictal",
        "case1_label": "interictal",
        "case2_label": "non_ictal",
        "case3_label": None,
    },
    "D": {
        "description": "Inter-ictal, epileptogenic zone",
        "class_label": "interictal",
        "case1_label": "interictal",
        "case2_label": "non_ictal",
        "case3_label": "interictal",
    },
    "E": {
        "description": "Ictal, seizure activity",
        "class_label": "ictal",
        "case1_label": "ictal",
        "case2_label": "ictal",
        "case3_label": "ictal",
    },
}


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_preprocessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def discover_recordings(extracted_dir: Path) -> list[dict]:
    """
    Discover and validate all 500 Bonn EEG recordings.
    Enforces strict subset and filename patterns.
    """
    recordings = []
    subsets = ["A", "B", "C", "D", "E"]

    for s in subsets:
        subset_dir = extracted_dir / s
        if not subset_dir.exists() or not subset_dir.is_dir():
            raise FileNotFoundError(f"Missing subset directory: {subset_dir}")

        # Filter out hidden files, macOS resource forks (._*), and non-text files
        valid_files = [
            f for f in subset_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in [".txt"]
        ]

        if len(valid_files) != 100:
            raise ValueError(f"Subset {s} expected exactly 100 files, found {len(valid_files)}")

        # Sort naturally by filename
        valid_files.sort(key=lambda x: x.name)

        for f in valid_files:
            rec_id = f.stem
            recordings.append({
                "recording_id": rec_id,
                "filename": f.name,
                "filepath": f,
                "subset": s,
                "meta": SUBSET_METADATA[s]
            })

    if len(recordings) != 500:
        raise ValueError(f"Expected exactly 500 recordings total, found {len(recordings)}")

    # Verify uniqueness of recording IDs
    unique_ids = set(r["recording_id"] for r in recordings)
    if len(unique_ids) != 500:
        raise ValueError(f"Duplicate recording IDs detected! Unique count: {len(unique_ids)}")

    return recordings


def get_peak_rss_mb() -> float:
    """Return peak resident set size in megabytes (macOS returns bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024.0 * 1024.0)


def run_batch_preprocessing(
    output_dir: Path | None = None,
    clean_restart: bool = True
) -> dict:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "audit" / "experiments" / "grp_quantization_K" / "batch_preprocessing"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "batch_preprocessing.log"
    features_csv = output_dir / "bonn_features_96.csv"
    diagnostics_csv = output_dir / "batch_diagnostics.csv"
    summary_json = output_dir / "run_summary.json"

    logger = setup_logger(log_file)
    logger.info("=" * 80)
    logger.info("STARTING BONN EEG BATCH PREPROCESSING (500 RECORDINGS)")
    logger.info(f"Target Output Directory: {output_dir}")
    logger.info("=" * 80)

    # GATE 1: Preflight Verification
    extracted_dir = PROJECT_ROOT / "data" / "bonn_eeg" / "extracted"
    logger.info(f"Discovering recordings in {extracted_dir}...")
    recordings = discover_recordings(extracted_dir)
    logger.info(f"Preflight Check Passed: Found exactly 500 recordings across subsets A-E (100 each).")

    # Check free disk space
    import shutil
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / (1024.0 ** 3)
    logger.info(f"Filesystem free space: {free_gb:.2f} GiB (Safe threshold: > 2 GiB)")
    if free_gb < 1.0:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GiB available.")

    # Initialize configuration
    config = PreprocessingConfig()
    logger.info(f"Preprocessing Configuration: sampling_rate={config.sampling_rate_hz}Hz, "
                f"expected_samples={config.expected_sample_count}, "
                f"AMI: bins={config.ami_n_bins}, max_lag={config.ami_max_lag}, fallback={config.ami_fallback_tau}, "
                f"FNN: R_tol={config.fnn_r_tol}, W=tau, max_dim={config.fnn_max_dim}, threshold={config.fnn_ratio_threshold}, "
                f"GRP: levels={config.grp_num_levels}, metric={config.grp_metric}, "
                f"GLCM: angles={config.glcm_orientations}, offsets={config.glcm_offsets}, 1-based indexing=True")

    # Clean restart handling
    existing_ids = set()
    if clean_restart:
        if features_csv.exists():
            features_csv.unlink()
        if diagnostics_csv.exists():
            diagnostics_csv.unlink()
        logger.info("Clean restart requested: removed previous partial outputs.")
    else:
        if features_csv.exists() and diagnostics_csv.exists():
            with open(features_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row:
                        existing_ids.add(row[0])
            logger.info(f"Resuming batch: {len(existing_ids)} recordings already completed.")

    # Prepare CSV headers
    feature_header = [
        "recording_id",
        "filename",
        "subset",
        "class_label",
        "case1_label",
        "case2_label",
        "case3_label"
    ]
    # We will determine exact feature column names from the first processed recording
    feature_columns = None

    diag_header = [
        "recording_id",
        "filename",
        "subset",
        "class_label",
        "raw_sample_count",
        "raw_sha256_before",
        "raw_sha256_after",
        "is_file_unmodified",
        "raw_mean",
        "raw_std",
        "norm_mean",
        "norm_std",
        "tau_estimated",
        "tau_is_fallback",
        "tau_selection_mode",
        "m_estimated",
        "m_converged",
        "m_selected_fnn_ratio",
        "m_is_below_threshold",
        "m_selection_mode",
        "trajectory_length",
        "feature_count",
        "nan_count",
        "inf_count",
        "elapsed_seconds",
        "status"
    ]

    # Open CSV files for incremental append
    write_feat_header = not features_csv.exists()
    write_diag_header = not diagnostics_csv.exists()

    f_feat = open(features_csv, "a", newline="", encoding="utf-8")
    f_diag = open(diagnostics_csv, "a", newline="", encoding="utf-8")
    writer_feat = csv.writer(f_feat)
    writer_diag = csv.writer(f_diag)

    if write_diag_header:
        writer_diag.writerow(diag_header)
        f_diag.flush()

    start_time_all = time.time()
    processed_count = len(existing_ids)
    failed_count = 0
    tau_fallback_count = 0
    fnn_non_converged_count = 0
    per_subset_processed = {s: 0 for s in ["A", "B", "C", "D", "E"]}

    logger.info(f"Starting sequential processing of {len(recordings) - len(existing_ids)} pending recordings...")

    for idx, rec in enumerate(recordings, start=1):
        rec_id = rec["recording_id"]
        if rec_id in existing_ids:
            per_subset_processed[rec["subset"]] += 1
            continue

        filepath = rec["filepath"]
        subset = rec["subset"]
        meta = rec["meta"]

        t0 = time.time()

        # GATE 2: SHA-256 pre-verification
        sha_before = calculate_file_sha256(filepath)

        try:
            # Process single recording
            res = preprocess_recording(filepath, config=config)

            # Verification of immutability
            sha_after = res["sha256_input"]
            if sha_before != sha_after or not res["is_file_unmodified"]:
                raise RuntimeError(f"Immutability check failed for {filepath.name}! Hash mismatch.")

            # Validate outputs
            if res["feature_count"] != 96:
                raise ValueError(f"Expected 96 features, got {res['feature_count']} in {filepath.name}")
            if res["nan_count"] > 0 or res["inf_count"] > 0:
                raise ValueError(f"Non-finite values detected in {filepath.name}: NaNs={res['nan_count']}, Infs={res['inf_count']}")

            # Setup feature header on first item if not written
            if feature_columns is None:
                feature_columns = list(res["features"].keys())
                if write_feat_header:
                    writer_feat.writerow(feature_header + feature_columns)
                    f_feat.flush()
                    write_feat_header = False

            # Write feature row
            feat_row = [
                rec_id,
                rec["filename"],
                subset,
                meta["class_label"],
                meta["case1_label"],
                meta["case2_label"],
                meta["case3_label"] if meta["case3_label"] is not None else "",
            ] + [res["features"][col] for col in feature_columns]
            writer_feat.writerow(feat_row)
            f_feat.flush()

            # Write diagnostics row
            diag_row = [
                rec_id,
                rec["filename"],
                subset,
                meta["class_label"],
                res["signal_length"],
                sha_before,
                sha_after,
                res["is_file_unmodified"],
                round(res["raw_mean"], 4),
                round(res["raw_std"], 4),
                round(res["norm_mean"], 6),
                round(res["norm_std"], 6),
                res["tau_estimated"],
                res["tau_is_fallback"],
                res["tau_selection_mode"],
                res["m_estimated"],
                res["m_converged"],
                round(res["m_selected_fnn_ratio"], 6),
                res["m_is_below_threshold"],
                res["m_selection_mode"],
                res["trajectory_shape"][0],
                res["feature_count"],
                res["nan_count"],
                res["inf_count"],
                res["elapsed_seconds"],
                "SUCCESS"
            ]
            writer_diag.writerow(diag_row)
            f_diag.flush()

            # Update stats
            processed_count += 1
            per_subset_processed[subset] += 1
            if res["tau_is_fallback"]:
                tau_fallback_count += 1
            if not res["m_converged"]:
                fnn_non_converged_count += 1

            # Periodic logging
            if processed_count % 25 == 0 or processed_count == 500 or idx <= 5:
                elapsed_total = time.time() - start_time_all
                rate = processed_count / elapsed_total if elapsed_total > 0 else 0
                rss_mb = get_peak_rss_mb()
                logger.info(
                    f"[{processed_count:03d}/500 ({processed_count/500*100:5.1f}%)] "
                    f"Subset {subset} ({rec_id}): tau={res['tau_estimated']}(mode={res['tau_selection_mode']}), "
                    f"m={res['m_estimated']}(conv={res['m_converged']}), time={res['elapsed_seconds']:.2f}s | "
                    f"Rate: {rate:.2f} rec/s | Peak RSS: {rss_mb:.1f} MB"
                )

            # Explicit memory safeguard: delete result dictionary and clear local references
            del res
            if idx % 50 == 0:
                gc.collect()

        except Exception as e:
            failed_count += 1
            logger.error(f"FAILURE on recording {rec_id} ({filepath}): {e}", exc_info=True)
            # Record failure in diagnostics
            diag_row = [
                rec_id,
                rec["filename"],
                subset,
                meta["class_label"],
                0,
                sha_before,
                "ERROR",
                False,
                0.0, 0.0, 0.0, 0.0,
                0, False, "error",
                0, False, 0.0, False, "error",
                0, 0, 0, 0,
                round(time.time() - t0, 3),
                f"FAILED: {str(e)}"
            ]
            writer_diag.writerow(diag_row)
            f_diag.flush()
            raise RuntimeError(f"Halting batch processing due to unexpected error in {rec_id}: {e}") from e

    f_feat.close()
    f_diag.close()

    total_duration = time.time() - start_time_all
    peak_rss_mb = get_peak_rss_mb()

    logger.info("=" * 80)
    logger.info("BATCH PREPROCESSING EXECUTION COMPLETED")
    logger.info(f"Total Recordings Processed: {processed_count}/500")
    logger.info(f"Per-Subset Counts: {per_subset_processed}")
    logger.info(f"Failures: {failed_count}")
    logger.info(f"AMI Fallbacks: {tau_fallback_count}")
    logger.info(f"FNN Non-Convergence Cases: {fnn_non_converged_count}")
    logger.info(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    logger.info(f"Peak Process RSS: {peak_rss_mb:.2f} MB")
    logger.info("=" * 80)

    # GATE 3: Post-Processing Validation
    logger.info("Executing Post-Processing Validation Gates...")
    if processed_count != 500:
        raise ValueError(f"Post-processing check failed: processed {processed_count} != 500")
    if failed_count != 0:
        raise ValueError(f"Post-processing check failed: {failed_count} recordings failed")
    for s, cnt in per_subset_processed.items():
        if cnt != 100:
            raise ValueError(f"Subset {s} has {cnt} recordings (expected exactly 100)")

    # Read back feature table to verify row counts, column counts, and zero NaNs
    with open(features_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        feat_headers = next(reader)
        feat_rows = list(reader)

    if len(feat_rows) != 500:
        raise ValueError(f"Feature table row count mismatch: {len(feat_rows)} != 500")
    if len(feat_headers) != 7 + 96:
        raise ValueError(f"Feature table column count mismatch: {len(feat_headers)} != 103 (7 metadata + 96 features)")

    # Read back diagnostics table to verify row counts and match
    with open(diagnostics_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        diag_headers = next(reader)
        diag_rows = list(reader)

    if len(diag_rows) != 500:
        raise ValueError(f"Diagnostics table row count mismatch: {len(diag_rows)} != 500")

    # Verify ID alignment between feature table and diagnostics table
    feat_ids = [r[0] for r in feat_rows]
    diag_ids = [r[0] for r in diag_rows]
    if feat_ids != diag_ids:
        raise ValueError("Recording ID ordering mismatch between feature table and diagnostics table!")

    logger.info("ALL VALIDATION GATES PASSED SUCCESSFULLY.")

    # Save Run Summary JSON
    summary_data = {
        "status": "SUCCESS",
        "start_time_iso": datetime.fromtimestamp(start_time_all, timezone.utc).isoformat(),
        "end_time_iso": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(total_duration, 2),
        "total_recordings_expected": 500,
        "total_recordings_processed": processed_count,
        "total_recordings_failed": failed_count,
        "per_subset_counts": per_subset_processed,
        "ami_fallback_count": tau_fallback_count,
        "fnn_non_converged_count": fnn_non_converged_count,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
        },
        "output_files": {
            "features_csv": str(features_csv),
            "diagnostics_csv": str(diagnostics_csv),
            "log_file": str(log_file),
            "summary_json": str(summary_json)
        }
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved run summary to {summary_json}")
    return summary_data


if __name__ == "__main__":
    run_batch_preprocessing()
