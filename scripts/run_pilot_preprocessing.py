#!/usr/bin/env python3
"""
run_pilot_preprocessing.py

Executes Phase 5: Single-Recording Pilot and Forensic Verification on a single EEG recording (Z001.txt).
Strictly verifies:
- Input file SHA-256 before and after (zero tampering).
- Intermediate tensor/matrix shapes.
- Estimated tau (AMI) and m (FNN).
- Quantization levels and GLCM properties.
- NaN / Inf checks on the complete 96-feature vector.
- Generates pilot diagnostic report and visual validation figures.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.preprocessing.config import PreprocessingConfig
from scripts.preprocessing.pipeline import preprocess_recording


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pilot_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def generate_pilot_figure(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

    # Subplot 1: AMI Curve
    ami = result["ami_curve"]
    lags = np.arange(1, len(ami) + 1)
    tau = result["tau_estimated"]
    axes[0, 0].plot(lags, ami, "b.-", label="AMI Curve (32 bins)")
    axes[0, 0].axvline(tau, color="red", linestyle="--", linewidth=1.5, label=f"Optimal tau = {tau}")
    axes[0, 0].plot([tau], [ami[tau - 1]], "ro", markersize=8)
    axes[0, 0].set_title(f"Average Mutual Information (AMI) vs Lag (Z001.txt)\nEstimated tau = {tau}", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Time Lag (samples)")
    axes[0, 0].set_ylabel("AMI (bits)")
    axes[0, 0].legend(frameon=True)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: FNN Ratios
    ratios = result["fnn_ratios"]
    dims = np.arange(1, len(ratios) + 1)
    m = result["m_estimated"]
    axes[0, 1].plot(dims, ratios, "g.-", label="FNN Ratio (R_tol=15)")
    axes[0, 1].axhline(0.01, color="red", linestyle="--", linewidth=1.5, label="1% Threshold (0.01)")
    axes[0, 1].axvline(m, color="purple", linestyle=":", linewidth=1.5, label=f"Selected m = {m}")
    axes[0, 1].plot([m], [ratios[m - 1]], "mo", markersize=8)
    axes[0, 1].set_title(f"False Nearest Neighbours (FNN) vs Dimension (Z001.txt)\nSelected m = {m}", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Embedding Dimension m")
    axes[0, 1].set_ylabel("FNN Ratio")
    axes[0, 1].legend(frameon=True)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Quantized GRP Preview
    glcm_sample = result["glcm_matrices"]["theta_0deg_d_1"]
    im1 = axes[1, 0].imshow(glcm_sample, cmap="viridis", origin="lower")
    axes[1, 0].set_title("Sample GLCM (theta = 0°, d = 1, 16x16)\nNormalized Co-occurrence Probabilities", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Gray Level j (0 to 15)")
    axes[1, 0].set_ylabel("Gray Level i (0 to 15)")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Subplot 4: Extracted Haralick Feature Values
    feat_names = list(result["features"].keys())
    feat_vals = list(result["features"].values())
    # Plot feature 1 (energy) across the 8 GLCMs as a representative contrast
    energy_keys = [k for k in feat_names if "energy" in k]
    contrast_keys = [k for k in feat_names if "contrast" in k]
    homog_keys = [k for k in feat_names if "homogeneity" in k]

    labels = ["a0_d1", "a0_d2", "a0_d3", "a0_d4", "a45_d1", "a45_d2", "a45_d3", "a45_d4"]
    x_idx = np.arange(8)
    width = 0.25

    axes[1, 1].bar(x_idx - width, [result["features"][k] for k in energy_keys], width, label="Energy (f1)", color="#337ab7")
    axes[1, 1].bar(x_idx, [result["features"][k] for k in homog_keys], width, label="Homogeneity (f4)", color="#5cb85c")
    axes[1, 1].set_xticks(x_idx)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[1, 1].set_title("Representative Haralick Features Across 8 GLCMs (Z001.txt)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Feature Value")
    axes[1, 1].legend(frameon=True)
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent.parent
    target_file = project_root / "data" / "bonn_eeg" / "extracted" / "A" / "Z001.txt"
    log_file = project_root / "audit" / "logs" / "pilot_recording.log"
    fig_file = project_root / "audit" / "reports" / "figures" / "pilot_z001_diagnostics.png"
    csv_file = project_root / "audit" / "tables" / "pilot_z001_features.csv"

    logger = setup_logger(log_file)
    logger.info("=" * 60)
    logger.info("Starting Phase 5: Single-Recording Preprocessing Pilot on Z001.txt")
    logger.info(f"Target recording: {target_file}")

    config = PreprocessingConfig()
    result = preprocess_recording(target_file, config=config)

    logger.info("=== VERIFICATION RESULTS ===")
    logger.info(f"Input SHA-256 before & after match: {result['is_file_unmodified']} (SHA: {result['sha256_input']})")
    logger.info(f"Raw sample count: {result['signal_length']}")
    logger.info(f"Raw mean: {result['raw_mean']:.4f}, raw std: {result['raw_std']:.4f}")
    logger.info(f"Normalized mean: {result['norm_mean']:.6f}, normalized std: {result['norm_std']:.6f}")
    logger.info(f"Estimated delay tau: {result['tau_estimated']} samples ({result['tau_estimated'] / config.sampling_rate_hz * 1000:.2f} ms) | fallback_used: {result['tau_is_fallback']} | mode: {result['tau_selection_mode']}")
    logger.info(f"Estimated dimension m: {result['m_estimated']} | mode: {result['m_selection_mode']} | selected_fnn_ratio: {result['m_selected_fnn_ratio']:.4f} | below_1%: {result['m_is_below_threshold']}")
    logger.info(f"Phase-space trajectory shape: {result['trajectory_shape']}")
    logger.info(f"Continuous GRP shape: {result['grp_continuous_shape']} (range: [{result['grp_continuous_min']:.2f}, {result['grp_continuous_max']:.2f}])")
    logger.info(f"Quantized GRP shape: {result['grp_quantized_shape']} (levels: [{result['grp_quantized_min']}, {result['grp_quantized_max']}], unique: {result['grp_quantized_unique_levels']})")
    logger.info(f"GLCM count: {result['num_glcms']} (orientations: {config.glcm_orientations}, offsets: {config.glcm_offsets})")
    logger.info(f"Extracted feature count: {result['feature_count']} (expected: 96)")
    logger.info(f"NaN count: {result['nan_count']}, Inf count: {result['inf_count']}")
    logger.info(f"Execution time: {result['elapsed_seconds']}s")

    # Save feature CSV
    import csv
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "feature_value"])
        for k, v in result["features"].items():
            writer.writerow([k, v])
    logger.info(f"Saved 96-feature vector to {csv_file}")

    # Generate diagnostic plot
    generate_pilot_figure(result, fig_file)
    logger.info(f"Saved diagnostic plots to {fig_file}")

    logger.info("Phase 5 Single-Recording Pilot completed successfully.")


if __name__ == "__main__":
    main()
