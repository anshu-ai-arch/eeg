"""
pipeline.py

Orchestrates the full preprocessing and feature extraction pipeline on a single EEG recording.
Maintains strict reproducibility, traceability, and immutability.
"""

import time
import hashlib
from pathlib import Path
import numpy as np

from .config import PreprocessingConfig
from .normalization import zscore_normalize
from .embedding import compute_ami, compute_fnn
from .grp import reconstruct_phase_space, compute_grp, quantize_grp
from .glcm import extract_all_glcm_features


def calculate_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def preprocess_recording(
    filepath: Path,
    config: PreprocessingConfig = PreprocessingConfig()
) -> dict:
    """
    Run the complete preprocessing pipeline on a single raw EEG recording.

    Parameters
    ----------
    filepath : Path
        Path to the ASCII text file containing 4,097 integer EEG samples.
    config : PreprocessingConfig
        Explicit configuration dataclass.

    Returns
    -------
    result : dict
        Comprehensive audit dictionary containing intermediate array shapes,
        estimated parameters (tau, m), diagnostic curves, quality checks,
        and the 96-feature extracted vector.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Recording file not found: {filepath}")

    # Immutability check - compute SHA256 before loading
    sha_before = calculate_file_sha256(filepath)
    start_time = time.time()

    # Step 1: Load raw signal
    raw_signal = np.loadtxt(filepath, dtype=np.float64)
    if len(raw_signal) != config.expected_sample_count:
        raise ValueError(
            f"Expected {config.expected_sample_count} samples, got {len(raw_signal)} in {filepath.name}"
        )

    # Step 2: Per-recording Z-score normalization
    norm_signal, mean_val, std_val = zscore_normalize(
        raw_signal, ddof=config.norm_ddof, epsilon=config.norm_epsilon
    )

    # Step 3: Time delay tau via AMI first local minimum
    tau_opt, ami_curve, ami_diag = compute_ami(
        norm_signal,
        max_lag=config.ami_max_lag,
        n_bins=config.ami_n_bins,
        fallback_tau=config.ami_fallback_tau,
        return_diagnostics=True
    )

    # Step 4: Embedding dimension m via False Nearest Neighbours (< 1%)
    theiler_w = tau_opt if config.fnn_theiler_window else 1
    m_opt, fnn_ratios, fnn_diag = compute_fnn(
        norm_signal,
        tau=tau_opt,
        max_dim=config.fnn_max_dim,
        r_tol=config.fnn_r_tol,
        a_tol=config.fnn_a_tol,
        theiler_w=theiler_w,
        threshold=config.fnn_ratio_threshold,
        fallback_m=config.fnn_fallback_m,
        return_diagnostics=True
    )

    # Step 5: Phase-space trajectory reconstruction
    trajectory = reconstruct_phase_space(norm_signal, tau=tau_opt, m=m_opt)
    N_t = trajectory.shape[0]

    # Step 6: Continuous Generalized Recurrence Plot (GRP)
    grp_continuous = compute_grp(trajectory, metric=config.grp_metric)

    # Step 7: Quantize GRP to K gray levels
    grp_quantized = quantize_grp(grp_continuous, num_levels=config.grp_num_levels)

    # Step 8: Compute 8 GLCMs and extract 96 Haralick features
    features_dict, glcm_matrices = extract_all_glcm_features(
        grp_quantized,
        orientations=config.glcm_orientations,
        offsets=config.glcm_offsets,
        num_levels=config.grp_num_levels + 1,
        eps=config.log_epsilon
    )

    elapsed_time = time.time() - start_time

    # Verify input file immutability after processing
    sha_after = calculate_file_sha256(filepath)
    is_file_unmodified = (sha_before == sha_after)

    # Quality checks
    feature_values = np.array(list(features_dict.values()), dtype=np.float64)
    nan_count = int(np.sum(np.isnan(feature_values)))
    inf_count = int(np.sum(np.isinf(feature_values)))

    return {
        "filename": filepath.name,
        "filepath": str(filepath),
        "sha256_input": sha_before,
        "is_file_unmodified": is_file_unmodified,
        "signal_length": len(raw_signal),
        "raw_mean": mean_val,
        "raw_std": std_val,
        "norm_mean": float(np.mean(norm_signal)),
        "norm_std": float(np.std(norm_signal, ddof=config.norm_ddof)),
        "tau_estimated": tau_opt,
        "tau_is_fallback": ami_diag["tau_is_fallback"],
        "tau_selection_mode": ami_diag["tau_selection_mode"],
        "m_estimated": m_opt,
        "m_converged": fnn_diag["m_converged"],
        "m_selected_fnn_ratio": fnn_diag["m_selected_fnn_ratio"],
        "m_is_below_threshold": fnn_diag["m_is_below_threshold"],
        "m_selection_mode": fnn_diag["m_selection_mode"],
        "ami_curve": ami_curve.tolist(),
        "fnn_ratios": fnn_ratios,
        "trajectory_shape": trajectory.shape,
        "grp_continuous_shape": grp_continuous.shape,
        "grp_continuous_min": float(grp_continuous.min()),
        "grp_continuous_max": float(grp_continuous.max()),
        "grp_quantized_shape": grp_quantized.shape,
        "grp_quantized_min": int(grp_quantized.min()),
        "grp_quantized_max": int(grp_quantized.max()),
        "grp_quantized_unique_levels": int(len(np.unique(grp_quantized))),
        "num_glcms": len(glcm_matrices),
        "feature_count": len(features_dict),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "elapsed_seconds": round(elapsed_time, 3),
        "features": features_dict,
        "glcm_matrices": glcm_matrices
    }
