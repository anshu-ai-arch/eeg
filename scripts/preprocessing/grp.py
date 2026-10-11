"""
grp.py

Generalized Recurrence Plot (GRP) construction and gray-level quantization.
- Phase-space reconstruction via Takens' delay embedding.
- Continuous distance matrix computation (Euclidean metric).
- Quantization to K discrete gray levels (Eq. 20).
"""

import numpy as np
from scipy.spatial.distance import cdist


def reconstruct_phase_space(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Reconstruct multi-dimensional trajectory from 1D EEG signal via Takens' embedding.

    X(t) = [x(t), x(t + tau), x(t + 2*tau), ..., x(t + (m-1)*tau)]

    Parameters
    ----------
    signal : np.ndarray
        1D normalized EEG signal array of length N.
    tau : int
        Time delay lag.
    m : int
        Embedding dimension.

    Returns
    -------
    trajectory : np.ndarray
        2D trajectory array of shape (N_t, m), where N_t = N - (m - 1) * tau.
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    N = len(signal)
    N_t = N - (m - 1) * tau
    if N_t <= 0:
        raise ValueError(f"Signal length {N} is too short for dimension {m} and lag {tau}")

    trajectory = np.column_stack([signal[i * tau : i * tau + N_t] for i in range(m)])
    return trajectory.astype(np.float32)


def compute_grp(trajectory: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distance matrix GRP(i, j) = ||X(i) - X(j)||.

    Parameters
    ----------
    trajectory : np.ndarray
        2D phase-space trajectory array of shape (N_t, m).
    metric : str, default="euclidean"
        Distance metric name for scipy cdist.

    Returns
    -------
    grp : np.ndarray
        Continuous distance matrix of shape (N_t, N_t), dtype float32.
    """
    if trajectory.ndim != 2:
        raise ValueError("Trajectory must be 2D array of shape (N_t, m)")

    grp = cdist(trajectory, trajectory, metric=metric).astype(np.float32)
    return grp


def quantize_grp(
    grp: np.ndarray,
    num_levels: int = 16,
    mode: str = "k_minus_1"
) -> np.ndarray:
    """
    Quantize continuous recurrence distance matrix to discrete gray levels.

    Modes:
    1. 'k_minus_1' (Standard GLCM reproduction default):
       RP_quantized = round( (RP / max(RP)) * (K - 1) )
       Maps values into [0, K - 1], producing exactly K discrete levels (e.g. 0..15 for K=16).
       Produces a standard K x K GLCM (16 x 16).

    2. 'literal_k' (Paper Eq. 20 literal mode):
       RP_quantized = round( (RP / max(RP)) * K )
       Maps values into [0, K], producing K + 1 discrete levels (e.g. 0..16 for K=16).
       Requires a (K + 1) x (K + 1) GLCM (17 x 17).

    Parameters
    ----------
    grp : np.ndarray
        Continuous 2D recurrence distance matrix.
    num_levels : int, default=16
        Base number of gray levels K. Must be >= 2.
    mode : {'k_minus_1', 'literal_k'}, default='k_minus_1'
        Quantization scaling convention.

    Returns
    -------
    quantized_grp : np.ndarray
        2D integer array of shape (N_t, N_t), dtype uint8.
        Values are in [0, num_levels - 1] for 'k_minus_1', or [0, num_levels] for 'literal_k'.
    """
    if grp.ndim != 2:
        raise ValueError(f"GRP must be 2D matrix, got shape {grp.shape}")
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2, got {num_levels}")
    if mode not in ("k_minus_1", "literal_k"):
        raise ValueError(f"Unsupported quantization mode '{mode}'. Must be 'k_minus_1' or 'literal_k'.")

    # Guard against non-finite values (NaN / Inf)
    if not np.all(np.isfinite(grp)):
        grp = np.nan_to_num(grp, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = float(grp.max())
    if max_val < 1e-12:
        # Trivial zero or constant flatline matrix
        return np.zeros_like(grp, dtype=np.uint8)

    # Calculate discrete integer gray levels based on selected mode
    if mode == "k_minus_1":
        multiplier = num_levels - 1
        max_level = num_levels - 1
    else:  # mode == "literal_k"
        multiplier = num_levels
        max_level = num_levels

    scaled = (grp / max_val) * multiplier
    quantized = np.clip(np.round(scaled), 0, max_level).astype(np.uint8)
    return quantized


def get_quantization_metadata(num_levels: int = 16, mode: str = "k_minus_1") -> dict:
    """
    Return dictionary describing quantization metadata for provenance tracking.
    """
    if mode == "k_minus_1":
        return {
            "mode": "k_minus_1",
            "base_K": num_levels,
            "multiplier": num_levels - 1,
            "min_level": 0,
            "max_level": num_levels - 1,
            "total_discrete_levels": num_levels,
            "glcm_dimension": num_levels,
            "description": "Standard reproduction scaling round(RP / max(RP) * (K - 1)) yielding 0..K-1"
        }
    elif mode == "literal_k":
        return {
            "mode": "literal_k",
            "base_K": num_levels,
            "multiplier": num_levels,
            "min_level": 0,
            "max_level": num_levels,
            "total_discrete_levels": num_levels + 1,
            "glcm_dimension": num_levels + 1,
            "description": "Paper Eq. 20 literal scaling round(RP / max(RP) * K) yielding 0..K"
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
