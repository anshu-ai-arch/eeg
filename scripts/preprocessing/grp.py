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


def quantize_grp(grp: np.ndarray, num_levels: int = 16) -> np.ndarray:
    """
    Quantize continuous recurrence distance matrix to K discrete gray levels.

    RP_quantized = round( (RP / max(RP)) * (K - 1) )
    Produces discrete integer levels in {0, 1, ..., K - 1}.

    Replication Implementation Note:
    The primary paper's printed equation (Eq. 20) multiplies by K. This implementation
    deliberately scales by (K - 1) to obtain exactly K=16 discrete levels (0 through 15)
    compatible with a 16 x 16 GLCM. This is documented as a replication decision resolving
    an ambiguity in the paper, rather than a literal implementation of the printed equation.

    Parameters
    ----------
    grp : np.ndarray
        Continuous 2D recurrence distance matrix.
    num_levels : int, default=16
        Number of discrete gray levels K (produces K x K GLCM).

    Returns
    -------
    quantized_grp : np.ndarray
        2D integer array of shape (N_t, N_t), values in [0, num_levels - 1], dtype uint8.
    """
    if grp.ndim != 2:
        raise ValueError("GRP must be 2D matrix")

    max_val = float(grp.max())
    if max_val < 1e-12:
        # Trivial zero matrix
        return np.zeros_like(grp, dtype=np.uint8)

    # Scale to [0, num_levels] and round to discrete integer levels
    scaled = (grp / max_val) * num_levels
    quantized = np.clip(np.round(scaled), 0, num_levels).astype(np.uint8)
    return quantized
