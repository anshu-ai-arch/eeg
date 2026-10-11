"""
config.py

Preprocessing configuration dataclass for Bonn EEG reproduction.
All parameters represent explicitly approved scientific choices with zero hidden defaults.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    # Dataset & Signal Properties
    sampling_rate_hz: float = 173.61
    expected_sample_count: int = 4097

    # Z-Score Normalization
    norm_ddof: int = 0                  # Population standard deviation (ddof=0, denominator N)
    norm_epsilon: float = 1e-12         # Guard against zero-variance flatline

    # Average Mutual Information (AMI) for Time Delay tau
    ami_n_bins: int = 32               # Equi-width histogram bins
    ami_max_lag: int = 50              # Maximum time lag searched
    ami_fallback_tau: int = 10         # Fallback lag if curve is monotonic without local minimum

    # False Nearest Neighbours (FNN) for Embedding Dimension m
    # Replication Choice: Single-criterion FNN using relative distance increase (Criterion 1, R_tol).
    fnn_metric: str = "euclidean"      # Distance metric for phase-space vectors
    fnn_r_tol: float = 15.0            # Kennel distance ratio threshold (R_tol)
    fnn_a_tol: float = 2.0             # DEPRECATED/UNUSED: Retained for signature compatibility; only Criterion 1 is applied.
    fnn_theiler_window: bool = True    # Exclude temporal autocorrelation neighbors (|i - j| <= tau)
    fnn_max_dim: int = 10              # Search range m in [1, 10]
    fnn_ratio_threshold: float = 0.01  # Stopping threshold: FNN ratio < 1%
    fnn_fallback_m: int = 5            # Fallback dimension if < 1% not reached by m=10

    # Generalized Recurrence Plot (GRP)
    # Default 'k_minus_1': Quantized to exactly K=16 levels (0..15) using round(RP / max(RP) * (K - 1))
    # Optional 'literal_k': Quantized to K+1=17 levels (0..16) using round(RP / max(RP) * K) (Eq. 20)
    grp_metric: str = "euclidean"      # Distance metric for recurrence distance matrix
    grp_num_levels: int = 16           # Number of quantized gray levels K (produces K x K GLCMs)
    grp_quantization_mode: str = "k_minus_1"  # 'k_minus_1' (0..K-1) or 'literal_k' (0..K)

    # Gray-Level Co-occurrence Matrix (GLCM) - Section 3.5 Ground Truth
    # Orientations: 0 and pi/4 (0° and 45°)
    glcm_orientations: tuple[float, ...] = (0.0, np.pi / 4.0)
    # Offsets: 1, 2, 3, 4
    glcm_offsets: tuple[int, ...] = (1, 2, 3, 4)

    # Haralick Feature Extraction
    use_1_based_indexing: bool = True  # Gray levels i, j in {1, ..., Ng} matching Eqs. (6)-(19)
    log_epsilon: float = 1e-12         # Epsilon in Entropy formula: log2(p(i,j) + epsilon)
