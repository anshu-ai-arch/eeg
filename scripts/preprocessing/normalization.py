"""
normalization.py

Per-recording z-score normalization for EEG signals.
Adheres strictly to the user-approved decision:
- Population standard deviation (ddof=0, denominator N).
- Epsilon guard against zero-variance signals.
- Immutable handling: original array is never modified in-place.
"""

import numpy as np


def zscore_normalize(signal: np.ndarray, ddof: int = 0, epsilon: float = 1e-12) -> tuple[np.ndarray, float, float]:
    """
    Apply per-recording z-score normalization to an EEG signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array of raw EEG signal samples.
    ddof : int, default=0
        Delta degrees of freedom for standard deviation (0 for population, 1 for sample).
    epsilon : float, default=1e-12
        Small constant added to denominator if standard deviation is below threshold.

    Returns
    -------
    normalized : np.ndarray
        Z-score normalized 1D float64 array.
    mean_val : float
        Calculated mean of the input signal.
    std_val : float
        Calculated standard deviation of the input signal.
    """
    if signal.ndim != 1:
        raise ValueError(f"Input signal must be 1-dimensional, got shape {signal.shape}")
    if len(signal) == 0:
        raise ValueError("Input signal is empty")

    signal_f = signal.astype(np.float64, copy=True)
    mean_val = float(np.mean(signal_f))
    std_val = float(np.std(signal_f, ddof=ddof))

    if std_val < epsilon:
        # Constant or near-constant signal guard
        denom = epsilon
    else:
        denom = std_val

    normalized = (signal_f - mean_val) / denom
    return normalized, mean_val, std_val
