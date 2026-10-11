"""
embedding.py

Time delay (tau) and embedding dimension (m) estimation for Takens phase-space reconstruction.
- tau: First local minimum of Average Mutual Information (AMI) with 32 equi-width bins.
- m: False Nearest Neighbours (FNN) criterion stopping when FNN ratio < 1%.
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_ami(
    signal: np.ndarray,
    max_lag: int = 50,
    n_bins: int = 32,
    fallback_tau: int = 10,
    return_diagnostics: bool = False
) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, dict]:
    """
    Compute Average Mutual Information (AMI) curve and determine first local minimum tau.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal array.
    max_lag : int, default=50
        Maximum time lag to evaluate.
    n_bins : int, default=32
        Number of equi-width histogram bins.
    fallback_tau : int, default=10
        Fallback lag if no local minimum is found in range [1, max_lag].
    return_diagnostics : bool, default=False
        If True, returns an additional diagnostic dictionary containing fallback tracking.

    Returns
    -------
    tau : int
        Estimated optimal time delay.
    ami_curve : np.ndarray
        Computed AMI values for lags 1 through max_lag.
    diagnostics : dict (optional, returned if return_diagnostics=True)
        Dictionary containing 'tau_is_fallback' (bool) and 'tau_selection_mode' (str).
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    if len(signal) <= max_lag:
        raise ValueError(f"Signal length {len(signal)} must be greater than max_lag {max_lag}")

    amis = []
    # Fixed bin edges across the full range of the signal
    bins = np.linspace(signal.min(), signal.max(), n_bins + 1)

    for lag in range(1, max_lag + 1):
        x1 = signal[:-lag]
        x2 = signal[lag:]
        hist2d, _, _ = np.histogram2d(x1, x2, bins=[bins, bins])

        total = np.sum(hist2d)
        if total == 0:
            amis.append(0.0)
            continue

        p_xy = hist2d / total
        p_x = np.sum(p_xy, axis=1, keepdims=True)
        p_y = np.sum(p_xy, axis=0, keepdims=True)

        mask = p_xy > 0
        denom = (p_x @ p_y)[mask]
        valid_mask = denom > 0

        p_xy_valid = p_xy[mask][valid_mask]
        denom_valid = denom[valid_mask]

        ami_val = np.sum(p_xy_valid * np.log2(p_xy_valid / denom_valid))
        amis.append(float(ami_val))

    ami_curve = np.array(amis, dtype=np.float64)

    # Find first local minimum
    tau_opt = None
    for i in range(1, len(ami_curve) - 1):
        if ami_curve[i] < ami_curve[i - 1] and ami_curve[i] < ami_curve[i + 1]:
            tau_opt = i + 1  # 1-indexed lag
            break

    tau_is_fallback = (tau_opt is None)
    if tau_is_fallback:
        # Check monotonic decay or boundary
        tau_opt = fallback_tau

    if return_diagnostics:
        diag = {
            "tau_is_fallback": tau_is_fallback,
            "tau_selection_mode": "fallback" if tau_is_fallback else "first_local_minimum"
        }
        return tau_opt, ami_curve, diag

    return tau_opt, ami_curve


def compute_fnn(
    signal: np.ndarray,
    tau: int,
    max_dim: int = 10,
    r_tol: float = 15.0,
    a_tol: float = 2.0,
    theiler_w: int | None = None,
    threshold: float = 0.01,
    fallback_m: int = 5,
    return_diagnostics: bool = False
) -> tuple[int, list[float]] | tuple[int, list[float], dict]:
    """
    Determine embedding dimension m using False Nearest Neighbours (FNN).

    Replication Implementation Note:
    Applies Kennel et al. (1992) Criterion 1 (relative distance increase ratio R_tol).
    The second criterion (A_tol, attractor size) is deliberately not applied to prevent
    spurious high-dimensional penalties on finite noisy EEG data.
    a_tol is retained in the function signature for compatibility but is marked as UNUSED.

    Parameters
    ----------
    signal : np.ndarray
        1D normalized EEG signal array.
    tau : int
        Time delay lag.
    max_dim : int, default=10
        Maximum dimension m to test.
    r_tol : float, default=15.0
        Distance ratio increase threshold for false neighbor classification (Criterion 1).
    a_tol : float, default=2.0
        DEPRECATED/UNUSED: Retained for signature compatibility; not applied.
    theiler_w : int or None, default=None
        Theiler window width to exclude temporal autocorrelation neighbors. If None, set to tau.
    threshold : float, default=0.01
        FNN ratio stopping threshold (1% = 0.01).
    fallback_m : int, default=5
        Fallback dimension if ratio does not drop below threshold.
    return_diagnostics : bool, default=False
        If True, returns an additional diagnostic dictionary containing convergence tracking.

    Returns
    -------
    m_opt : int
        Optimal embedding dimension.
    ratios : list[float]
        FNN ratio calculated for each evaluated dimension.
    diagnostics : dict (optional, returned if return_diagnostics=True)
        Dictionary containing 'm_converged' (bool), 'm_selected_fnn_ratio' (float),
        'm_is_below_threshold' (bool), and 'm_selection_mode' (str).
    """
    if theiler_w is None:
        theiler_w = tau

    ratios = []
    m_opt = None

    for d in range(1, max_dim + 1):
        N_d = len(signal) - d * tau
        if N_d <= 2 * theiler_w + 5:
            break

        X_d = np.column_stack([signal[i * tau : i * tau + N_d] for i in range(d)])
        tree = cKDTree(X_d)

        k_query = min(2 * theiler_w + 5, len(X_d))
        dists, idxs = tree.query(X_d, k=k_query)

        false_count = 0
        valid_count = 0

        for i in range(len(X_d)):
            mask = np.abs(idxs[i] - i) > theiler_w
            if not np.any(mask):
                continue
            first_valid_pos = np.where(mask)[0][0]
            nn_idx = idxs[i, first_valid_pos]
            dist_d = dists[i, first_valid_pos]

            if dist_d < 1e-12:
                continue

            valid_count += 1
            diff_next = abs(signal[i + d * tau] - signal[nn_idx + d * tau])

            # Kennel distance ratio criterion
            if (diff_next / dist_d) > r_tol:
                false_count += 1

        ratio = (false_count / valid_count) if valid_count > 0 else 0.0
        ratios.append(float(ratio))

        if ratio < threshold:
            m_opt = d
            break

    if m_opt is not None:
        selection_mode = "threshold_convergence"
        m_converged = True
        selected_ratio = ratios[m_opt - 1]
    else:
        m_converged = False
        min_dim = int(np.argmin(ratios)) + 1 if ratios else fallback_m
        if min_dim > 1:
            m_opt = min_dim
            selection_mode = "minimum_ratio_fallback"
            selected_ratio = ratios[m_opt - 1]
        else:
            m_opt = fallback_m
            selection_mode = "default_fallback"
            selected_ratio = ratios[m_opt - 1] if m_opt <= len(ratios) else None

    if return_diagnostics:
        diag = {
            "m_converged": m_converged,
            "m_selected_fnn_ratio": selected_ratio,
            "m_is_below_threshold": bool(selected_ratio is not None and selected_ratio < threshold),
            "m_selection_mode": selection_mode,
        }
        return m_opt, ratios, diag

    return m_opt, ratios
