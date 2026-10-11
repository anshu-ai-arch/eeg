"""
Unit tests for embedding module (AMI and FNN).
"""

import numpy as np
import pytest
from scripts.preprocessing.embedding import compute_ami, compute_fnn


def test_compute_ami_sine():
    # Pure sine wave has clear mutual information minima at quarter/half cycle
    t = np.linspace(0, 8 * np.pi, 1000)
    signal = np.sin(t)
    tau, curve = compute_ami(signal, max_lag=30, n_bins=16)

    assert isinstance(tau, int)
    assert 1 <= tau <= 30
    assert len(curve) == 30
    assert not np.any(np.isnan(curve))


def test_compute_ami_monotonic_fallback():
    # Strictly decreasing AMI curve over first 5 lags has no local minimum
    signal = np.linspace(0, 100, 200)
    tau, curve = compute_ami(signal, max_lag=5, n_bins=10, fallback_tau=7)
    assert tau == 7


def test_compute_fnn_shape_and_bounds():
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 20, 500)) + 0.05 * np.random.randn(500)
    m, ratios = compute_fnn(signal, tau=5, max_dim=6, r_tol=15.0, threshold=0.01)

    assert isinstance(m, int)
    assert 1 <= m <= 6
    assert len(ratios) > 0
    assert all(0.0 <= r <= 1.0 for r in ratios)


def test_compute_ami_diagnostics():
    # 1. Non-fallback case (sine wave)
    t = np.linspace(0, 8 * np.pi, 1000)
    signal_sine = np.sin(t)
    tau, curve, diag = compute_ami(signal_sine, max_lag=30, n_bins=16, return_diagnostics=True)
    assert diag["tau_is_fallback"] is False
    assert diag["tau_selection_mode"] == "first_local_minimum"

    # 2. Fallback case (strictly monotonic ramp)
    signal_ramp = np.linspace(0, 100, 200)
    tau_fb, curve_fb, diag_fb = compute_ami(signal_ramp, max_lag=5, n_bins=10, fallback_tau=7, return_diagnostics=True)
    assert tau_fb == 7
    assert diag_fb["tau_is_fallback"] is True
    assert diag_fb["tau_selection_mode"] == "fallback"


def test_compute_fnn_diagnostics():
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 20, 500)) + 0.05 * np.random.randn(500)

    # 1. Normal convergence case
    m, ratios, diag = compute_fnn(signal, tau=5, max_dim=6, r_tol=15.0, threshold=0.05, return_diagnostics=True)
    assert diag["m_converged"] is True
    assert diag["m_selection_mode"] == "threshold_convergence"
    assert diag["m_is_below_threshold"] is True
    assert isinstance(diag["m_selected_fnn_ratio"], float)
    assert diag["m_selected_fnn_ratio"] < 0.05

    # 2. Fallback case when threshold cannot be reached (impossible threshold)
    m_fb, ratios_fb, diag_fb = compute_fnn(signal, tau=5, max_dim=3, r_tol=15.0, threshold=1e-8, return_diagnostics=True)
    assert diag_fb["m_converged"] is False
    assert diag_fb["m_selection_mode"] in ["minimum_ratio_fallback", "default_fallback"]
    assert diag_fb["m_is_below_threshold"] is False
    assert isinstance(diag_fb["m_selected_fnn_ratio"], float)

