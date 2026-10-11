"""
Unit tests for GRP module (phase space trajectory, distance matrix, quantization).
"""

import numpy as np
import pytest
from scripts.preprocessing.grp import reconstruct_phase_space, compute_grp, quantize_grp


def test_reconstruct_phase_space():
    N = 100
    signal = np.arange(N, dtype=np.float64)
    tau = 5
    m = 3
    # N_t = N - (m - 1) * tau = 100 - 2 * 5 = 90
    trajectory = reconstruct_phase_space(signal, tau=tau, m=m)

    assert trajectory.shape == (90, 3)
    assert np.allclose(trajectory[0], [0.0, 5.0, 10.0])
    assert np.allclose(trajectory[-1], [89.0, 94.0, 99.0])


def test_compute_grp_symmetry_and_diagonal():
    trajectory = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    grp = compute_grp(trajectory, metric="euclidean")
    assert grp.shape == (3, 3)
    # Diagonal must be 0
    assert np.allclose(np.diag(grp), 0.0)
    # Matrix must be symmetric
    assert np.allclose(grp, grp.T)
    # Distance between (0, 0) and (1, 0) is 1.0
    assert np.isclose(grp[0, 1], 1.0)


def test_quantize_grp_bounds():
    grp = np.array([
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 2.5],
        [10.0, 2.5, 0.0]
    ], dtype=np.float32)

    K = 16
    q = quantize_grp(grp, num_levels=K)
    assert q.shape == (3, 3)
    assert q.dtype == np.uint8
    assert q.min() >= 0
    assert q.max() <= K - 1
    # Max value 10.0 maps to K - 1 = 15
    assert q[0, 2] == 15
    # Min value 0.0 maps to 0
    assert q[0, 0] == 0


def test_quantize_grp_zero_maximum():
    # Zero matrix must return zero uint8 matrix without division-by-zero error
    grp_zeros = np.zeros((5, 5), dtype=np.float32)
    q = quantize_grp(grp_zeros, num_levels=16)

    assert q.shape == (5, 5)
    assert q.dtype == np.uint8
    assert np.all(q == 0)


def test_quantize_grp_discrete_levels():
    # Array with values spanning 0 to 15 uniformly
    vals = np.linspace(0.0, 100.0, 16, dtype=np.float32)
    grp_linear = np.outer(vals, np.ones(16, dtype=np.float32))
    q = quantize_grp(grp_linear, num_levels=16)

    assert q.dtype == np.uint8
    assert q.min() == 0
    assert q.max() == 15
    # Verify exact set of levels produced
    unique_levels = np.unique(q)
    assert len(unique_levels) == 16
    assert np.array_equal(unique_levels, np.arange(16, dtype=np.uint8))
