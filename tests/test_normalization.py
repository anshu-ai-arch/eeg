"""
Unit tests for z-score normalization module.
"""

import numpy as np
import pytest
from scripts.preprocessing.normalization import zscore_normalize


def test_zscore_normalize_population():
    signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    orig_copy = signal.copy()
    norm, mean_v, std_v = zscore_normalize(signal, ddof=0)

    # Immutability
    assert np.array_equal(signal, orig_copy)
    # Mean and std
    assert np.isclose(mean_v, 30.0)
    assert np.isclose(std_v, np.std(signal, ddof=0))
    assert np.isclose(np.mean(norm), 0.0)
    assert np.isclose(np.std(norm, ddof=0), 1.0)


def test_zscore_constant_signal_guard():
    constant_signal = np.array([5.0, 5.0, 5.0, 5.0])
    norm, mean_v, std_v = zscore_normalize(constant_signal, ddof=0, epsilon=1e-12)
    assert np.isclose(mean_v, 5.0)
    assert std_v < 1e-12
    # Guard prevents division by zero
    assert not np.any(np.isnan(norm))
    assert not np.any(np.isinf(norm))


def test_zscore_invalid_input():
    with pytest.raises(ValueError, match="1-dimensional"):
        zscore_normalize(np.ones((5, 5)))

    with pytest.raises(ValueError, match="empty"):
        zscore_normalize(np.array([]))
