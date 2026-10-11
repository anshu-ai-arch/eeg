"""
Unit tests for end-to-end preprocessing pipeline.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from scripts.preprocessing.config import PreprocessingConfig
from scripts.preprocessing.pipeline import preprocess_recording


def test_pipeline_on_synthetic_recording():
    # Create temporary synthetic EEG file of 4,097 samples
    np.random.seed(123)
    t = np.linspace(0, 20 * np.pi, 4097)
    synthetic_signal = (50.0 * np.sin(t) + 10.0 * np.cos(3 * t) + np.random.randn(4097)).astype(np.int32)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        tmp_path = Path(f.name)
        for val in synthetic_signal:
            f.write(f"{val}\n")

    try:
        cfg = PreprocessingConfig()
        result = preprocess_recording(tmp_path, config=cfg)

        assert result["is_file_unmodified"] is True
        assert result["signal_length"] == 4097
        assert result["feature_count"] == 96
        assert result["nan_count"] == 0
        assert result["inf_count"] == 0
        assert result["grp_quantized_unique_levels"] > 1
        assert result["grp_quantized_max"] <= cfg.grp_num_levels - 1
        assert len(result["features"]) == 96

        # Check diagnostic metadata
        assert "tau_is_fallback" in result
        assert isinstance(result["tau_is_fallback"], bool)
        assert result["tau_selection_mode"] in ["first_local_minimum", "fallback"]
        assert "m_converged" in result
        assert isinstance(result["m_converged"], bool)
        assert isinstance(result["m_selected_fnn_ratio"], float)
        assert isinstance(result["m_is_below_threshold"], bool)
        assert result["m_selection_mode"] in [
            "threshold_convergence", "minimum_ratio_fallback", "default_fallback"
        ]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
