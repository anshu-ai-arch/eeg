"""
Bonn EEG Preprocessing Package

Implements preprocessing, phase-space reconstruction, GRP generation,
and GLCM Haralick texture feature extraction according to:
Lashkari et al., "Improved Detection of Epileptic Seizures via EEG Signals and Texture
Analysis of Recurrence Plots", Biomedical Signal Processing and Control 118 (2026) 109675.
"""

from .config import PreprocessingConfig
from .normalization import zscore_normalize
from .embedding import compute_ami, compute_fnn
from .grp import reconstruct_phase_space, compute_grp, quantize_grp
from .glcm import compute_single_glcm, extract_haralick_12, extract_all_glcm_features, HARALICK_FEATURE_NAMES
from .pipeline import preprocess_recording

__all__ = [
    "PreprocessingConfig",
    "zscore_normalize",
    "compute_ami",
    "compute_fnn",
    "reconstruct_phase_space",
    "compute_grp",
    "quantize_grp",
    "compute_single_glcm",
    "extract_haralick_12",
    "extract_all_glcm_features",
    "HARALICK_FEATURE_NAMES",
    "preprocess_recording",
]
