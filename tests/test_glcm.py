"""
Unit tests for GLCM and Haralick texture feature extraction.
"""

import numpy as np
import pytest
from scripts.preprocessing.glcm import (
    compute_single_glcm,
    extract_haralick_12,
    extract_all_glcm_features,
    HARALICK_FEATURE_NAMES
)


def test_glcm_normalization_and_symmetry():
    img = np.random.randint(0, 16, (50, 50), dtype=np.uint8)
    for theta in [0.0, np.pi / 4.0]:
        for d in [1, 2, 3, 4]:
            glcm = compute_single_glcm(img, offset=d, theta=theta, num_levels=16, symmetric=True)
            assert glcm.shape == (16, 16)
            # Verify sum is 1.0 within documented tolerance of 1e-7
            assert np.isclose(np.sum(glcm), 1.0, atol=1e-7)
            assert np.allclose(glcm, glcm.T)


def test_haralick_identity_matrix():
    # An identity matrix represents perfect correlation and homogeneity
    K = 16
    p_ident = np.eye(K, dtype=np.float64) / K
    feats = extract_haralick_12(p_ident, num_levels=K)

    assert len(feats) == 12
    # Energy should be sum( (1/16)^2 * 16 ) = 1/16
    assert np.isclose(feats["f01_energy"], 1.0 / K)
    # Contrast must be 0 for purely diagonal co-occurrence
    assert np.isclose(feats["f03_contrast"], 0.0)
    # Homogeneity must be 1.0 because |i - j| = 0 everywhere
    assert np.isclose(feats["f04_homogeneity"], 1.0)
    # Dissimilarity must be 0.0
    assert np.isclose(feats["f06_dissimilarity"], 0.0)


def test_extract_all_glcm_features_count_and_names():
    img = np.random.randint(0, 16, (60, 60), dtype=np.uint8)
    features, matrices = extract_all_glcm_features(
        img,
        orientations=(0.0, np.pi / 4.0),
        offsets=(1, 2, 3, 4),
        num_levels=16
    )

    # 8 GLCMs (2 angles x 4 offsets)
    assert len(matrices) == 8
    # 96 features (8 GLCMs x 12 Haralick features)
    assert len(features) == 96

    # Verify naming convention and zero NaN/Inf
    for key, val in features.items():
        assert any(h_name in key for h_name in HARALICK_FEATURE_NAMES)
        assert not np.isnan(val)
        assert not np.isinf(val)


def test_haralick_analytical_values():
    # Analytical verification on a 16x16 identity matrix
    K = 16
    p_ident = np.eye(K, dtype=np.float64) / K
    feats = extract_haralick_12(p_ident, num_levels=K)

    # 1. Energy: sum( (1/16)^2 * 16 ) = 1/16 = 0.0625
    assert np.isclose(feats["f01_energy"], 1.0 / K)
    # 2. Contrast: sum((i - j)^2 * p) = 0.0
    assert np.isclose(feats["f03_contrast"], 0.0)
    # 3. Homogeneity: sum(p / (1 + (i-j)^2)) = 1.0
    assert np.isclose(feats["f04_homogeneity"], 1.0)
    # 4. Correlation: cov/(std*std) = 1.0
    assert np.isclose(feats["f05_correlation"], 1.0, atol=1e-12)
    # 5. Sum Average: sum_{k=2}^{32} k * p_{x+y}(k) = 2 * mu = Ng + 1 = 17.0
    assert np.isclose(feats["f09_sum_average"], float(K + 1))
    # 6. Autocorrelation: sum(i^2 * p) = sum_{i=1}^16 i^2 / 16 = 1496 / 16 = 93.5
    assert np.isclose(feats["f12_autocorrelation"], 93.5)


def test_haralick_zero_variance_guard():
    # Single-pixel co-occurrence matrix (zero variance flatline)
    K = 16
    p_flat = np.zeros((K, K), dtype=np.float64)
    p_flat[4, 4] = 1.0  # (5, 5) in 1-based indexing
    feats = extract_haralick_12(p_flat, num_levels=K)

    # Correlation denominator is 0.0, guard returns 0.0
    assert feats["f05_correlation"] == 0.0
    # Variance is 0.0
    assert np.isclose(feats["f07_variance"], 0.0)
    # Entropy of deterministic outcome is 0.0 (within epsilon tolerance)
    assert np.isclose(feats["f02_entropy"], 0.0, atol=1e-9)


def test_glcm_spatial_offsets_analytical():
    # 2x2 raster image: [[0, 1], [2, 3]]
    img = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    # Horizontal (0 deg, d=1): pairs (0,1), (2,3) -> symmetric pairs (0,1),(1,0),(2,3),(3,2)
    glcm_h = compute_single_glcm(img, offset=1, theta=0.0, num_levels=4, symmetric=True)
    assert np.count_nonzero(glcm_h) == 4
    assert np.isclose(np.sum(glcm_h), 1.0)
    assert glcm_h[0, 1] == 0.25
    assert glcm_h[1, 0] == 0.25
    assert glcm_h[2, 3] == 0.25
    assert glcm_h[3, 2] == 0.25

    # Diagonal 45-deg up-right (pi/4, d=1):
    # Only valid pair is img[1, 0] (value 2) and img[0, 1] (value 1)
    # Symmetric pairs: (2, 1) and (1, 2)
    glcm_d = compute_single_glcm(img, offset=1, theta=np.pi / 4.0, num_levels=4, symmetric=True)
    assert np.count_nonzero(glcm_d) == 2
    assert np.isclose(np.sum(glcm_d), 1.0)
    assert glcm_d[2, 1] == 0.5
    assert glcm_d[1, 2] == 0.5

