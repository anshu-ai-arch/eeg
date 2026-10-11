"""
glcm.py

Gray-Level Co-occurrence Matrix (GLCM) computation and Haralick texture feature extraction.
Strictly adheres to:
- Ground Truth Section 3.5: 8 GLCMs (orientations theta in {0, pi/4}, offsets d in {1, 2, 3, 4}).
- Section 3.3.4 - 3.3.15: 12 Haralick features (Eqs. 6-19) using 1-based indexing (i, j in {1, ..., Ng}).
- Total Feature Vector: 96 features (12 features x 8 GLCMs) as explicitly stated on p. 7.
"""

import numpy as np

# Canonical feature names and equation references
HARALICK_FEATURE_NAMES = [
    "f01_energy",             # Eq. 6
    "f02_entropy",            # Eq. 7
    "f03_contrast",           # Eq. 8
    "f04_homogeneity",        # Eq. 9
    "f05_correlation",        # Eq. 10
    "f06_dissimilarity",      # Eq. 11
    "f07_variance",           # Eq. 12
    "f08_max_probability",    # Eq. 14
    "f09_sum_average",        # Eq. 16
    "f10_cluster_shade",      # Eq. 17
    "f11_cluster_prominence", # Eq. 18
    "f12_autocorrelation"     # Eq. 19
]


def compute_single_glcm(
    image: np.ndarray,
    offset: int,
    theta: float,
    num_levels: int = 16,
    symmetric: bool = True
) -> np.ndarray:
    """
    Compute normalized Gray-Level Co-occurrence Matrix p(i, j) for a given offset and orientation.

    Replication Implementation Note:
    The GLCMs were symmetrized before probability normalization, following the existing
    implementation retained for this replication. The available description in the primary
    paper does not clearly establish whether symmetrization was used; therefore, this is
    recorded as an implementation choice rather than a verified author-specified parameter.

    Parameters
    ----------
    image : np.ndarray
        2D quantized integer image with values in [0, num_levels - 1].
    offset : int
        Pixel distance d (1, 2, 3, 4).
    theta : float
        Orientation angle in radians (0.0 for 0°, np.pi/4 for 45°).
    num_levels : int, default=16
        Number of discrete gray levels K (matrix shape is K x K).
    symmetric : bool, default=True
        Whether to accumulate pairs symmetrically (standard in Haralick texture analysis).

    Returns
    -------
    p : np.ndarray
        Normalized GLCM probability matrix of shape (num_levels, num_levels), sum(p) == 1.0.
    """
    d = offset
    H, W = image.shape

    if np.isclose(theta, 0.0):
        # Horizontal: delta_r = 0, delta_c = d
        p1 = image[:, :-d].ravel()
        p2 = image[:, d:].ravel()
    elif np.isclose(theta, np.pi / 4.0):
        # Diagonal up-right (45°): delta_r = -d, delta_c = d
        # Pairs: image[r, c] and image[r - d, c + d]
        p1 = image[d:, :-d].ravel()
        p2 = image[:-d, d:].ravel()
    else:
        raise ValueError(f"Unsupported orientation theta={theta}. Ground Truth allows only 0 and pi/4.")

    idx = p1.astype(np.int32) * num_levels + p2.astype(np.int32)
    P = np.bincount(idx, minlength=num_levels * num_levels).reshape(num_levels, num_levels).astype(np.float64)

    if symmetric:
        P = P + P.T

    total = np.sum(P)
    if total > 0:
        p_norm = P / total
        # Verify probability sum within floating-point tolerance
        if not np.isclose(np.sum(p_norm), 1.0, atol=1e-7):
            raise ValueError(f"GLCM probability normalization failed: sum is {np.sum(p_norm)}")
    else:
        p_norm = np.zeros((num_levels, num_levels), dtype=np.float64)

    return p_norm


def extract_haralick_12(p: np.ndarray, num_levels: int = 16, eps: float = 1e-12) -> dict[str, float]:
    """
    Extract all 12 Haralick texture features from a normalized GLCM matrix.
    Uses 1-based gray-level coordinate indexing (i, j in {1, ..., Ng}) strictly matching Eqs. (6)-(19).

    Parameters
    ----------
    p : np.ndarray
        Normalized GLCM probability matrix of shape (num_levels, num_levels).
    num_levels : int, default=16
        Number of gray levels Ng.
    eps : float, default=1e-12
        Numerical stability epsilon.

    Returns
    -------
    features : dict[str, float]
        Dictionary mapping feature name to scalar value.
    """
    Ng = num_levels
    # 1-based indexing grids matching Eqs. (6)-(19)
    i_coords = np.arange(1, Ng + 1, dtype=np.float64)
    j_coords = np.arange(1, Ng + 1, dtype=np.float64)
    I, J = np.meshgrid(i_coords, j_coords, indexing="ij")

    # Marginals
    p_x = np.sum(p, axis=1)  # px(i) = sum_j p(i, j)
    p_y = np.sum(p, axis=0)  # py(j) = sum_i p(i, j)

    mu_x = float(np.sum(i_coords * p_x))
    mu_y = float(np.sum(j_coords * p_y))
    sigma_x = float(np.sqrt(np.sum((i_coords - mu_x) ** 2 * p_x)))
    sigma_y = float(np.sqrt(np.sum((j_coords - mu_y) ** 2 * p_y)))

    # f1: Energy (Angular Second Moment, Eq. 6)
    f1 = float(np.sum(p ** 2))

    # f2: Entropy (Eq. 7)
    # f2 = - sum_i sum_j p(i, j) * log2(p(i, j) + eps)
    mask = p > 0
    f2 = float(-np.sum(p[mask] * np.log2(p[mask] + eps)))

    # f3: Contrast (Eq. 8)
    f3 = float(np.sum(((I - J) ** 2) * p))

    # f4: Homogeneity (Inverse Difference Moment, Eq. 9)
    f4 = float(np.sum(p / (1.0 + (I - J) ** 2)))

    # f5: Correlation (Eq. 10)
    cov = float(np.sum((I - mu_x) * (J - mu_y) * p))
    denom_corr = sigma_x * sigma_y
    f5 = cov / (denom_corr + eps) if denom_corr > eps else 0.0

    # f6: Dissimilarity (Eq. 11)
    f6 = float(np.sum(np.abs(I - J) * p))

    # f7: Variance (Eq. 12)
    # Global intensity variance: sum_i sum_j (i - mu)^2 p(i, j) where mu = mu_x
    f7 = float(np.sum(((I - mu_x) ** 2) * p))

    # f8: Maximum probability (Eq. 14)
    f8 = float(np.max(p))

    # f9: Sum average (Eq. 16)
    # px+y(k) = sum_{i+j=k} p(i, j) for k = 2, 3, ..., 2*Ng (Eq. 15)
    sum_grid = (I + J).astype(np.int32)
    k_vals = np.arange(2, 2 * Ng + 1)
    p_x_plus_y = np.array([np.sum(p[sum_grid == k]) for k in k_vals], dtype=np.float64)
    f9 = float(np.sum(k_vals * p_x_plus_y))

    # f10: Cluster shade (Eq. 17)
    f10 = float(np.sum(((I + J - mu_x - mu_y) ** 3) * p))

    # f11: Cluster prominence (Eq. 18)
    f11 = float(np.sum(((I + J - mu_x - mu_y) ** 4) * p))

    # f12: Autocorrelation (Eq. 19)
    f12 = float(np.sum(I * J * p))

    return {
        "f01_energy": f1,
        "f02_entropy": f2,
        "f03_contrast": f3,
        "f04_homogeneity": f4,
        "f05_correlation": f5,
        "f06_dissimilarity": f6,
        "f07_variance": f7,
        "f08_max_probability": f8,
        "f09_sum_average": f9,
        "f10_cluster_shade": f10,
        "f11_cluster_prominence": f11,
        "f12_autocorrelation": f12
    }


def extract_all_glcm_features(
    quantized_grp: np.ndarray,
    orientations: tuple[float, ...] = (0.0, np.pi / 4.0),
    offsets: tuple[int, ...] = (1, 2, 3, 4),
    num_levels: int = 16,
    eps: float = 1e-12
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    Extract the complete 96-feature vector across 8 GLCMs (2 orientations x 4 offsets).

    Returns
    -------
    features : dict[str, float]
        Dictionary of 96 named features with format 'glcm_{angle_deg}deg_d{offset}_{feature_name}'.
    glcm_matrices : dict[str, np.ndarray]
        The 8 normalized GLCM matrices of shape (num_levels, num_levels).
    """
    features = {}
    glcm_matrices = {}

    for theta in orientations:
        angle_deg = 0 if np.isclose(theta, 0.0) else 45
        for d in offsets:
            matrix_key = f"theta_{angle_deg}deg_d_{d}"
            p_matrix = compute_single_glcm(quantized_grp, offset=d, theta=theta, num_levels=num_levels)
            glcm_matrices[matrix_key] = p_matrix

            feats_12 = extract_haralick_12(p_matrix, num_levels=num_levels, eps=eps)
            for f_name, f_val in feats_12.items():
                full_key = f"glcm_a{angle_deg:02d}_d{d}_{f_name}"
                features[full_key] = f_val

    return features, glcm_matrices
