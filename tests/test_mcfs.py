"""
tests/test_mcfs.py

Unit tests for Supervised Multi-Cluster Feature Selection (MCFS).
Verifies:
- Scikit-learn transformer interface (fit, transform, fit_transform).
- Output shape and unique valid feature indices for candidate k in {5, 10, 15}.
- Deterministic ranking on identical inputs.
- Input validation (NaN, Inf, dimensional mismatch, un-fitted transform).
- Graph symmetry, supervised label-gating, and Laplacian eigenproblem properties.
- Mathematical discriminative power (informative feature recovery).
- Degenerate graphs, zero-variance features, and numerical stability.
"""

import numpy as np
import pytest
from scripts.classification.mcfs import MCFS


def test_mcfs_output_shapes_and_uniqueness():
    np.random.seed(42)
    n_samples = 60
    n_features = 30
    X = np.random.randn(n_samples, n_features)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)

    for k in [5, 10, 15]:
        mcfs = MCFS(n_selected_features=k, n_neighbors=5, random_state=42)
        X_sel = mcfs.fit_transform(X, y)

        assert X_sel.shape == (n_samples, k)
        assert len(mcfs.selected_indices_) == k
        # Must be unique indices
        assert len(set(mcfs.selected_indices_)) == k
        # Must be valid column indices within [0, n_features - 1]
        assert all(0 <= idx < n_features for idx in mcfs.selected_indices_)
        assert len(mcfs.ranking_) == n_features
        assert len(mcfs.feature_scores_) == n_features
        assert np.all(np.isfinite(mcfs.feature_scores_))


def test_mcfs_deterministic_ranking():
    np.random.seed(123)
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 2, 50)

    mcfs1 = MCFS(n_selected_features=8, random_state=42)
    mcfs1.fit(X, y)

    mcfs2 = MCFS(n_selected_features=8, random_state=42)
    mcfs2.fit(X, y)

    np.testing.assert_array_equal(mcfs1.selected_indices_, mcfs2.selected_indices_)
    np.testing.assert_array_equal(mcfs1.ranking_, mcfs2.ranking_)
    np.testing.assert_allclose(mcfs1.feature_scores_, mcfs2.feature_scores_)


def test_mcfs_input_validation():
    mcfs = MCFS(n_selected_features=5)

    # 1. Transform before fit
    with pytest.raises(RuntimeError, match="not fitted"):
        mcfs.transform(np.ones((10, 10)))

    # 2. Non-finite values
    X_nan = np.ones((10, 10))
    X_nan[2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mcfs.fit(X_nan, np.zeros(10))

    X_inf = np.ones((10, 10))
    X_inf[2, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        mcfs.fit(X_inf, np.zeros(10))

    # 3. Shape mismatch between X and y
    with pytest.raises(ValueError, match="mismatch"):
        mcfs.fit(np.ones((10, 5)), np.zeros(8))

    # 4. k > n_features
    with pytest.raises(ValueError, match="cannot exceed"):
        MCFS(n_selected_features=15).fit(np.ones((10, 10)), np.zeros(10))

    # 5. k < 1
    with pytest.raises(ValueError, match="must be >= 1"):
        MCFS(n_selected_features=0).fit(np.ones((10, 10)), np.zeros(10))

    # 6. Single class in supervised mode
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
        MCFS(n_selected_features=3, supervised=True).fit(np.ones((10, 5)), np.zeros(10))


def test_mcfs_training_only_isolation():
    # Verify strict training-only isolation
    np.random.seed(99)
    X_train = np.random.randn(40, 12)
    y_train = np.array([0] * 20 + [1] * 20)

    X_test = np.random.randn(15, 12)

    mcfs = MCFS(n_selected_features=4, random_state=42)
    mcfs.fit(X_train, y_train)

    # Transform on held-out test data
    X_test_sel = mcfs.transform(X_test)
    assert X_test_sel.shape == (15, 4)

    # Selected columns must match the training learned columns exactly
    for col_i, feat_idx in enumerate(mcfs.selected_indices_):
        np.testing.assert_array_equal(X_test_sel[:, col_i], X_test[:, feat_idx])


def test_mcfs_supervised_graph_symmetry_and_laplacian():
    # Construct small 6-sample dataset with 2 classes (3 per class)
    X = np.array([
        [0.0, 0.1],
        [0.1, 0.0],
        [0.0, 0.2],
        [10.0, 10.1],
        [10.1, 10.0],
        [10.0, 10.2],
    ], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1])

    mcfs = MCFS(n_selected_features=1, n_neighbors=2, supervised=True, random_state=42)
    mcfs.fit(X, y)

    # Eigenvalues of normalized Laplacian must lie in [0, 2]
    assert np.all(mcfs.eigenvalues_ >= -1e-10)
    assert np.all(mcfs.eigenvalues_ <= 2.0 + 1e-10)

    # The first feature (separating class 0 at 0.0 from class 1 at 10.0) should have high score
    assert mcfs.selected_indices_[0] in [0, 1]


def test_mcfs_informative_feature_recovery():
    # Synthetic problem where features 0, 1, 2 are strongly discriminative
    # and features 3 to 19 are pure random noise
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    y = np.random.choice([0, 1], size=n_samples)
    X = np.random.randn(n_samples, n_features) * 0.5

    # Inject strong class signals into features 0, 1, 2
    X[y == 0, 0] += 5.0
    X[y == 1, 0] -= 5.0
    X[y == 0, 1] += 4.0
    X[y == 1, 1] -= 4.0
    X[y == 0, 2] += 3.0
    X[y == 1, 2] -= 3.0

    mcfs = MCFS(n_selected_features=5, n_neighbors=5, alpha=0.01, random_state=42)
    mcfs.fit(X, y)

    top_3 = set(mcfs.selected_indices_[:3])
    # Features 0 and 1 must be among the top selected features
    assert 0 in top_3
    assert 1 in top_3


def test_mcfs_weighting_modes():
    np.random.seed(77)
    X = np.random.randn(30, 10)
    y = np.array([0] * 15 + [1] * 15)

    # Heat kernel
    mcfs_hk = MCFS(n_selected_features=4, weighting="heat_kernel", random_state=42)
    X_hk = mcfs_hk.fit_transform(X, y)
    assert X_hk.shape == (30, 4)

    # Binary 0-1 weighting
    mcfs_bin = MCFS(n_selected_features=4, weighting="binary", random_state=42)
    X_bin = mcfs_bin.fit_transform(X, y)
    assert X_bin.shape == (30, 4)


def test_mcfs_constant_feature_handling():
    # Feature 0 is constant zero (zero variance)
    np.random.seed(88)
    X = np.random.randn(30, 6)
    X[:, 0] = 0.0
    y = np.array([0] * 15 + [1] * 15)

    mcfs = MCFS(n_selected_features=3, random_state=42)
    # Should not raise ZeroDivisionError or crash
    X_sel = mcfs.fit_transform(X, y)
    assert X_sel.shape == (30, 3)
    assert not np.any(np.isnan(X_sel))
    # Constant feature 0 should have zero or minimal score
    assert mcfs.feature_scores_[0] <= np.min(mcfs.feature_scores_) + 1e-6


def test_mcfs_disconnected_components_and_repeated_zero_eigenvalues():
    # 3 distinct classes forming 3 completely disconnected graph components
    np.random.seed(42)
    n_per_class = 15
    n_features = 10
    X = np.random.randn(3 * n_per_class, n_features) * 0.1
    # Strongly separate classes along feature 0 and 1
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    X[y == 0, 0] += 10.0
    X[y == 1, 0] -= 10.0
    X[y == 2, 1] += 10.0

    mcfs = MCFS(n_selected_features=3, n_clusters=3, supervised=True, random_state=42)
    mcfs.fit(X, y)

    # For 3 classes, the class-discriminant targets from the null space have eigenvalue 0.0
    assert len(mcfs.eigenvalues_) == 3
    # The first 2 discriminant targets (C - 1 = 2) must have eigenvalue 0.0
    assert np.allclose(mcfs.eigenvalues_[:2], 0.0)

    # Generalized eigenvectors must be finite and normalized
    assert np.all(np.isfinite(mcfs.eigenvectors_))
    # All targets must be non-constant across samples
    for col in range(mcfs.eigenvectors_.shape[1]):
        assert np.std(mcfs.eigenvectors_[:, col]) > 1e-4


def test_mcfs_isolated_vertices_handling():
    # Construct a dataset where one sample in class 1 has no neighbors
    np.random.seed(123)
    X = np.random.randn(21, 6)
    # 20 samples in class 0, exactly 1 sample in class 1 (isolated node in within-class k-NN)
    y = np.array([0] * 20 + [1] * 1)

    mcfs = MCFS(n_selected_features=2, supervised=True, random_state=42)
    # Must fit cleanly without ZeroDivisionError or numerical explosion
    mcfs.fit(X, y)

    assert mcfs.eigenvectors_.shape == (21, 2)
    assert np.all(np.isfinite(mcfs.eigenvectors_))
    # The isolated vertex coordinate must not explode to 10^5
    assert np.max(np.abs(mcfs.eigenvectors_[-1, :])) < 10.0
    assert len(mcfs.selected_indices_) == 2


def test_mcfs_deterministic_eigenvector_selection():
    # Synthetic problem with degenerate zero eigenvalues
    np.random.seed(999)
    X = np.random.randn(40, 8)
    y = np.array([0] * 20 + [1] * 20)
    X[y == 0, 0] += 5.0
    X[y == 1, 0] -= 5.0

    mcfs_1 = MCFS(n_selected_features=4, supervised=True, random_state=42)
    mcfs_1.fit(X, y)

    mcfs_2 = MCFS(n_selected_features=4, supervised=True, random_state=42)
    mcfs_2.fit(X, y)

    # Exact bitwise equality of eigenvalues, eigenvectors, and selected feature rankings
    np.testing.assert_array_equal(mcfs_1.eigenvalues_, mcfs_2.eigenvalues_)
    np.testing.assert_allclose(mcfs_1.eigenvectors_, mcfs_2.eigenvectors_, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(mcfs_1.selected_indices_, mcfs_2.selected_indices_)
    np.testing.assert_array_equal(mcfs_1.ranking_, mcfs_2.ranking_)


def test_mcfs_lasso_fallback_tracking():
    np.random.seed(55)
    X = np.random.randn(30, 8)
    y = np.array([0] * 15 + [1] * 15)

    # Normal alpha = 0.01: no fallback
    mcfs_norm = MCFS(n_selected_features=3, alpha=0.01, random_state=42)
    mcfs_norm.fit(X, y)
    assert hasattr(mcfs_norm, "alpha_used_")
    assert hasattr(mcfs_norm, "lasso_fallback_used_")
    assert np.allclose(mcfs_norm.alpha_used_, 0.01)
    assert all(fb == "none" for fb in mcfs_norm.lasso_fallback_used_)

    # Extreme alpha = 1000.0: triggers adaptive fallback
    mcfs_high = MCFS(n_selected_features=3, alpha=1000.0, random_state=42)
    mcfs_high.fit(X, y)
    assert any(fb in ["retry_alpha_scaled", "ridge"] for fb in mcfs_high.lasso_fallback_used_)
    assert any(a < 1000.0 for a in mcfs_high.alpha_used_)
    assert np.all(np.isfinite(mcfs_high.feature_scores_))

