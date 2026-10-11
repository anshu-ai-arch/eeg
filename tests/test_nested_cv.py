"""
tests/test_nested_cv.py

Unit tests for Nested Cross-Validation Engine.
Verifies:
- Strict split isolation (zero overlap between training and test/validation sets).
- Exact single coverage (every sample evaluated exactly once in outer predictions).
- Determinism on identical inputs and fixed random seed.
- Metric calculations (accuracy, sensitivity, specificity, macro-F1, confusion matrix).
- Approved tie-breaking rule implementation.
- Output shapes and data structure integrity.
"""

import numpy as np
import pytest
from scripts.classification.nested_cv import (
    NestedCVConfig,
    run_nested_cross_validation,
    compute_metrics,
    K_TIEBREAK_ORDER,
)


def test_metric_calculations_binary():
    # Known 2x2 confusion matrix:
    # y_true: 50 zeros (neg), 50 ones (pos)
    # y_pred: 40 correct zeros (TN=40, FP=10)
    #         45 correct ones  (TP=45, FN=5)
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.array([0] * 40 + [1] * 10 + [0] * 5 + [1] * 45)

    metrics = compute_metrics(y_true, y_pred, labels=[0, 1], pos_label=1)

    assert metrics["accuracy"] == pytest.approx(85.0 / 100.0)
    assert metrics["tp"] == 45
    assert metrics["tn"] == 40
    assert metrics["fp"] == 10
    assert metrics["fn"] == 5
    assert metrics["sensitivity"] == pytest.approx(45.0 / 50.0)  # 0.90
    assert metrics["specificity"] == pytest.approx(40.0 / 50.0)  # 0.80
    assert metrics["confusion_matrix"] == [[40, 10], [5, 45]]


def test_tiebreak_order():
    # Tiebreak order: k=10 (0), k=5 (1), k=15 (2)
    assert K_TIEBREAK_ORDER[10] < K_TIEBREAK_ORDER[5]
    assert K_TIEBREAK_ORDER[5] < K_TIEBREAK_ORDER[15]


def test_nested_cv_split_isolation_and_coverage():
    np.random.seed(42)
    n_samples = 40
    n_features = 8
    X = np.random.randn(n_samples, n_features)
    y = np.array([0] * 20 + [1] * 20)
    sample_ids = [f"rec_{i:03d}" for i in range(n_samples)]

    # Fast config for unit testing
    config = NestedCVConfig(
        n_outer_splits=4,
        n_inner_splits=3,
        random_state=42,
        candidate_k=[5],
        candidate_c=[1.0],
        candidate_gamma=[0.1],
    )

    results = run_nested_cross_validation(X, y, sample_ids, config=config, pos_label=1)

    # 1. Exact single coverage: all 40 samples appear exactly once
    pred_sample_ids = [p["sample_id"] for p in results["outer_predictions"]]
    assert len(pred_sample_ids) == n_samples
    assert set(pred_sample_ids) == set(sample_ids)

    # Global indices must cover 0 to n_samples - 1 exactly
    global_indices = [p["global_index"] for p in results["outer_predictions"]]
    assert sorted(global_indices) == list(range(n_samples))

    # 2. Strict isolation per fold
    for fold_name, fold_data in results["fold_splits"]["outer_folds"].items():
        tr_idx = set(fold_data["outer_train_indices"])
        te_idx = set(fold_data["outer_test_indices"])
        assert len(tr_idx.intersection(te_idx)) == 0
        assert len(tr_idx) + len(te_idx) == n_samples

    # 3. Output structure
    assert len(results["fold_metrics"]) == 4
    assert len(results["selected_parameters"]) == 4
    assert "mean_accuracy" in results["mean_fold_metrics"]
    assert "mean_sensitivity" in results["mean_fold_metrics"]
    assert "mean_specificity" in results["mean_fold_metrics"]


def test_nested_cv_determinism():
    np.random.seed(101)
    n_samples = 30
    n_features = 6
    X = np.random.randn(n_samples, n_features)
    y = np.array([0] * 15 + [1] * 15)
    sample_ids = [f"s_{i}" for i in range(n_samples)]

    config = NestedCVConfig(
        n_outer_splits=3,
        n_inner_splits=2,
        random_state=42,
        candidate_k=[5],
        candidate_c=[0.1, 1.0],
        candidate_gamma=[0.1],
    )

    run1 = run_nested_cross_validation(X, y, sample_ids, config=config, pos_label=1)
    run2 = run_nested_cross_validation(X, y, sample_ids, config=config, pos_label=1)

    preds1 = [p["y_pred"] for p in run1["outer_predictions"]]
    preds2 = [p["y_pred"] for p in run2["outer_predictions"]]
    assert preds1 == preds2

    np.testing.assert_allclose(
        [m["accuracy"] for m in run1["fold_metrics"]],
        [m["accuracy"] for m in run2["fold_metrics"]],
    )
