"""
scripts/classification/nested_cv.py

Leak-Free Stratified Nested Cross-Validation Engine.
Implements the classification protocol specified in:
- Lashkari et al. (2026), Section 3.6 & 3.6.3
- Approved Protocol (Stage 1.5 Audit):
  - Stratified 10-fold outer CV and Stratified 10-fold inner CV (10x10)
  - Fixed random seed 42
  - Feature selection: Supervised MCFS on training fold only
  - Classifier: SVM with RBF kernel and One-vs-One (OvO) decision strategy
  - Hyperparameter Grid:
      C in {10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3}
      gamma in {10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3}
      k in {5, 10, 15}
  - Primary Inner Selection: Mean macro-F1 across 10 inner folds
  - Tie-breakers: Mean accuracy -> prefer k=10, then k=5, then k=15 -> smaller C -> smaller gamma
  - Strict Isolation: Scaling, MCFS, and SVM tuning never see outer test samples
"""

import os
import sys
import json
import tempfile
import hashlib
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from scripts.classification.mcfs import MCFS
except (ImportError, ModuleNotFoundError):
    try:
        from .mcfs import MCFS
    except (ImportError, ModuleNotFoundError):
        from mcfs import MCFS

logger = logging.getLogger(__name__)


CANDIDATE_K = [5, 10, 15]
CANDIDATE_C = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
CANDIDATE_GAMMA = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

K_TIEBREAK_ORDER = {10: 0, 5: 1, 15: 2}


@dataclass
class NestedCVConfig:
    """Configuration for nested cross-validation."""
    n_outer_splits: int = 10
    n_inner_splits: int = 10
    random_state: int = 42
    candidate_k: List[int] = field(default_factory=lambda: CANDIDATE_K.copy())
    candidate_c: List[float] = field(default_factory=lambda: CANDIDATE_C.copy())
    candidate_gamma: List[float] = field(default_factory=lambda: CANDIDATE_GAMMA.copy())
    mcfs_alpha: float = 0.01
    mcfs_neighbors: int = 5


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int] | None = None,
    pos_label: int | None = None,
) -> Dict[str, Any]:
    """
    Compute classification performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels.
    y_pred : np.ndarray
        Predicted class labels.
    labels : list of int or None
        Unique class labels to consider in confusion matrix.
    pos_label : int or None
        Positive class label for binary sensitivity/specificity.

    Returns
    -------
    metrics : dict
        Computed metrics including accuracy, macro precision, recall, F1,
        confusion matrix, and sensitivity/specificity if binary.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if labels is None:
        labels = [int(l) for l in sorted(list(np.unique(np.concatenate([y_true, y_pred]))))]
    else:
        labels = [int(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    result = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }

    if len(labels) == 2 and pos_label is not None:
        neg_label = [l for l in labels if l != pos_label][0]
        # Binary confusion matrix layout: [[TN, FP], [FN, TP]]
        # where row 0 is neg_label, row 1 is pos_label
        cm_binary = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
        tn, fp, fn, tp = cm_binary.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        result["sensitivity"] = sensitivity
        result["specificity"] = specificity
        result["tp"] = int(tp)
        result["tn"] = int(tn)
        result["fp"] = int(fp)
        result["fn"] = int(fn)

    return result


def tune_inner_cv(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    config: NestedCVConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run stratified 10-fold inner CV to select optimal (k, C, gamma).

    Parameters
    ----------
    X_outer_train : np.ndarray of shape (n_samples, n_features)
        Training features for the outer fold.
    y_outer_train : np.ndarray of shape (n_samples,)
        Training labels for the outer fold.
    config : NestedCVConfig
        Nested CV hyperparameters and grid specifications.

    Returns
    -------
    best_params : dict
        Selected hyperparameters {'k': int, 'C': float, 'gamma': float, 'inner_macro_f1': float, 'inner_accuracy': float}.
    inner_results_df : pd.DataFrame
        Table of all candidate parameter evaluations with mean inner metrics.
    """
    inner_cv = StratifiedKFold(
        n_splits=config.n_inner_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    max_k = max(config.candidate_k)
    param_grid: List[Tuple[int, float, float]] = [
        (k, c, g)
        for k in config.candidate_k
        for c in config.candidate_c
        for g in config.candidate_gamma
    ]

    # Initialize score accumulators: grid_idx -> list of fold f1s and accuracies
    grid_f1s: Dict[Tuple[int, float, float], List[float]] = {p: [] for p in param_grid}
    grid_accs: Dict[Tuple[int, float, float], List[float]] = {p: [] for p in param_grid}

    for inner_fold_idx, (in_train_idx, in_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
        # Verify strict separation
        assert len(set(in_train_idx).intersection(set(in_val_idx))) == 0, "Inner train and validation overlap!"

        # 1. Fit scaling on inner-train only
        scaler = StandardScaler()
        X_in_train_scaled = scaler.fit_transform(X_outer_train[in_train_idx])
        X_in_val_scaled = scaler.transform(X_outer_train[in_val_idx])

        y_in_train = y_outer_train[in_train_idx]
        y_in_val = y_outer_train[in_val_idx]

        # 2. Fit MCFS on inner-train only once up to max_k
        mcfs = MCFS(
            n_selected_features=max_k,
            n_neighbors=config.mcfs_neighbors,
            alpha=config.mcfs_alpha,
            supervised=True,
            random_state=config.random_state,
        )
        mcfs.fit(X_in_train_scaled, y_in_train)
        ranking = mcfs.ranking_

        # 3. Evaluate candidates by slicing ranking
        for k in config.candidate_k:
            selected_features = ranking[:k]
            X_tr_k = X_in_train_scaled[:, selected_features]
            X_va_k = X_in_val_scaled[:, selected_features]

            for c in config.candidate_c:
                for g in config.candidate_gamma:
                    clf = SVC(
                        C=c,
                        gamma=g,
                        kernel="rbf",
                        decision_function_shape="ovo",
                        random_state=config.random_state,
                    )
                    clf.fit(X_tr_k, y_in_train)
                    preds = clf.predict(X_va_k)

                    f1 = float(f1_score(y_in_val, preds, average="macro", zero_division=0))
                    acc = float(accuracy_score(y_in_val, preds))

                    grid_f1s[(k, c, g)].append(f1)
                    grid_accs[(k, c, g)].append(acc)

    # Aggregate inner CV results
    records = []
    for (k, c, g) in param_grid:
        m_f1 = float(np.mean(grid_f1s[(k, c, g)]))
        m_acc = float(np.mean(grid_accs[(k, c, g)]))
        k_tie = K_TIEBREAK_ORDER.get(k, 99)
        records.append({
            "k": k,
            "C": c,
            "gamma": g,
            "mean_macro_f1": m_f1,
            "mean_accuracy": m_acc,
            "k_tiebreak_rank": k_tie,
        })

    results_df = pd.DataFrame(records)

    # Approved tie-breaking rule:
    # 1. Higher mean macro-F1
    # 2. Higher mean accuracy
    # 3. Preferred k: k=10 (rank 0), then k=5 (rank 1), then k=15 (rank 2)
    # 4. Smaller C
    # 5. Smaller gamma
    def tiebreak_key(row: Dict[str, Any]) -> Tuple[float, float, int, float, float]:
        return (
            -row["mean_macro_f1"],
            -row["mean_accuracy"],
            row["k_tiebreak_rank"],
            row["C"],
            row["gamma"],
        )

    best_record = min(records, key=tiebreak_key)

    best_params = {
        "k": int(best_record["k"]),
        "C": float(best_record["C"]),
        "gamma": float(best_record["gamma"]),
        "inner_macro_f1": float(best_record["mean_macro_f1"]),
        "inner_accuracy": float(best_record["mean_accuracy"]),
    }

    return best_params, results_df


def run_nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: List[str],
    config: NestedCVConfig | None = None,
    pos_label: int | None = None,
    checkpoint_dir: Path | None = None,
    fold_callback: Callable[[int, Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """
    Execute full Stratified 10x10 Nested Cross-Validation.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target class labels.
    sample_ids : list of str of length n_samples
        Unique sample/recording identifiers.
    config : NestedCVConfig or None
        Configuration options; defaults to standard protocol if None.
    pos_label : int or None
        Positive class label for binary sensitivity/specificity metrics.

    Returns
    -------
    run_results : dict
        Complete dictionary containing:
        - outer_predictions: list of dict per sample
        - fold_metrics: list of dict per outer fold
        - selected_parameters: list of dict per outer fold
        - overall_metrics: dict of pooled and mean/std metrics
        - fold_splits: dict of outer and inner fold index splits
        - runtime_seconds: float
    """
    if config is None:
        config = NestedCVConfig()

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    n_samples, n_features = X_arr.shape

    if len(sample_ids) != n_samples:
        raise ValueError(f"sample_ids length ({len(sample_ids)}) does not match sample count ({n_samples})")

    unique_classes = sorted(list(np.unique(y_arr)))
    outer_cv = StratifiedKFold(
        n_splits=config.n_outer_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    outer_predictions: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    selected_parameters: List[Dict[str, Any]] = []
    fold_splits: Dict[str, Any] = {"outer_folds": {}}

    # Create an integrity signature combining config, random state, and exact dataset sums
    state_str = f"rs{config.random_state}_out{config.n_outer_splits}_in{config.n_inner_splits}_k{config.candidate_k}_X{X_arr.shape}_Xsum{float(np.sum(X_arr))}_ysum{int(np.sum(y_arr))}"
    run_signature = hashlib.sha256(state_str.encode()).hexdigest()

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Load completed folds
        for outer_fold_idx in range(config.n_outer_splits):
            chk_file = checkpoint_dir / f"fold_{outer_fold_idx}.json"
            if chk_file.exists():
                try:
                    with open(chk_file, "r") as f:
                        fold_data = json.load(f)
                    
                    if fold_data.get("run_signature") != run_signature:
                        raise ValueError(f"Signature mismatch in {chk_file}. The dataset or config has changed.")
                        
                    outer_predictions.extend(fold_data["outer_predictions"])
                    fold_metrics.append(fold_data["metrics"])
                    selected_parameters.append(fold_data["selected_parameters"])
                    fold_splits["outer_folds"][f"fold_{outer_fold_idx}"] = fold_data["fold_splits"]
                    logger.info(f"Loaded completed outer fold {outer_fold_idx} from checkpoint.")
                except Exception as e:
                    raise RuntimeError(f"Failed to load corrupt checkpoint {chk_file}: {e}")

    start_time = time.perf_counter()

    for outer_fold_idx, (out_train_idx, out_test_idx) in enumerate(outer_cv.split(X_arr, y_arr)):
        if checkpoint_dir is not None:
            chk_file = checkpoint_dir / f"fold_{outer_fold_idx}.json"
            if chk_file.exists():
                continue

        fold_start = time.perf_counter()

        # Strict isolation assertion
        assert len(set(out_train_idx).intersection(set(out_test_idx))) == 0, (
            f"Fold {outer_fold_idx}: Train and test sets overlap!"
        )
        assert len(out_train_idx) + len(out_test_idx) == n_samples, (
            f"Fold {outer_fold_idx}: Missing samples in split!"
        )

        # Record split indices
        fold_splits["outer_folds"][f"fold_{outer_fold_idx}"] = {
            "outer_train_indices": out_train_idx.tolist(),
            "outer_test_indices": out_test_idx.tolist(),
            "outer_train_sample_ids": [sample_ids[i] for i in out_train_idx],
            "outer_test_sample_ids": [sample_ids[i] for i in out_test_idx],
        }

        X_out_train, y_out_train = X_arr[out_train_idx], y_arr[out_train_idx]
        X_out_test, y_out_test = X_arr[out_test_idx], y_arr[out_test_idx]

        # 1. Inner Cross-Validation Tuning
        best_params, _ = tune_inner_cv(X_out_train, y_out_train, config)

        k_star = best_params["k"]
        c_star = best_params["C"]
        gamma_star = best_params["gamma"]

        # 2. Refit Outer Scaler & MCFS on 100% of outer training fold
        outer_scaler = StandardScaler()
        X_out_train_scaled = outer_scaler.fit_transform(X_out_train)
        X_out_test_scaled = outer_scaler.transform(X_out_test)

        outer_mcfs = MCFS(
            n_selected_features=k_star,
            n_neighbors=config.mcfs_neighbors,
            alpha=config.mcfs_alpha,
            supervised=True,
            random_state=config.random_state,
        )
        outer_mcfs.fit(X_out_train_scaled, y_out_train)

        X_out_train_sel = outer_mcfs.transform(X_out_train_scaled)
        X_out_test_sel = outer_mcfs.transform(X_out_test_scaled)

        # 3. Train final Outer SVM
        outer_svm = SVC(
            C=c_star,
            gamma=gamma_star,
            kernel="rbf",
            decision_function_shape="ovo",
            random_state=config.random_state,
        )
        outer_svm.fit(X_out_train_sel, y_out_train)

        # 4. Predict on held-out outer test fold exactly once
        outer_preds = outer_svm.predict(X_out_test_sel)
        fold_time = time.perf_counter() - fold_start

        # Record predictions
        for i_local, global_idx in enumerate(out_test_idx):
            outer_predictions.append({
                "sample_id": sample_ids[global_idx],
                "global_index": int(global_idx),
                "outer_fold": outer_fold_idx,
                "y_true": int(y_out_test[i_local]),
                "y_pred": int(outer_preds[i_local]),
                "is_correct": bool(y_out_test[i_local] == outer_preds[i_local]),
            })

        # Evaluate fold metrics
        metrics = compute_metrics(
            y_out_test,
            outer_preds,
            labels=unique_classes,
            pos_label=pos_label,
        )
        metrics["outer_fold"] = outer_fold_idx
        metrics["n_test_samples"] = len(out_test_idx)
        metrics["selected_k"] = k_star
        metrics["selected_C"] = c_star
        metrics["selected_gamma"] = gamma_star
        metrics["runtime_seconds"] = fold_time
        fold_metrics.append(metrics)

        # Record selected parameters
        selected_parameters.append({
            "outer_fold": outer_fold_idx,
            "k": k_star,
            "C": c_star,
            "gamma": gamma_star,
            "inner_macro_f1": best_params["inner_macro_f1"],
            "inner_accuracy": best_params["inner_accuracy"],
            "selected_features_indices": outer_mcfs.selected_indices_.tolist(),
            "selected_features_scores": [float(s) for s in outer_mcfs.feature_scores_[outer_mcfs.selected_indices_]],
            "alpha_used": [float(a) for a in outer_mcfs.alpha_used_],
            "lasso_fallback_used": outer_mcfs.lasso_fallback_used_,
            "runtime_seconds": fold_time,
        })

        if checkpoint_dir is not None:
            fold_state = {
                "run_signature": run_signature,
                "outer_fold_idx": outer_fold_idx,
                "metrics": fold_metrics[-1],
                "selected_parameters": selected_parameters[-1],
                "outer_predictions": outer_predictions[-len(out_test_idx):],
                "fold_splits": fold_splits["outer_folds"][f"fold_{outer_fold_idx}"]
            }
            chk_file = checkpoint_dir / f"fold_{outer_fold_idx}.json"
            tmp_file = checkpoint_dir / f"fold_{outer_fold_idx}.json.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(fold_state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, chk_file)

        if fold_callback is not None:
            fold_callback(outer_fold_idx, fold_metrics[-1])

    total_runtime = time.perf_counter() - start_time
    cumulative_fold_runtime = float(sum(m["runtime_seconds"] for m in fold_metrics))

    # Sort pooled outer predictions by global index
    outer_predictions_sorted = sorted(outer_predictions, key=lambda x: x["global_index"])
    y_true_pooled = np.array([p["y_true"] for p in outer_predictions_sorted])
    y_pred_pooled = np.array([p["y_pred"] for p in outer_predictions_sorted])

    # Pooled metrics
    pooled_metrics = compute_metrics(
        y_true_pooled,
        y_pred_pooled,
        labels=unique_classes,
        pos_label=pos_label,
    )

    # Mean and standard deviation across the 10 folds
    mean_fold_metrics = {
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics])),
        "mean_macro_precision": float(np.mean([m["macro_precision"] for m in fold_metrics])),
        "std_macro_precision": float(np.std([m["macro_precision"] for m in fold_metrics])),
        "mean_macro_recall": float(np.mean([m["macro_recall"] for m in fold_metrics])),
        "std_macro_recall": float(np.std([m["macro_recall"] for m in fold_metrics])),
        "mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "std_macro_f1": float(np.std([m["macro_f1"] for m in fold_metrics])),
    }

    if pos_label is not None and len(unique_classes) == 2:
        mean_fold_metrics["mean_sensitivity"] = float(np.mean([m["sensitivity"] for m in fold_metrics]))
        mean_fold_metrics["std_sensitivity"] = float(np.std([m["sensitivity"] for m in fold_metrics]))
        mean_fold_metrics["mean_specificity"] = float(np.mean([m["specificity"] for m in fold_metrics]))
        mean_fold_metrics["std_specificity"] = float(np.std([m["specificity"] for m in fold_metrics]))

    return {
        "outer_predictions": outer_predictions_sorted,
        "fold_metrics": fold_metrics,
        "selected_parameters": selected_parameters,
        "pooled_metrics": pooled_metrics,
        "mean_fold_metrics": mean_fold_metrics,
        "fold_splits": fold_splits,
        "total_runtime_seconds": total_runtime,
        "cumulative_fold_runtime_seconds": cumulative_fold_runtime,
        "n_samples": n_samples,
        "n_features": n_features,
        "unique_classes": unique_classes,
    }
