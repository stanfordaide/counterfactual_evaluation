"""Metric computation for PE classifier evaluation.

Handles the all-positive counterfactual case (sensitivity-focused)
as well as standard binary classification metrics when both classes are present.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    """Container for binary classification metrics."""
    n_samples: int
    n_positive: int
    n_negative: int
    sensitivity: float          # recall for positive class
    specificity: float
    f1: float
    balanced_accuracy: float
    auc_roc: float | None       # None if single class
    auc_pr: float | None        # None if single class
    mean_score: float
    median_score: float
    miss_rate: float            # 1 - sensitivity
    threshold: float

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute classification metrics from targets and continuous scores.

    Args:
        targets: Ground truth binary labels (0 or 1).
        scores: Model output probabilities (after sigmoid).
        threshold: Decision threshold for binarization.

    Returns:
        ClassificationMetrics with all computed values.
    """
    targets = np.asarray(targets, dtype=float)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    n_pos = int(targets.sum())
    n_neg = int(len(targets) - n_pos)

    # Sensitivity and specificity from confusion matrix
    if n_pos > 0 and n_neg > 0:
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    elif n_pos > 0:
        # All positive (counterfactual case)
        tp = int(preds.sum())
        fn = n_pos - tp
        sensitivity = tp / n_pos
        specificity = float("nan")
    else:
        # All negative
        tn = int((preds == 0).sum())
        fp = n_neg - tn
        sensitivity = float("nan")
        specificity = tn / n_neg

    # AUC metrics require both classes
    auc_roc = None
    auc_pr = None
    if n_pos > 0 and n_neg > 0:
        try:
            auc_roc = roc_auc_score(targets, scores)
            auc_pr = average_precision_score(targets, scores)
        except ValueError:
            pass

    f1 = f1_score(targets, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(targets, preds)

    return ClassificationMetrics(
        n_samples=len(targets),
        n_positive=n_pos,
        n_negative=n_neg,
        sensitivity=sensitivity,
        specificity=specificity,
        f1=f1,
        balanced_accuracy=bal_acc,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        mean_score=float(scores.mean()),
        median_score=float(np.median(scores)),
        miss_rate=1.0 - sensitivity if not np.isnan(sensitivity) else float("nan"),
        threshold=threshold,
    )


def find_optimal_threshold(
    targets: np.ndarray,
    scores: np.ndarray,
    method: str = "youden",
) -> tuple[float, float]:
    """Find the optimal decision threshold.

    Args:
        targets: Ground truth binary labels.
        scores: Model output probabilities.
        method: 'youden' for Youden's J statistic, 'f1' for F1 maximization.

    Returns:
        Tuple of (optimal_threshold, metric_value_at_threshold).
    """
    targets = np.asarray(targets, dtype=float)
    scores = np.asarray(scores, dtype=float)

    if method == "youden":
        if len(np.unique(targets)) < 2:
            # Can't compute ROC with single class; sweep thresholds for sensitivity
            thresholds = np.linspace(0, 1, 201)
            best_t, best_sens = 0.5, 0.0
            for t in thresholds:
                preds = (scores >= t).astype(int)
                sens = preds.sum() / len(preds) if targets.sum() == len(targets) else 0.0
                if sens > best_sens:
                    best_sens = sens
                    best_t = t
            return float(best_t), float(best_sens)

        fpr, tpr, thresholds = roc_curve(targets, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])

    elif method == "f1":
        thresholds = np.linspace(0.01, 0.99, 199)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (scores >= t).astype(int)
            f = f1_score(targets, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        return float(best_t), float(best_f1)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'youden' or 'f1'.")


def bootstrap_ci(
    targets: np.ndarray,
    scores: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        targets: Ground truth labels.
        scores: Model output probabilities.
        metric_fn: Function(targets, scores) -> float.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    scores = np.asarray(scores)
    n = len(targets)

    point = metric_fn(targets, scores)
    boot_values = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_fn(targets[idx], scores[idx])
            boot_values.append(val)
        except (ValueError, ZeroDivisionError):
            continue

    if len(boot_values) < 10:
        return point, float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_values, 100 * alpha))
    upper = float(np.percentile(boot_values, 100 * (1 - alpha)))
    return point, lower, upper


def sensitivity_at_threshold(targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """Compute sensitivity at a given threshold."""
    targets = np.asarray(targets, dtype=float)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    n_pos = targets.sum()
    if n_pos == 0:
        return float("nan")
    return float((preds[targets == 1] == 1).sum() / n_pos)
