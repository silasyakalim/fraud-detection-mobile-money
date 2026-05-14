"""Evaluation metrics: expected loss, precision@k, PR-AUC, cost curve.

All functions take numpy arrays so they're framework-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.metrics import average_precision_score

from fraud_detection.config import settings


class CostBreakdown(NamedTuple):
    """Decomposition of expected loss at a chosen threshold."""

    threshold: float
    n_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    fn_loss: float
    fp_cost: float
    total_cost: float


def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    cost_false_negative: float | None = None,
    cost_false_positive: float | None = None,
) -> CostBreakdown:
    """Compute expected dollar loss at a probability threshold.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted fraud probabilities or scores.
        threshold: Decision threshold; predictions >= threshold are alerted.
        cost_false_negative: Average loss per missed fraud. Defaults to
            ``settings.cost.false_negative``.
        cost_false_positive: Cost per false alert (investigator time, customer
            friction). Defaults to ``settings.cost.false_positive``.

    Returns:
        CostBreakdown with the constituent counts and totals.
    """
    cfn = cost_false_negative if cost_false_negative is not None else settings.cost.false_negative
    cfp = cost_false_positive if cost_false_positive is not None else settings.cost.false_positive

    y_pred = y_score >= threshold
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fn_loss = fn * cfn
    fp_cost = fp * cfp
    return CostBreakdown(
        threshold=threshold,
        n_alerts=tp + fp,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        fn_loss=fn_loss,
        fp_cost=fp_cost,
        total_cost=fn_loss + fp_cost,
    )


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int | float,
) -> float:
    """Precision among the top-k highest-scoring predictions.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted scores.
        k: Either an integer count of top items, or a fraction in (0, 1)
            interpreted as ``ceil(k * len(y_score))``.

    Returns:
        Precision in [0, 1].
    """
    n = len(y_score)
    if isinstance(k, float):
        if not 0 < k <= 1:
            raise ValueError(f"fractional k must be in (0, 1], got {k}")
        k_int = max(1, int(np.ceil(k * n)))
    else:
        k_int = max(1, min(k, n))

    top_idx = np.argsort(y_score)[::-1][:k_int]
    return float(y_true[top_idx].mean())


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (PR-AUC). Preferred over ROC-AUC for rare events."""
    return float(average_precision_score(y_true, y_score))


def threshold_for_alert_volume(
    y_score: np.ndarray,
    target_rate: float,
) -> float:
    """Return the threshold that produces the target alert rate.

    Args:
        y_score: Predicted scores.
        target_rate: Desired fraction of items alerted, in (0, 1).

    Returns:
        The score threshold producing approximately ``target_rate`` alerts.
    """
    if not 0 < target_rate < 1:
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    return float(np.quantile(y_score, 1 - target_rate))


def cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_points: int = 100,
    cost_false_negative: float | None = None,
    cost_false_positive: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thresholds and return ``(alert_rates, total_costs)`` arrays."""
    thresholds = np.quantile(
        y_score, np.linspace(0.001, 0.999, n_points)
    )
    alert_rates = np.zeros(n_points)
    costs = np.zeros(n_points)
    for i, t in enumerate(thresholds):
        breakdown = expected_cost(
            y_true,
            y_score,
            float(t),
            cost_false_negative=cost_false_negative,
            cost_false_positive=cost_false_positive,
        )
        alert_rates[i] = breakdown.n_alerts / len(y_score)
        costs[i] = breakdown.total_cost
    return alert_rates, costs
