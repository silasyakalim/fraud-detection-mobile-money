"""Tests for cost-sensitive metrics."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.evaluation.metrics import (
    cost_curve,
    expected_cost,
    pr_auc,
    precision_at_k,
    threshold_for_alert_volume,
)


@pytest.fixture
def y_true_scores() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.random(n) < 0.05).astype(int)
    # Scores correlated with truth but noisy.
    y_score = np.clip(0.7 * y_true + 0.3 * rng.random(n), 0, 1)
    return y_true, y_score


def test_expected_cost_decomposes_correctly(
    y_true_scores: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_score = y_true_scores
    cb = expected_cost(
        y_true, y_score, threshold=0.5, cost_false_negative=100, cost_false_positive=5
    )
    assert cb.fn_loss == cb.false_negatives * 100
    assert cb.fp_cost == cb.false_positives * 5
    assert cb.total_cost == cb.fn_loss + cb.fp_cost
    assert cb.n_alerts == cb.true_positives + cb.false_positives


def test_precision_at_k_int(y_true_scores: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = y_true_scores
    p = precision_at_k(y_true, y_score, k=50)
    assert 0.0 <= p <= 1.0


def test_precision_at_k_fractional(y_true_scores: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = y_true_scores
    p = precision_at_k(y_true, y_score, k=0.05)
    assert 0.0 <= p <= 1.0


def test_precision_at_k_top_is_at_least_random(
    y_true_scores: tuple[np.ndarray, np.ndarray],
) -> None:
    """A non-trivial scorer should beat the base rate at the top."""
    y_true, y_score = y_true_scores
    base_rate = float(y_true.mean())
    top_precision = precision_at_k(y_true, y_score, k=0.05)
    assert top_precision >= base_rate


def test_pr_auc_in_range(y_true_scores: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = y_true_scores
    score = pr_auc(y_true, y_score)
    assert 0.0 <= score <= 1.0


def test_threshold_for_alert_volume(y_true_scores: tuple[np.ndarray, np.ndarray]) -> None:
    _, y_score = y_true_scores
    threshold = threshold_for_alert_volume(y_score, target_rate=0.1)
    actual_rate = float((y_score >= threshold).mean())
    assert abs(actual_rate - 0.1) < 0.02


def test_cost_curve_shapes(y_true_scores: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = y_true_scores
    rates, costs = cost_curve(y_true, y_score, n_points=20)
    assert rates.shape == (20,)
    assert costs.shape == (20,)
    assert (rates >= 0).all()
    assert (rates <= 1).all()
    assert (costs >= 0).all()
