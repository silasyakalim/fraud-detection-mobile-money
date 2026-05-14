"""End-to-end training flow on Prefect 3.

Stages: ingest -> features -> train -> evaluate -> register.

Run locally:
    uv run python -m fraud_detection.pipelines.train_flow
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import polars as pl
from prefect import flow, task
from prefect.logging import get_run_logger

from fraud_detection.config import settings
from fraud_detection.data.load import load_paysim
from fraud_detection.data.splits import temporal_holdout_split, walk_forward_splits
from fraud_detection.evaluation.metrics import (
    expected_cost,
    pr_auc,
    precision_at_k,
    threshold_for_alert_volume,
)
from fraud_detection.features.tabular import build_tabular_features
from fraud_detection.models.lightgbm import LightGBMConfig, train_lightgbm


@task(retries=2, retry_delay_seconds=10)
def ingest(sample_frac: float | None = None) -> pl.DataFrame:
    """Load PaySim from disk."""
    return load_paysim(sample_frac=sample_frac)


@task
def featurize(df: pl.DataFrame) -> pl.DataFrame:
    """Build tabular features."""
    return build_tabular_features(df)


@task
def train_and_eval(
    df: pl.DataFrame, feature_cols: list[str], categorical_cols: list[str]
) -> dict[str, float]:
    """Walk-forward train and aggregate metrics across folds."""
    logger = get_run_logger()
    metrics_per_fold: list[dict[str, float]] = []

    for train, val, fold in walk_forward_splits(df, n_folds=3, val_window=48):
        logger.info(fold.describe())
        booster = train_lightgbm(
            train,
            val,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            config=LightGBMConfig(),
        )
        y_val = val["isFraud"].to_numpy()
        y_score = booster.predict(val.select(feature_cols).to_pandas())
        threshold = threshold_for_alert_volume(y_score, target_rate=0.005)
        cost = expected_cost(y_val, y_score, threshold)

        metrics_per_fold.append(
            {
                "fold": fold.fold_id,
                "pr_auc": pr_auc(y_val, y_score),
                "p_at_0.1pct": precision_at_k(y_val, y_score, 0.001),
                "p_at_0.5pct": precision_at_k(y_val, y_score, 0.005),
                "p_at_1pct": precision_at_k(y_val, y_score, 0.01),
                "expected_cost": cost.total_cost,
                "n_alerts": cost.n_alerts,
            }
        )

    aggregate = {
        f"mean_{k}": sum(m[k] for m in metrics_per_fold) / len(metrics_per_fold)
        for k in metrics_per_fold[0]
        if k != "fold"
    }
    return aggregate


@task
def register_model(metrics: dict[str, float], model_name: str = "fraud-detection") -> None:
    """Log final metrics to MLflow and tag the run."""
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    with mlflow.start_run(run_name=f"{model_name}-walkforward"):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("stage", "candidate")


@flow(name="fraud-detection-train")
def train_flow(sample_frac: float | None = 0.1) -> dict[str, float]:
    """End-to-end training flow.

    Args:
        sample_frac: Use a stratified sample for fast iteration. Set to None
            to train on the full dataset.

    Returns:
        Aggregated walk-forward metrics.
    """
    raw = ingest(sample_frac=sample_frac)
    feats = featurize(raw)

    feature_cols = [
        "step",
        "type",
        "amount",
        "oldbalance",
        "newbalance",
        "oldbalanceDest",
        "newbalanceDest",
        "balance_drained",
        "orig_balance_delta",
        "dest_balance_delta",
        "amount_to_orig_balance",
        "dest_balance_zero",
        "hour_of_day",
        "is_night",
        "orig_is_merchant",
        "dest_is_merchant",
    ]
    categorical_cols = ["type"]

    metrics = train_and_eval(feats, feature_cols, categorical_cols)
    register_model(metrics)
    return metrics


if __name__ == "__main__":
    train_flow()
