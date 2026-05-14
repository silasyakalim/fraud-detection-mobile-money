"""LightGBM trainer with cost-aware class weighting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from fraud_detection.config import settings
from fraud_detection.logging import get_logger

log = get_logger(__name__)


@dataclass
class LightGBMConfig:
    """Hyperparameters for the LightGBM fraud classifier."""

    objective: str = "binary"
    metric: str = "average_precision"
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    verbose: int = -1
    seed: int = field(default_factory=lambda: settings.model.random_seed)
    n_jobs: int = field(default_factory=lambda: settings.model.n_jobs)

    def to_params(self) -> dict[str, Any]:
        """Render to the dict shape LightGBM's training API expects."""
        params = {
            "objective": self.objective,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "verbose": self.verbose,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
        }
        return params


def cost_aware_scale_pos_weight(
    y: np.ndarray,
    *,
    cost_fn: float,  # noqa: ARG001
    cost_fp: float,  # noqa: ARG001
    cost_aware_factor: float = 1.0,
) -> float:
    """Compute ``scale_pos_weight = (n_neg / n_pos) * cost_aware_factor``.

    At PaySim's 0.13% fraud rate, ``n_neg/n_pos`` alone is already a strong
    signal; multiplying by the full ``cost_fn/cost_fp`` ratio tends to over-
    weight positives and cause the booster to stop after a few trees. Cost
    asymmetry is easier to apply at threshold-selection time. The factor
    knob lets callers blend the two; default 1.0 = standard class weighting.

    Args:
        y: Binary target.
        cost_fn: Reserved for future use.
        cost_fp: Reserved for future use.
        cost_aware_factor: Multiplier on the class-imbalance correction.

    Returns:
        ``scale_pos_weight`` value.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive samples in y")
    return (n_neg / n_pos) * cost_aware_factor


def train_lightgbm(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "isFraud",
    config: LightGBMConfig | None = None,
    categorical_cols: list[str] | None = None,
) -> lgb.Booster:
    """Train a single LightGBM model with cost-aware weighting.

    Args:
        train_df: Training set (Polars).
        val_df: Validation set for early stopping.
        feature_cols: Feature column names.
        target_col: Binary target column name.
        config: Hyperparameter config; defaults applied if None.
        categorical_cols: Subset of ``feature_cols`` to treat as categorical.

    Returns:
        Trained LightGBM Booster.
    """
    cfg = config or LightGBMConfig()
    cat_cols = categorical_cols or []

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df[target_col].to_numpy()
    X_val = val_df.select(feature_cols).to_pandas()
    y_val = val_df[target_col].to_numpy()

    params = cfg.to_params()
    params["scale_pos_weight"] = cost_aware_scale_pos_weight(
        y_train,
        cost_fn=settings.cost.false_negative,
        cost_fp=settings.cost.false_positive,
    )
    log.info(
        "training lightgbm",
        n_train=len(y_train),
        n_val=len(y_val),
        train_fraud_rate=float(y_train.mean()),
        scale_pos_weight=params["scale_pos_weight"],
    )

    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_cols or "auto"
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, reference=train_set, categorical_feature=cat_cols or "auto"
    )

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    log.info("training complete", best_iter=booster.best_iteration)
    return booster
