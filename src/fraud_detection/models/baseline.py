"""Logistic regression baseline."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import settings
from fraud_detection.logging import get_logger

log = get_logger(__name__)


def train_baseline(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "isFraud",
) -> Pipeline:
    """Fit a class-weighted logistic regression on the given features."""
    X = train_df.select(feature_cols).to_numpy()
    y = train_df[target_col].to_numpy()

    pipe: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=500,
                    random_state=settings.model.random_seed,
                    n_jobs=settings.model.n_jobs,
                ),
            ),
        ]
    )
    log.info("fitting baseline lr", n=len(y), positive_rate=float(np.mean(y)))
    pipe.fit(X, y)
    return pipe
