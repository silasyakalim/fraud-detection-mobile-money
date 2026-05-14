"""Tests for time-aware splits."""

from __future__ import annotations

import polars as pl
import pytest

from fraud_detection.data.splits import temporal_holdout_split, walk_forward_splits


def test_walk_forward_splits_no_leakage(synthetic_paysim: pl.DataFrame) -> None:
    """Validation steps must always be strictly after training steps."""
    folds = list(walk_forward_splits(synthetic_paysim, n_folds=3, val_window=24))
    assert len(folds) >= 1

    for train, val, fold in folds:
        train_max = int(train["step"].max() or 0)
        val_min = int(val["step"].min() or 0)
        assert train_max < val_min, fold.describe()


def test_walk_forward_splits_expanding_window(synthetic_paysim: pl.DataFrame) -> None:
    """Each successive fold should see at least as much training data."""
    folds = list(walk_forward_splits(synthetic_paysim, n_folds=3, val_window=24))
    train_sizes = [t.height for t, _, _ in folds]
    assert all(b >= a for a, b in zip(train_sizes, train_sizes[1:], strict=False))


def test_walk_forward_rejects_short_span(synthetic_paysim: pl.DataFrame) -> None:
    """Should raise if the time span can't accommodate the requested folds."""
    with pytest.raises(ValueError, match="too small"):
        list(
            walk_forward_splits(
                synthetic_paysim, n_folds=20, val_window=200, min_train_window=600
            )
        )


def test_temporal_holdout_split_shapes(synthetic_paysim: pl.DataFrame) -> None:
    train, val, test = temporal_holdout_split(synthetic_paysim, train_frac=0.7, val_frac=0.15)
    total = train.height + val.height + test.height
    assert total == synthetic_paysim.height
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()


def test_temporal_holdout_rejects_bad_fractions(synthetic_paysim: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="must each be in"):
        temporal_holdout_split(synthetic_paysim, train_frac=0.9, val_frac=0.2)
