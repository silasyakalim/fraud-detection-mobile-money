"""Time-aware splits.

Two strategies:

1. ``walk_forward_splits`` — expanding-window CV. Each fold trains on data
   up to time t and validates on a window (t, t+w].
2. ``temporal_holdout_split`` — train/val/test cut at fixed time boundaries.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import polars as pl


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold with explicit train/val ranges on the time column."""

    fold_id: int
    train_start: int
    train_end: int  # inclusive
    val_start: int
    val_end: int  # inclusive

    def describe(self) -> str:
        return (
            f"fold {self.fold_id}: "
            f"train=[{self.train_start}, {self.train_end}] "
            f"val=[{self.val_start}, {self.val_end}]"
        )


def walk_forward_splits(
    df: pl.DataFrame,
    *,
    time_col: str = "step",
    n_folds: int = 5,
    val_window: int = 24,
    min_train_window: int = 168,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame, Fold]]:
    """Yield expanding-window train/val splits.

    Args:
        df: Source frame, ordered or unordered. Must contain ``time_col``.
        time_col: Name of the integer time column (PaySim's ``step`` is hourly).
        n_folds: Number of folds to generate.
        val_window: Size of each validation window in time units.
        min_train_window: Minimum size of the initial training window.

    Yields:
        ``(train_df, val_df, fold_metadata)`` triples.

    Example:
        >>> for train, val, meta in walk_forward_splits(df, n_folds=3):
        ...     model.fit(train)
        ...     score = evaluate(model, val)
        ...     log.info(meta.describe(), score=score)
    """
    if time_col not in df.columns:
        raise KeyError(f"time column {time_col!r} not in dataframe")

    t_min = cast(int, df[time_col].min() or 0)
    t_max = cast(int, df[time_col].max() or 0)
    span = t_max - t_min + 1

    if span < min_train_window + n_folds * val_window:
        raise ValueError(
            f"Time span {span} too small for {n_folds} folds of {val_window} "
            f"with min_train_window={min_train_window}"
        )

    # Stride between fold validation starts so folds tile the remaining span.
    remainder = span - min_train_window
    step = max(val_window, remainder // n_folds)

    for i in range(n_folds):
        train_start = t_min
        train_end = t_min + min_train_window + i * step - 1
        val_start = train_end + 1
        val_end = min(val_start + val_window - 1, t_max)

        if val_start > t_max:
            break

        train_df = df.filter((pl.col(time_col) >= train_start) & (pl.col(time_col) <= train_end))
        val_df = df.filter((pl.col(time_col) >= val_start) & (pl.col(time_col) <= val_end))

        yield (
            train_df,
            val_df,
            Fold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            ),
        )


def temporal_holdout_split(
    df: pl.DataFrame,
    *,
    time_col: str = "step",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Simple time-ordered train/val/test split.

    Test fraction is ``1 - train_frac - val_frac``.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must each be in (0,1) and sum < 1")

    t_min = cast(int, df[time_col].min() or 0)
    t_max = cast(int, df[time_col].max() or 0)
    span = t_max - t_min + 1

    train_cut = t_min + int(span * train_frac)
    val_cut = t_min + int(span * (train_frac + val_frac))

    train = df.filter(pl.col(time_col) < train_cut)
    val = df.filter((pl.col(time_col) >= train_cut) & (pl.col(time_col) < val_cut))
    test = df.filter(pl.col(time_col) >= val_cut)
    return train, val, test
