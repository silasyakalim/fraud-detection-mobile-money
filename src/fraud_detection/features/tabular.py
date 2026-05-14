"""Tabular feature engineering for PaySim.

Groups:

- ``balance_*`` for the account-drainage pattern (fraud transfers that empty
  the originator).
- ``velocity_*`` for rapid-fire transfers within short windows (common in
  account takeover).
- ``hour_of_day``, ``is_night`` for time-of-day effects.
- ``*_is_merchant`` for the account-prefix convention (M vs C).

Functions return ``LazyFrame``s so they compose cheaply.
"""

from __future__ import annotations

import polars as pl


def add_balance_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add features derived from before/after balances.

    PaySim's balance fields are noisy; ``newbalanceDest`` is often 0 for
    fraud even when it shouldn't be. We surface that as a binary flag rather
    than imputing.
    """
    return lf.with_columns(
        balance_drained=(pl.col("newbalance") == 0)
        & (pl.col("oldbalance") > 0),
        orig_balance_delta=pl.col("oldbalance") - pl.col("newbalance"),
        dest_balance_delta=pl.col("newbalanceDest") - pl.col("oldbalanceDest"),
        amount_to_orig_balance=(
            pl.col("amount") / pl.when(pl.col("oldbalance") > 0)
            .then(pl.col("oldbalance"))
            .otherwise(1.0)
        ),
        dest_balance_zero=(pl.col("oldbalanceDest") == 0)
        & (pl.col("newbalanceDest") == 0),
    )


def add_temporal_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Hour-of-day and day-of-month derived from ``step``."""
    return lf.with_columns(
        hour_of_day=pl.col("step") % 24,
        day_of_simulation=(pl.col("step") - 1) // 24 + 1,
        is_night=((pl.col("step") % 24) < 6) | ((pl.col("step") % 24) >= 22),
    )


def add_account_type_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Flag merchant accounts based on the M/C prefix convention."""
    return lf.with_columns(
        orig_is_merchant=pl.col("accountID").str.starts_with("M"),
        dest_is_merchant=pl.col("accountDest").str.starts_with("M"),
    )


def add_velocity_features(
    lf: pl.LazyFrame, *, window_hours: tuple[int, ...] = (1, 6, 24)
) -> pl.LazyFrame:
    """Rolling counts and sums per originator over recent windows.

    Computed causally: at time t only events with step <= t are visible.
    Input is sorted by ``step`` to make this work regardless of insertion
    order.
    """
    lf = lf.sort("step")
    for w in window_hours:
        lf = lf.with_columns(
            pl.col("amount")
            .rolling_sum_by("step", window_size=f"{w}h", closed="right")
            .over("accountID")
            .alias(f"orig_amount_sum_{w}h"),
            pl.col("amount")
            .rolling_mean_by("step", window_size=f"{w}h", closed="right")
            .over("accountID")
            .alias(f"orig_amount_mean_{w}h"),
            pl.lit(1)
            .rolling_sum_by("step", window_size=f"{w}h", closed="right")
            .over("accountID")
            .alias(f"orig_tx_count_{w}h"),
        )
    return lf


def build_tabular_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Apply all tabular feature transforms in order.

    Args:
        df: Raw PaySim frame. Either eager or lazy.

    Returns:
        Eager DataFrame with original + engineered columns.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    lf = add_balance_features(lf)
    lf = add_temporal_features(lf)
    lf = add_account_type_features(lf)
    # Velocity features are expensive on the full dataset; enable selectively.
    return lf.collect()
