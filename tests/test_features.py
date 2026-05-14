"""Tests for tabular feature engineering."""

from __future__ import annotations

import polars as pl

from fraud_detection.features.tabular import (
    add_account_type_features,
    add_balance_features,
    add_temporal_features,
    build_tabular_features,
)


def test_balance_features_present(synthetic_paysim: pl.DataFrame) -> None:
    out = add_balance_features(synthetic_paysim.lazy()).collect()
    for col in [
        "balance_drained",
        "orig_balance_delta",
        "dest_balance_delta",
        "amount_to_orig_balance",
        "dest_balance_zero",
    ]:
        assert col in out.columns


def test_balance_drained_logic(synthetic_paysim: pl.DataFrame) -> None:
    """balance_drained should fire when newbalance is 0 and old was > 0."""
    out = add_balance_features(synthetic_paysim.lazy()).collect()
    drained = out.filter(pl.col("balance_drained"))
    assert (drained["newbalance"] == 0).all()
    assert (drained["oldbalance"] > 0).all()


def test_temporal_features_in_range(synthetic_paysim: pl.DataFrame) -> None:
    out = add_temporal_features(synthetic_paysim.lazy()).collect()
    assert out["hour_of_day"].min() >= 0
    assert out["hour_of_day"].max() <= 23


def test_account_type_flags(synthetic_paysim: pl.DataFrame) -> None:
    out = add_account_type_features(synthetic_paysim.lazy()).collect()
    merchants = out.filter(pl.col("dest_is_merchant"))
    assert merchants["accountDest"].str.starts_with("M").all()


def test_build_tabular_features_idempotent(synthetic_paysim: pl.DataFrame) -> None:
    """Re-running on its own output should be a no-op for column count."""
    once = build_tabular_features(synthetic_paysim)
    twice = build_tabular_features(once)
    assert set(twice.columns) == set(once.columns)
