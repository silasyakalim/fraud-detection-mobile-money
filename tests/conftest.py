"""Shared pytest fixtures. Synthetic data so tests don't need real PaySim."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_paysim(rng: np.random.Generator) -> pl.DataFrame:
    """Generate a small synthetic PaySim-shaped frame for unit tests."""
    n = 5000
    types = rng.choice(
        ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        size=n,
        p=[0.22, 0.35, 0.01, 0.34, 0.08],
    )
    amount = rng.exponential(scale=20_000, size=n)
    old_orig = rng.exponential(scale=100_000, size=n)
    new_orig = np.maximum(0, old_orig - amount)
    old_dest = rng.exponential(scale=80_000, size=n)
    new_dest = old_dest + amount

    # ~1% fraud as TRANSFER/CASH-OUT rows that fully drain the source.
    is_fraud = np.zeros(n, dtype=np.int8)
    fraud_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    is_fraud[fraud_idx] = 1
    types = np.array(types)
    types[fraud_idx] = rng.choice(["TRANSFER", "CASH-OUT"], size=len(fraud_idx))
    amount[fraud_idx] = old_orig[fraud_idx]
    new_orig[fraud_idx] = 0

    return pl.DataFrame(
        {
            "step": rng.integers(1, 745, size=n).astype(np.int32),
            "type": pl.Series(types).cast(pl.Categorical),
            "amount": amount,
            "accountID": [f"C{i}" for i in rng.integers(1, 1000, size=n)],
            "oldbalance": old_orig,
            "newbalance": new_orig,
            "accountDest": [
                ("M" if rng.random() < 0.4 else "C") + str(i)
                for i in rng.integers(1, 800, size=n)
            ],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": np.zeros(n, dtype=np.int8),
        }
    )
