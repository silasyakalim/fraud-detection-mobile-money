#!/usr/bin/env python
"""Generate synthetic PaySim-shaped data for demos and tests.

Reproduces the structural properties of PaySim (Lopez-Rojas et al., EMSS
2016):

- Transaction type mix: ~22% CASH-IN, ~35% CASH-OUT, ~1% DEBIT, ~34%
  PAYMENT, ~8% TRANSFER.
- Fraud rate ~0.13%, concentrated in TRANSFER and CASH-OUT.
- Fraud pattern: drain originator account via TRANSFER to a fresh mule,
  then CASH-OUT from the mule. Generated here as paired (TRANSFER,
  CASH-OUT) sequences with matching amounts.
- The PaySim quirk where many fraud rows have ``newbalanceDest = 0``.
- 30 days of activity (step in [1, 744] hourly).

For real analysis, use the actual PaySim CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_paysim(
    n_rows: int = 500_000,
    fraud_rate: float = 0.0013,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic PaySim data."""
    rng = np.random.default_rng(seed)

    # Transaction type mix
    types = rng.choice(
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        size=n_rows,
        p=[0.22, 0.35, 0.01, 0.34, 0.08],
    )

    # Time steps spread across 30 days with daily peaks
    base_steps = rng.integers(1, 745, size=n_rows)
    # Slight bias toward business hours within each day
    hour_offset = rng.normal(0, 4, size=n_rows).astype(int)
    steps = np.clip(base_steps + hour_offset, 1, 744).astype(np.int32)

    # Account names: Customers C... and Merchants M...
    n_customers = max(10_000, n_rows // 50)
    n_merchants = max(2_000, n_rows // 250)
    customer_ids = np.array([f"C{i}" for i in range(n_customers)])
    merchant_ids = np.array([f"M{i}" for i in range(n_merchants)])

    account_id = customer_ids[rng.integers(0, n_customers, size=n_rows)]

    # Destinations: PAYMENT goes to merchants, others to customers
    is_payment = types == "PAYMENT"
    account_dest = np.empty(n_rows, dtype=object)
    n_payments = int(is_payment.sum())
    account_dest[is_payment] = merchant_ids[rng.integers(0, n_merchants, size=n_payments)]
    account_dest[~is_payment] = customer_ids[rng.integers(0, n_customers, size=n_rows - n_payments)]

    # Amounts: log-normal with type-specific scales
    log_mean = np.where(
        types == "TRANSFER", 11.5,
        np.where(types == "CASH_OUT", 10.5,
        np.where(types == "CASH_IN", 9.5,
        np.where(types == "PAYMENT", 8.5, 7.0)))
    )
    amount = np.exp(rng.normal(log_mean, 1.2)).round(2)
    amount = np.clip(amount, 0.01, 1e7)

    # Originator balances: log-normal, bounded
    old_balance = np.exp(rng.normal(10.5, 1.5, size=n_rows)).round(2)
    old_balance = np.clip(old_balance, 0.0, 1e7)

    # Compute newbalance based on transaction direction
    new_balance = np.where(
        np.isin(types, ["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT"]),
        np.maximum(0, old_balance - amount),
        old_balance + amount,  # CASH_IN
    ).round(2)

    # Destination balances: noisy
    old_balance_dest = np.where(
        np.isin(types, ["PAYMENT"]),  # Merchants don't have user-style balances
        0.0,
        np.exp(rng.normal(9.5, 1.8, size=n_rows)).round(2),
    )
    old_balance_dest = np.clip(old_balance_dest, 0.0, 1e7)
    new_balance_dest = np.where(
        types == "PAYMENT",
        0.0,
        old_balance_dest + amount,
    ).round(2)

    is_fraud = np.zeros(n_rows, dtype=np.int8)
    is_flagged = np.zeros(n_rows, dtype=np.int8)

    # ---- Inject fraud ----
    n_fraud_pairs = int(n_rows * fraud_rate / 2)  # each fraud is a (TRANSFER, CASH_OUT) pair
    if n_fraud_pairs > 0:
        # Convert pairs of indices to fraud
        # Fraud pattern: pick rich originator, drain via TRANSFER to a fresh mule,
        # then mule CASH_OUTs the same amount shortly after.
        fraud_pair_indices = rng.choice(
            np.arange(n_rows - 1), size=n_fraud_pairs, replace=False
        )
        fraud_pair_indices = np.sort(fraud_pair_indices)

        for idx in fraud_pair_indices:
            mule = f"C_mule_{idx}"
            victim_balance = float(rng.uniform(50_000, 5_000_000))
            # Mix of full drains and partial drains so the signal isn't deterministic
            full_drain = rng.random() < 0.75
            drain_fraction = 1.0 if full_drain else float(rng.uniform(0.55, 0.95))
            drain_amount = round(victim_balance * drain_fraction, 2)
            t = int(rng.integers(1, 744))

            types[idx] = "TRANSFER"
            steps[idx] = t
            amount[idx] = drain_amount
            old_balance[idx] = victim_balance
            new_balance[idx] = round(max(0.0, victim_balance - drain_amount), 2)
            account_dest[idx] = mule
            old_balance_dest[idx] = 0.0
            # PaySim quirk: dest balance often stays at 0 for fraud, but not always
            new_balance_dest[idx] = 0.0 if rng.random() < 0.85 else drain_amount
            is_fraud[idx] = 1

            # Second leg: CASH_OUT from mule, mostly the same amount with small skim
            j = idx + 1
            skim = float(rng.uniform(0.92, 1.0))
            types[j] = "CASH_OUT"
            steps[j] = min(t + int(rng.integers(0, 6)), 744)
            amount[j] = round(drain_amount * skim, 2)
            account_id[j] = mule
            old_balance[j] = drain_amount
            new_balance[j] = round(drain_amount - drain_amount * skim, 2)
            old_balance_dest[j] = 0.0
            new_balance_dest[j] = 0.0 if rng.random() < 0.7 else round(drain_amount * skim, 2)
            is_fraud[j] = 1

        # Inject "near-fraud" legitimate transfers (e.g. account closures)
        # so fraud isn't trivially separable from large legit transfers.
        candidate_idx = np.where((types == "TRANSFER") & (is_fraud == 0))[0]
        n_lookalikes = min(n_fraud_pairs * 5, len(candidate_idx))
        if n_lookalikes > 0:
            lookalike_idx = rng.choice(candidate_idx, size=n_lookalikes, replace=False)
            for idx in lookalike_idx:
                balance = float(rng.uniform(20_000, 500_000))
                drain_pct = float(rng.uniform(0.85, 1.0))
                old_balance[idx] = balance
                amount[idx] = round(balance * drain_pct, 2)
                new_balance[idx] = round(max(0.0, balance - amount[idx]), 2)

        # The legacy isFlaggedFraud rule: amounts > 200K on TRANSFER
        flag_mask = (types == "TRANSFER") & (amount > 200_000) & (is_fraud == 1)
        is_flagged[flag_mask] = 1

    df = pd.DataFrame(
        {
            "step": steps,
            "type": pd.Categorical(types, categories=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]),
            "amount": amount.astype(np.float64),
            "accountID": account_id,
            "oldbalance": old_balance.astype(np.float64),
            "newbalance": new_balance.astype(np.float64),
            "accountDest": account_dest,
            "oldbalanceDest": old_balance_dest.astype(np.float64),
            "newbalanceDest": new_balance_dest.astype(np.float64),
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )
    return df.sort_values("step").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--fraud-rate", type=float, default=0.0013)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv",
    )
    args = parser.parse_args()

    print(f"Generating {args.rows:,} rows with fraud rate {args.fraud_rate:.4%}...")
    df = generate_paysim(n_rows=args.rows, fraud_rate=args.fraud_rate, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")
    print(f"  Fraud rate: {df['isFraud'].mean():.4%}")
    print(f"  Fraud by type:")
    print(df.groupby("type", observed=True)["isFraud"].agg(["sum", "mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
