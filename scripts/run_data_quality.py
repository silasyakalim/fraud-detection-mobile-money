#!/usr/bin/env python
"""Run data quality checks on the loaded PaySim data.

Outputs findings to ``docs/data_quality.md`` and ``docs/data_quality.json``.
Exits with non-zero status if any *critical* check fails (nulls in target,
schema violations) so it can be wired into CI later.

Checks performed:
    1. Schema and dtypes match expected.
    2. Missing values per column.
    3. Exact duplicate rows.
    4. Balance arithmetic consistency.
    5. Outlier detection (negative amounts/balances, zero amounts).
    6. Account ID prefix conventions (C... for customers, M... for merchants).
    7. Transaction type values match the documented enum.
    8. Label consistency (isFlaggedFraud subset of isFraud).
    9. Class balance summary.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_detection.data.load import load_paysim  # noqa: E402

EXPECTED_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
EXPECTED_DTYPES = {
    "step": pl.Int32,
    "amount": pl.Float64,
    "oldbalance": pl.Float64,
    "newbalance": pl.Float64,
    "oldbalanceDest": pl.Float64,
    "newbalanceDest": pl.Float64,
    "isFraud": pl.Int8,
    "isFlaggedFraud": pl.Int8,
}


@dataclass
class Check:
    """One data-quality check."""

    name: str
    passed: bool
    severity: str  # "critical" | "warning" | "info"
    detail: str
    counts: dict[str, int | float] = field(default_factory=dict)


def check_schema(df: pl.DataFrame) -> Check:
    issues = []
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            issues.append(f"missing column {col}")
            continue
        actual = df.schema[col]
        if actual != expected:
            issues.append(f"{col}: got {actual}, expected {expected}")
    return Check(
        name="schema_and_dtypes",
        passed=not issues,
        severity="critical" if issues else "info",
        detail="; ".join(issues) if issues else "all columns present with expected dtypes",
    )


def check_missing(df: pl.DataFrame) -> Check:
    nulls = df.null_count().row(0, named=True)
    total_nulls = sum(nulls.values())
    return Check(
        name="missing_values",
        passed=total_nulls == 0,
        severity="critical" if total_nulls else "info",
        detail=f"total null cells: {total_nulls}",
        counts={k: int(v) for k, v in nulls.items() if v > 0} or {"all_columns": 0},
    )


def check_duplicates(df: pl.DataFrame) -> Check:
    n_unique = df.unique().height
    n_dupes = df.height - n_unique
    return Check(
        name="exact_duplicates",
        passed=n_dupes == 0,
        severity="warning" if n_dupes else "info",
        detail=f"{n_dupes:,} exact duplicate rows out of {df.height:,}",
        counts={"duplicate_rows": n_dupes, "unique_rows": n_unique},
    )


def check_balance_arithmetic(df: pl.DataFrame) -> Check:
    """Check whether oldbalance, newbalance, amount add up.

    For outflow types: ``newbalance ≈ oldbalance - amount``.
    For CASH_IN: ``newbalance ≈ oldbalance + amount``.

    Inconsistencies aren't necessarily bugs; PaySim documents balance-field
    noise that the modeling code uses as a feature. Reported as info, not
    a failure.
    """
    eps = 0.01
    expected = pl.when(pl.col("type") == "CASH_IN").then(
        pl.col("oldbalance") + pl.col("amount")
    ).otherwise(
        pl.col("oldbalance") - pl.col("amount")
    )
    df_ann = df.with_columns(
        balance_mismatch=(pl.col("newbalance") - expected).abs() > eps
    )
    n_mismatch = int(df_ann["balance_mismatch"].sum())
    n_mismatch_fraud = int(
        df_ann.filter(pl.col("balance_mismatch") & (pl.col("isFraud") == 1)).height
    )
    n_mismatch_legit = n_mismatch - n_mismatch_fraud
    n_fraud = int(df_ann["isFraud"].sum())

    pct_fraud_mismatched = n_mismatch_fraud / max(1, n_fraud)
    return Check(
        name="balance_arithmetic_consistency",
        passed=True,  # We don't fail on this — it's informational
        severity="info",
        detail=(
            f"{n_mismatch:,}/{df.height:,} rows have inconsistent balance arithmetic. "
            f"{n_mismatch_fraud:,}/{n_fraud:,} fraud rows ({pct_fraud_mismatched:.1%}) "
            f"are inconsistent vs {n_mismatch_legit:,} legit rows. "
            "Inconsistency is itself a fraud signal — captured by `dest_balance_zero` "
            "and similar features."
        ),
        counts={
            "total_inconsistent": n_mismatch,
            "inconsistent_fraud": n_mismatch_fraud,
            "inconsistent_legit": n_mismatch_legit,
            "fraud_inconsistent_pct": round(pct_fraud_mismatched, 4),
        },
    )


def check_negative_or_zero(df: pl.DataFrame) -> Check:
    neg_amount = int(df.filter(pl.col("amount") < 0).height)
    zero_amount = int(df.filter(pl.col("amount") == 0).height)
    neg_balance = int(
        df.filter((pl.col("oldbalance") < 0) | (pl.col("newbalance") < 0)).height
    )
    issues = []
    if neg_amount:
        issues.append(f"{neg_amount} negative amounts")
    if neg_balance:
        issues.append(f"{neg_balance} negative balances")
    return Check(
        name="negative_or_zero_values",
        passed=neg_amount == 0 and neg_balance == 0,
        severity="critical" if (neg_amount or neg_balance) else "info",
        detail=(
            "; ".join(issues) if issues
            else f"no negative values; {zero_amount:,} rows with amount==0"
        ),
        counts={
            "negative_amount": neg_amount,
            "negative_balance": neg_balance,
            "zero_amount": zero_amount,
        },
    )


def check_account_conventions(df: pl.DataFrame) -> Check:
    """Originators should always be customers (C...).

    Destinations may be customers (C...) or merchants (M...). Anything else is
    a schema violation.
    """
    bad_orig = int(df.filter(~pl.col("accountID").str.starts_with("C")).height)
    bad_dest = int(
        df.filter(
            ~pl.col("accountDest").str.starts_with("C")
            & ~pl.col("accountDest").str.starts_with("M")
        ).height
    )
    issues = []
    if bad_orig:
        issues.append(f"{bad_orig} originators don't start with C")
    if bad_dest:
        issues.append(f"{bad_dest} destinations don't start with C or M")
    return Check(
        name="account_id_conventions",
        passed=bad_orig == 0 and bad_dest == 0,
        severity="warning" if (bad_orig or bad_dest) else "info",
        detail="; ".join(issues) if issues else "all account IDs follow C/M convention",
        counts={"bad_originators": bad_orig, "bad_destinations": bad_dest},
    )


def check_transaction_types(df: pl.DataFrame) -> Check:
    actual = set(df["type"].unique().to_list())
    unknown = actual - EXPECTED_TYPES
    missing = EXPECTED_TYPES - actual
    issues = []
    if unknown:
        issues.append(f"unexpected types: {unknown}")
    if missing:
        issues.append(f"missing types: {missing}")
    return Check(
        name="transaction_type_enum",
        passed=not unknown,
        severity="critical" if unknown else "info",
        detail="; ".join(issues) if issues else f"all 5 documented types present: {sorted(actual)}",
    )


def check_label_consistency(df: pl.DataFrame) -> Check:
    """isFlaggedFraud=1 should always imply isFraud=1.

    The reverse is not required (the legacy rule misses most fraud).
    """
    violations = int(
        df.filter((pl.col("isFlaggedFraud") == 1) & (pl.col("isFraud") == 0)).height
    )
    return Check(
        name="label_consistency",
        passed=violations == 0,
        severity="critical" if violations else "info",
        detail=(
            f"{violations} rows where isFlaggedFraud=1 but isFraud=0"
            if violations else "isFlaggedFraud is a strict subset of isFraud (as expected)"
        ),
        counts={"flag_without_fraud": violations},
    )


def check_class_balance(df: pl.DataFrame) -> Check:
    n = df.height
    n_fraud = int(df["isFraud"].sum())
    rate = n_fraud / n
    by_type = df.group_by("type").agg(
        pl.len().alias("count"),
        pl.col("isFraud").sum().alias("fraud_count"),
        pl.col("isFraud").mean().alias("fraud_rate"),
    ).sort("type")

    return Check(
        name="class_balance",
        passed=True,
        severity="info",
        detail=f"fraud rate {rate:.4%} ({n_fraud:,}/{n:,})",
        counts={
            "total_rows": n,
            "fraud_rows": n_fraud,
            "fraud_rate": round(rate, 6),
            "by_type": {
                row["type"]: {"count": row["count"], "fraud_rate": round(row["fraud_rate"] or 0, 6)}
                for row in by_type.iter_rows(named=True)
            },
        },
    )


CHECKS = [
    check_schema,
    check_missing,
    check_duplicates,
    check_balance_arithmetic,
    check_negative_or_zero,
    check_account_conventions,
    check_transaction_types,
    check_label_consistency,
    check_class_balance,
]


def write_markdown(checks: list[Check], path: Path) -> None:
    lines = ["# Data quality report\n"]
    lines.append("Generated by `scripts/run_data_quality.py`. ")
    lines.append("Critical issues fail CI; warnings are surfaced for review.\n\n")

    n_critical = sum(1 for c in checks if not c.passed and c.severity == "critical")
    n_warning = sum(1 for c in checks if not c.passed and c.severity == "warning")
    lines.append("## Summary\n\n")
    lines.append(f"- Total checks: {len(checks)}\n")
    lines.append(f"- Critical failures: {n_critical}\n")
    lines.append(f"- Warnings: {n_warning}\n")
    lines.append(f"- Passed/info: {len(checks) - n_critical - n_warning}\n\n")

    lines.append("## Checks\n\n")
    lines.append("| Check | Status | Severity | Detail |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for c in checks:
        icon = "✅" if c.passed else ("❌" if c.severity == "critical" else "⚠️")
        lines.append(f"| `{c.name}` | {icon} {'pass' if c.passed else 'fail'} | {c.severity} | {c.detail} |\n")
    lines.append("\n")

    # Detailed counts
    lines.append("## Detailed counts\n\n")
    for c in checks:
        if c.counts:
            lines.append(f"### `{c.name}`\n\n")
            lines.append("```json\n")
            lines.append(json.dumps(c.counts, indent=2))
            lines.append("\n```\n\n")

    path.write_text("".join(lines))


def main() -> int:
    print("Loading data...")
    df = load_paysim()
    print(f"Loaded {df.height:,} rows.\n")

    results: list[Check] = []
    for check_fn in CHECKS:
        c = check_fn(df)
        icon = "✓" if c.passed else ("✗" if c.severity == "critical" else "!")
        print(f"  {icon} [{c.severity:8}] {c.name}: {c.detail}")
        results.append(c)

    # Write outputs
    out_md = REPO_ROOT / "docs" / "data_quality.md"
    out_json = REPO_ROOT / "docs" / "data_quality.json"
    write_markdown(results, out_md)
    out_json.write_text(json.dumps(
        [{"name": c.name, "passed": c.passed, "severity": c.severity, "detail": c.detail, "counts": c.counts} for c in results],
        indent=2,
    ))

    print(f"\nReport written to {out_md}")
    print(f"JSON written to {out_json}")

    # Fail if any critical check failed
    n_critical_failures = sum(1 for c in results if not c.passed and c.severity == "critical")
    if n_critical_failures:
        print(f"\n❌ {n_critical_failures} critical check(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
