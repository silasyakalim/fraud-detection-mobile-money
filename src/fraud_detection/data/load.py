"""PaySim loaders.

Columns:
  step           int32  - hour of simulation (1-744 over 30 days)
  type           cat    - CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
  amount         f64    - transaction amount
  nameOrig      str    - originator account
  oldbalanceOrg     f64    - originator balance before
  newbalanceOrig     f64    - originator balance after
  nameDest    str    - destination account
  oldbalanceDest f64    - destination balance before
  newbalanceDest f64    - destination balance after
  isFraud        i8     - target label (~0.13% positive rate)
  isFlaggedFraud i8     - legacy rule-based flag
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fraud_detection.config import settings
from fraud_detection.logging import get_logger

log = get_logger(__name__)


_SCHEMA = {
    "step": pl.Int32,
    "type": pl.Categorical,
    "amount": pl.Float64,
    "nameOrig": pl.String,
    "oldbalanceOrg": pl.Float64,
    "newbalanceOrig": pl.Float64,
    "nameDest": pl.String,
    "oldbalanceDest": pl.Float64,
    "newbalanceDest": pl.Float64,
    "isFraud": pl.Int8,
    "isFlaggedFraud": pl.Int8,
}


def load_paysim(
    path: Path | None = None,
    *,
    sample_frac: float | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Load PaySim as a Polars DataFrame.

    Args:
        path: Override the configured data path.
        sample_frac: If set, return a stratified sample by ``isFraud`` of this
            fraction. Useful for development on the full 6.3M-row dataset.
        seed: RNG seed for sampling.

    Returns:
        Polars DataFrame with typed columns.
    """
    csv_path = path or settings.data.paysim_path
    if not csv_path.exists():
        raise FileNotFoundError(f"PaySim CSV not found at {csv_path}. Run `make download-data`.")

    log.info("loading paysim", path=str(csv_path))
    df = pl.read_csv(csv_path, schema_overrides=_SCHEMA)

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")
        # Stratified by isFraud to preserve class balance.
        df = df.group_by("isFraud", maintain_order=True).map_groups(
            lambda g: g.sample(fraction=sample_frac, seed=seed)
        )
        log.info("sampled", frac=sample_frac, rows=df.height)

    log.info(
        "loaded",
        rows=df.height,
        cols=df.width,
        fraud_rate=float(df["isFraud"].mean() or 0),  # type: ignore[arg-type]
    )
    return df


def lazy_paysim(path: Path | None = None) -> pl.LazyFrame:
    """Return a lazy Polars frame for memory-conscious feature engineering."""
    csv_path = path or settings.data.paysim_path
    return pl.scan_csv(csv_path, schema_overrides=_SCHEMA)
