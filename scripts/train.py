#!/usr/bin/env python
"""Run the training pipeline.

Usage:
    uv run python scripts/train.py                    # 10% sample, fast
    uv run python scripts/train.py --full             # full dataset
"""

from __future__ import annotations

import argparse
import sys

from fraud_detection.logging import configure_logging
from fraud_detection.pipelines.train_flow import train_flow


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on the full PaySim dataset instead of a 10%% sample.",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0.1,
        help="Fraction to sample (0-1). Ignored if --full is set.",
    )
    args = parser.parse_args()

    sample_frac = None if args.full else args.sample
    metrics = train_flow(sample_frac=sample_frac)
    print("\nFinal aggregated metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
