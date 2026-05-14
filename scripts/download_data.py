#!/usr/bin/env python
"""Download the PaySim dataset.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --force  # re-download
"""

from __future__ import annotations

import sys

from fraud_detection.data.download import download_paysim
from fraud_detection.logging import configure_logging


def main() -> int:
    configure_logging()
    force = "--force" in sys.argv
    try:
        download_paysim(force=force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
