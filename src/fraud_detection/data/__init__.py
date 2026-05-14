"""Data acquisition, loading, and splitting."""

from __future__ import annotations

from fraud_detection.data.load import load_paysim
from fraud_detection.data.splits import walk_forward_splits

__all__ = ["load_paysim", "walk_forward_splits"]
