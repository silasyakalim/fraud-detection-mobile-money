"""Download PaySim from Kaggle.

Requires Kaggle credentials, either as ``~/.kaggle/kaggle.json`` or via
``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` in ``.env``. Generate them at
kaggle.com -> Settings -> API -> Create New Token.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from fraud_detection.config import settings
from fraud_detection.logging import get_logger

log = get_logger(__name__)


def download_paysim(*, force: bool = False) -> None:
    """Download and extract PaySim into ``data/raw/``.

    Args:
        force: If True, re-download even if the file already exists.

    Raises:
        RuntimeError: If Kaggle credentials are missing.
    """
    target = settings.data.paysim_path
    if target.exists() and not force:
        log.info("paysim already present, skipping", path=str(target))
        return

    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        kaggle_json = Path("~/.kaggle/kaggle.json").expanduser()
        if not kaggle_json.exists():
            raise RuntimeError(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "in .env, or place kaggle.json at ~/.kaggle/kaggle.json."
            )

    # Imported lazily so module load doesn't fail before credentials are checked.
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    api.authenticate()

    settings.data.raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading from kaggle", dataset=settings.data.paysim_kaggle_dataset)
    api.dataset_download_files(
        settings.data.paysim_kaggle_dataset,
        path=str(settings.data.raw_dir),
        unzip=False,
        quiet=False,
    )

    for zip_path in settings.data.raw_dir.glob("*.zip"):
        log.info("extracting", zip=str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(settings.data.raw_dir)
        zip_path.unlink()

    if not target.exists():
        raise RuntimeError(
            f"Expected {target.name} after extraction but didn't find it. "
            f"Files in raw dir: {list(settings.data.raw_dir.iterdir())}"
        )
    log.info("paysim ready", path=str(target), size_mb=target.stat().st_size / 1e6)


if __name__ == "__main__":
    from fraud_detection.logging import configure_logging

    configure_logging()
    download_paysim()
