"""Command-line entrypoint. Run ``uv run fraud-detect --help`` after install."""

from __future__ import annotations

import typer
from rich.console import Console

from fraud_detection import __version__
from fraud_detection.logging import configure_logging, get_logger

app = typer.Typer(
    name="fraud-detect",
    help="Mobile money fraud detection CLI.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _root(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON logs."),
) -> None:
    configure_logging(log_level, json_logs=json_logs)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"fraud-detection v{__version__}")


@app.command()
def download(
    force: bool = typer.Option(False, "--force", help="Re-download even if files exist."),
) -> None:
    """Download the PaySim dataset from Kaggle."""
    from fraud_detection.data.download import download_paysim

    log = get_logger(__name__)
    log.info("downloading paysim", force=force)
    download_paysim(force=force)
    log.info("download complete")


@app.command()
def train(
    sample: float = typer.Option(
        0.1, "--sample", help="Stratified sample fraction. Use --full for the full dataset."
    ),
    full: bool = typer.Option(False, "--full", help="Train on the full dataset."),
) -> None:
    """Run the Prefect training flow."""
    from fraud_detection.pipelines.train_flow import train_flow

    log = get_logger(__name__)
    sample_frac = None if full else sample
    log.info("starting training", sample_frac=sample_frac)
    metrics = train_flow(sample_frac=sample_frac)
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    app()
