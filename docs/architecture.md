# Architecture

Two layers: data and training. Each is independently testable and can be run on its own.

## Data layer

PaySim lives under `data/raw/` and is gitignored — re-download via `make download-data` (Kaggle) or regenerate a synthetic copy via `make generate-data`.

The loader wraps Polars with an explicit schema. Polars handles the 6.36M-row PaySim file faster than pandas, and its lazy evaluation lets us compose feature pipelines without materializing intermediates. We drop into pandas at the boundaries where downstream libraries (LightGBM, scikit-learn) need it.

The splits module is the part of the data layer that matters most. PaySim is time-ordered via the `step` column (hourly, 1–744 over 30 days). Random splits would let the model see fraud patterns from day 25 while training to predict day 10 — useless in production. The walk-forward splitter yields expanding-window folds where each validation set sits strictly after its training set.

## Training pipeline

The training pipeline is a Prefect 3 flow with independent tasks and retry policies, so a transient Kaggle download failure doesn't blow up the whole run.

**Ingest** — loads raw data, applies stratified sampling if requested, returns a typed Polars frame.

**Featurize** — composes the transforms in `features/tabular.py`. Each transform is pure and idempotent, so they can be reordered or replayed cheaply.

**Train** — for each walk-forward fold, fits a model with cost-aware class weighting. `scale_pos_weight` combines class imbalance correction (`n_neg/n_pos`) with the business cost ratio (`cost_FN/cost_FP`), so the loss function itself is cost-aware. Early stopping uses validation PR-AUC.

**Evaluate** — cost-sensitive metrics across folds: precision at 0.1%, 0.5%, and 1% alert volumes; expected loss at the threshold producing each volume; PR-AUC. Mean and std across folds go to MLflow.

**Register** — the best CV model is logged to the MLflow Model Registry with a `candidate` tag. Promotion to `Production` is a manual step.

## Cross-cutting

**Configuration** is centralized in `config.py` using Pydantic Settings. Every value can be overridden by environment variable or `.env` file. Tests use the same config object — there's no separate test config that can drift.

**Logging** is structured via structlog. In dev, the rich console renderer keeps logs readable. In production, JSON output ships cleanly to log aggregators.

**Testing** uses pytest with synthetic PaySim-shaped data generated in `conftest.py`, so CI can run without Kaggle credentials.

**Quality gates** are ruff (linting + formatting), mypy strict, and pytest with coverage. Pre-commit runs them locally; GitHub Actions runs them on every PR.
