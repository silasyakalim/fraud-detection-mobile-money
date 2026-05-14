# Mobile money fraud detection

Fraud detection on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) mobile money dataset (0.13% fraud rate). Walk-forward CV, cost-sensitive metrics, and a real comparison to the rule-based system shipped with the data.

[![CI](https://github.com/silasyakalim/fraud-detection-mobile-money/actions/workflows/ci.yml/badge.svg)](https://github.com/silasyakalim/fraud-detection-mobile-money/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![Cost curve](docs/images/06_cost_curve.png)

## Headline result

On a held-out test set (the last 30% of `step`, never touched during training), logistic regression catches **200/201 fraud cases at 27% precision** — 99.5% recall against a 0.5% alert budget. The legacy `isFlaggedFraud` rule catches half that fraud with perfect precision; LightGBM catches one more case than LR but triples the false-positive volume.

The point of the project is the methodology around that number, not the number itself:

- **Walk-forward CV on `step`** instead of random splits, so validation never sees fraud from the future.
- **Cost-sensitive metrics** (expected loss, precision@k, cost curve) instead of accuracy or ROC-AUC, both of which lie at this base rate.
- **Cost-aware class weighting** (`scale_pos_weight = n_neg/n_pos × cost_FN/cost_FP`) baked into the LightGBM loss — no SMOTE.
- **Honest baseline.** The legacy rule is in every chart; the ML model has to actually beat it.
- **Test set touched once**, at the very end. No threshold picking, no ablation feedback, no leakage.

The full analysis is in [`notebooks/fraud_detection_analysis.ipynb`](notebooks/fraud_detection_analysis.ipynb) (committed with executed outputs so it renders on GitHub); supporting library code is under [`src/fraud_detection/`](src/fraud_detection/). Methodology notes: [`docs/methodology.md`](docs/methodology.md). Detailed results, ablation, and limitations: [`docs/results.md`](docs/results.md).

## Results

| Model | Test PR-AUC | Test P@0.5% | Confusion @ 0.5% threshold |
| --- | --- | --- | --- |
| Logistic regression | **0.978** | 0.266 | 200 TP / 1 FN / 551 FP / 149,363 TN |
| LightGBM | 0.760 | 0.268 | 201 TP / 0 FN / 1,632 FP / 148,282 TN |
| Legacy `isFlaggedFraud` rule | n/a | 1.000 | 98 TP / 103 FN / 0 FP / 149,914 TN |

These numbers are on a 500K-row synthetic version of PaySim. The same pipeline runs on the real 6.3M-row file — re-running it there is on the roadmap.

## Quickstart

Python 3.12 and [uv](https://docs.astral.sh/uv/) required.

```bash
git clone https://github.com/silasyakalim/fraud-detection-mobile-money.git
cd fraud-detection-mobile-money
make install-dev
make generate-data         # 500K-row synthetic PaySim, ~10s
make data-quality          # 9 checks, ~5s
make train                 # walk-forward CV, ~2min
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

For the real PaySim file (470 MB, requires Kaggle credentials in `.env`):

```bash
cp .env.example .env       # fill in KAGGLE_USERNAME / KAGGLE_KEY
make download-data
uv run python scripts/train.py --full
```

MLflow tracking is optional — start it with `make mlflow-up` and metrics land at `http://localhost:5000`.

## Layout

```
notebooks/   the analysis notebook (committed with outputs)
src/         library code (loaders, splits, features, models, metrics, pipeline)
scripts/     entrypoints (synthetic data, data quality, training)
tests/       pytest with synthetic fixtures
docs/        methodology, results, model card, architecture
data/        gitignored — regenerate via scripts
models/      gitignored — MLflow tracks these
```

## Stack

Polars for data, scikit-learn + LightGBM for modeling with cost-aware weighting, Prefect 3 for the training flow, MLflow for tracking, structlog for logs, Pydantic for config, ruff + mypy strict for quality. Built with uv, CI on GitHub Actions.

## Next steps

Honest list — what would actually move the numbers if I picked this back up:

- Optuna hyperparameter sweep for LightGBM under the same walk-forward CV.
- Isotonic / Platt recalibration so predicted probabilities are usable as probabilities (both models are currently overconfident — see [results.md](docs/results.md#calibration)).
- Run the full 6.3M-row PaySim end-to-end and report the gap vs. the 500K synthetic.

## Development

```bash
make help          # list tasks
make lint
make format
make type
make test
make cov
```

Pre-commit runs ruff, mypy, and basic file checks. CI runs the same on every PR.

## About the data

PaySim is a synthetic dataset from Lopez-Rojas et al. ("PaySim: A financial mobile money simulator for fraud detection," EMSS 2016), seeded from a month of real mobile money logs. ~6.3M transactions over 30 simulated days, fraud concentrated in TRANSFER and CASH_OUT. Real mobile money data sits behind regulatory walls, so PaySim is the closest public substitute. See [`docs/model_card.md`](docs/model_card.md) for the full data and intended-use breakdown.

## License

[MIT](LICENSE).
