# Mobile money fraud detection

Fraud detection on the real [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) mobile money dataset (6.36M transactions, 0.13% fraud rate). Walk-forward CV, cost-sensitive metrics, and a real comparison to the rule-based system shipped with the data.

[![CI](https://github.com/silasyakalim/fraud-detection-mobile-money/actions/workflows/ci.yml/badge.svg)](https://github.com/silasyakalim/fraud-detection-mobile-money/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![Cost curve](docs/images/06_cost_curve.png)

## Headline result

On a held-out test set (the last 30% of `step`, never touched during training), **LightGBM catches 453/500 fraud at 89% precision** — 91% recall against a 0.5% alert budget. Logistic regression catches just 281/500 fraud but every alert is real (100% precision, 56% recall). The legacy `isFlaggedFraud` rule fires once on 56K test transactions: high-confidence but covers almost nothing.

The point of the project is the methodology that produces those numbers, not the numbers themselves:

- **Walk-forward CV on `step`** instead of random splits, so validation never sees fraud from the future.
- **Cost-sensitive metrics** (expected loss, precision@k, cost curve) instead of accuracy or ROC-AUC, both of which lie at this base rate.
- **Cost-aware class weighting** (`scale_pos_weight = n_neg/n_pos`) baked into the LightGBM loss — no SMOTE.
- **Honest baseline.** The legacy rule is in every chart; the ML model has to actually beat it.
- **Test set touched once**, at the very end. No threshold picking, no ablation feedback, no leakage.

The full analysis is in [`notebooks/fraud_detection_analysis.ipynb`](notebooks/fraud_detection_analysis.ipynb) (committed with executed outputs so it renders on GitHub); supporting library code is under [`src/fraud_detection/`](src/fraud_detection/). Methodology notes: [`docs/methodology.md`](docs/methodology.md). Detailed results, ablation, and limitations: [`docs/results.md`](docs/results.md).

## Results

Held-out test set, evaluated once at the end of the notebook:

| Model | Test PR-AUC | Test P@0.5% | Confusion @ 0.5% threshold |
| --- | --- | --- | --- |
| Logistic regression | 0.960 | **1.000** | 281 TP / 219 FN / 0 FP / 55,592 TN |
| LightGBM | **0.884** | 0.943 | 453 TP / 47 FN / 55 FP / 55,537 TN |
| Legacy `isFlaggedFraud` rule | n/a | 1.000 | 1 TP / 499 FN / 0 FP / 55,592 TN |

LightGBM's cost-optimal operating point on the test set is **2.6% alert rate at ~$11,120 expected loss** — 1,459 alerts out of 56,092 transactions, a tractable review queue for a single human reviewer. At the configured 0.5% rate, expected loss is $24,050 for LightGBM vs $109,500 for LR (LR misses far more fraud at the smaller alert budget).

These numbers are on a stratified 20% sample (~1.27M rows) of real PaySim, chosen to keep the ablation phase within memory on a typical 16 GB machine. Set `LOAD_SAMPLE_FRAC = None` in `scripts/build_notebook.py` to re-run end-to-end on the full 6.36M-row file.

## Quickstart

Python 3.12 and [uv](https://docs.astral.sh/uv/) required.

```bash
git clone https://github.com/silasyakalim/fraud-detection-mobile-money.git
cd fraud-detection-mobile-money
make install-dev
```

Get the data. Either real PaySim from Kaggle (needs credentials in `.env`):

```bash
cp .env.example .env       # fill in KAGGLE_USERNAME / KAGGLE_KEY
make download-data         # ~470 MB CSV into data/raw/
```

…or generate a synthetic PaySim-shaped copy with the same schema for offline development:

```bash
make generate-data         # 500K-row synthetic, ~10s
```

Then:

```bash
make data-quality          # 9 checks, ~30s on real PaySim
make train                 # walk-forward CV via the Prefect flow
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

MLflow tracking is optional — start it with `make mlflow-up` and metrics land at `http://localhost:5000`.

## Layout

```
notebooks/   the analysis notebook (committed with outputs)
src/         library code (loaders, splits, features, models, metrics, pipeline)
scripts/     entrypoints (synthetic data, data quality, notebook rebuild, training)
tests/       pytest with synthetic fixtures
docs/        methodology, results, model card, architecture, data-quality report
data/        gitignored — regenerate via scripts
models/      gitignored — MLflow tracks these
```

## Stack

Polars for data, scikit-learn + LightGBM for modeling with cost-aware weighting, Prefect 3 for the training flow, MLflow for tracking, structlog for logs, Pydantic for config, ruff + mypy strict for quality. Built with uv, CI on GitHub Actions.

## Next steps

Honest list — what would actually move the numbers if I picked this back up:

- Run end-to-end on the full 6.36M-row PaySim (currently sampled for memory headroom in the ablation phase) and report whether the gap to the 20% sample is meaningful.
- Optuna hyperparameter sweep for LightGBM under the same walk-forward CV. LightGBM's fold-0 and fold-1 PR-AUC are weak (~0.13) — defaults are probably hurting the early-fold case.
- Isotonic / Platt recalibration so predicted probabilities are usable as probabilities (both models are overconfident — see [results.md](docs/results.md#calibration)).

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

PaySim is a synthetic dataset from Lopez-Rojas et al. ("PaySim: A financial mobile money simulator for fraud detection," EMSS 2016), seeded from a month of real mobile money logs. 6,362,620 transactions over 30 simulated days (`step` = 1..743), fraud (8,213 cases, 0.129%) concentrated in TRANSFER and CASH_OUT. Real mobile money data sits behind regulatory walls, so PaySim is the closest public substitute. See [`docs/model_card.md`](docs/model_card.md) for the full data and intended-use breakdown.

## License

[MIT](LICENSE).
