# Model card — Mobile money fraud detection

Last updated: 2026-05-14

Modeled on Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).

## Model details

- **Model name**: `fraud-detection` (versioned via MLflow Model Registry)
- **Model type**: Tabular binary classifier — gradient-boosted decision trees (LightGBM) and L2 logistic regression baseline
- **Version**: 0.1.0
- **License**: MIT
- **Maintainer**: silasyakalim ([@silasyakalim](https://github.com/silasyakalim))
- **Citation**: See [`CITATION.cff`](../CITATION.cff)

Features are tabular only (balance ratios, drainage signals, temporal flags, account-type flags). Loss is cost-aware via `scale_pos_weight = n_neg/n_pos × cost_FN/cost_FP`; the model is trained on walk-forward expanding-window folds and selected on PR-AUC.

## Intended use

**Primary use case**: Demonstrate a methodology for evaluating fraud detection on highly imbalanced, time-ordered transaction data. The intended audience is technical reviewers (interviewers, hiring managers, collaborators) assessing how the author approaches modeling on real-world-shaped problems.

**Out of scope**: This model is not deployed and not intended to be deployed against real mobile money traffic. Its calibration is poor (see [Results](results.md#calibration)), its cost ratios are placeholders, and it has only been validated on a 500K-row synthetic dataset. Treating its outputs as production fraud scores would be inappropriate.

## Training data

- **Source**: 500K-row synthetic dataset shaped to match PaySim ([Lopez-Rojas et al., EMSS 2016](https://www.kaggle.com/datasets/ealaxi/paysim1)).
- **Generation**: `scripts/generate_synthetic_paysim.py` produces account-drain + cash-out fraud pairs at a 0.13% base rate, with the documented PaySim "destination-balance quirk" preserved.
- **Schema**: 11 columns (`step`, `type`, `amount`, `nameOrig`/`nameDest`, `oldbalanceOrg`/`newbalanceOrig`/`oldbalanceDest`/`newbalanceDest`, `isFraud`, `isFlaggedFraud`).
- **Class balance**: 650 fraud / 499,350 legit (0.13% positive rate).
- **Time ordering**: `step` is hourly, 1–744 covering 30 simulated days.

Real PaySim (6.3M rows) is supported via `make download-data` (Kaggle credentials required); the pipeline is dataset-agnostic.

## Evaluation data

The last 30% of `step` (150,115 rows, 201 fraud cases) is held out and scored exactly once at the end of training. Walk-forward CV inside the first 70% (3 folds, expanding window) drives all model selection.

## Metrics

Headline metrics on the held-out test set (single forward pass, no tuning against test):

| Model | PR-AUC | P@0.5% alert volume | TP | FN | FP |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.978 | 0.266 | 200 | 1 | 551 |
| LightGBM | 0.760 | 0.268 | 201 | 0 | 1,632 |
| Legacy `isFlaggedFraud` rule | n/a | 1.000 | 98 | 103 | 0 |

Per-fold validation numbers and ablation in [`docs/results.md`](results.md). All cost-related metrics use placeholder ratios (`cost_FN=$500, cost_FP=$10`) — a real deployment would calibrate these with finance.

## Quantitative analyses

- **Feature importance**: Three features dominate LightGBM gain: `oldbalanceDest`, `orig_balance_delta`, `newbalanceDest` — all encoding the destination-balance quirk and account-drainage pattern. Dropping `oldbalanceDest` collapses PR-AUC from 0.76 to 0.25.
- **Calibration**: Both models are overconfident; when LR predicts 23% fraud probability the actual rate is ~2%. Predictions should be used for ranking only, not as calibrated probabilities. Recalibration (Platt / isotonic) is on the roadmap.
- **Fraud-type breakdown**: PaySim restricts fraud to `TRANSFER` and `CASH_OUT`. Both models learn this hard cut from the `type` feature.

## Ethical considerations

- **Synthetic data only.** PaySim is itself a generator; this project uses a further-synthesized variant. Findings *will not* transfer linearly to production data — the synthetic generator preserves the destination-balance quirk that makes ML feel easy here, which is partly an artifact of the simulator, not real fraudster behavior.
- **Disparate impact unstudied.** Mobile money users in production span income levels and geographies. PaySim does not encode any demographic information, so fairness analyses are impossible on this data. A real deployment would need stratified evaluation across sender demographics, transaction-size bands, and geography before going live.
- **Investigator capacity is not a model parameter.** Sweeping the operating threshold matters: at the cost-optimum, the model fires ~256 alerts on 150K transactions; at the 0.5% alert rate the budget would commonly impose, it fires ~750. Either way, an alert means a human reviewer. The cost ratios used here (`$500 / $10`) are placeholders, not calibrated values.

## Limitations

- Trained only on a 500K-row synthetic shape; not validated on the full 6.3M-row PaySim.
- No hyperparameter tuning (defaults).
- No graph/network features (in/out degree, PageRank, community labels) — the destination-balance quirk dominates so heavily that tabular features alone hit the ceiling on this synthetic shape.
- No drift monitoring; in production, fraud labels arrive weeks to months late and lag-aware evaluation is required.
- Three-fold CV is too few for tight std estimates at 0.13% positive rate — usable for relative comparisons, not absolute precision.

## Maintenance

This is a portfolio / research project, not a maintained service. Issues and pull requests are welcome on [GitHub](https://github.com/silasyakalim/fraud-detection-mobile-money) but there is no SLA.
