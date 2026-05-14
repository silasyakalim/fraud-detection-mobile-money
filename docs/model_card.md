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

Features are tabular only (balance ratios, drainage signals, temporal flags, account-type flags). Loss is cost-aware via `scale_pos_weight = n_neg/n_pos`; the model is trained on walk-forward expanding-window folds and selected on PR-AUC.

## Intended use

**Primary use case**: Demonstrate a methodology for evaluating fraud detection on highly imbalanced, time-ordered transaction data. The intended audience is technical reviewers (interviewers, hiring managers, collaborators) assessing how the author approaches modeling on real-world-shaped problems.

**Out of scope**: This model is not deployed and not intended to be deployed against real mobile money traffic. Its calibration is poor (see [Results](results.md#calibration)), its cost ratios are placeholders, and it has been validated only on PaySim — a synthetic dataset seeded from a single month of real mobile money logs. Treating its outputs as production fraud scores would be inappropriate.

## Training data

- **Source**: Real PaySim 1.0 ([Lopez-Rojas et al., EMSS 2016](https://www.kaggle.com/datasets/ealaxi/paysim1)) downloaded from Kaggle.
- **Subsample used by the notebook**: stratified 20% by `isFraud`, ~1,272,523 rows (1,642 fraud, 1,270,881 legit). The sample exists for ablation-phase memory headroom; the methodology code itself handles the full 6.36M-row file (`LOAD_SAMPLE_FRAC = None` toggle in `scripts/build_notebook.py`).
- **Schema**: 11 columns — `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
- **Class balance**: 0.129% positive rate (8,213 fraud / 6,362,620 total in the full file).
- **Time ordering**: `step` is hourly, 1–743 covering 30 simulated days.

## Evaluation data

The last 30% of `step` (56,092 rows, 500 fraud cases) is held out and scored exactly once at the end of training. Walk-forward CV inside the first 70% (1,216,431 rows, 1,142 fraud cases; 3 folds, expanding window) drives all model selection.

## Metrics

Headline metrics on the held-out test set (single forward pass, no tuning against test):

| Model | PR-AUC | P@0.5% alert volume | TP | FN | FP |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.960 | **1.000** | 281 | 219 | 0 |
| LightGBM | **0.884** | 0.943 | 453 | 47 | 55 |
| Legacy `isFlaggedFraud` rule | n/a | 1.000 | 1 | 499 | 0 |

LightGBM's cost-optimal alert rate is 2.6%, where expected loss falls to ~$11,120 across 1,459 alerts. Per-fold validation numbers and ablation are in [`docs/results.md`](results.md). All cost-related metrics use placeholder ratios (`cost_FN=$500, cost_FP=$10`) — a real deployment would calibrate these with finance.

## Quantitative analyses

- **Feature importance**: `amount_to_orig_balance` dominates LightGBM gain on real PaySim — dropping it collapses PR-AUC from 0.88 to 0.40. `orig_balance_delta` and `newbalanceOrig` carry most of the rest. The source-side drainage pattern is doing the work; the destination-balance quirk is much less load-bearing on real data than on synthetic PaySim-shaped data.
- **Calibration**: Both models are overconfident; LightGBM more so than LR. Predictions should be used for ranking only, not as calibrated probabilities. Recalibration (Platt / isotonic) is on the roadmap.
- **Fraud-type breakdown**: PaySim restricts fraud to `TRANSFER` and `CASH_OUT`. Both models learn this hard cut from the `type` feature.
- **CV stability**: LightGBM's walk-forward folds vary widely (PR-AUC 0.13 / 0.14 / 0.83 across the three folds). The model is sensitive to the amount of training data in early folds. LR is much more stable (0.91 / 0.81 / 0.95).

## Ethical considerations

- **Synthetic data only.** PaySim is itself a generator seeded from one month of real mobile money logs. Findings *will not* transfer linearly to production data — generator artifacts are visible (the destination-balance quirk, clamped underflows). A real deployment would need re-validation on actual production traffic.
- **Disparate impact unstudied.** Mobile money users in production span income levels and geographies. PaySim does not encode any demographic information, so fairness analyses are impossible on this data. A real deployment would need stratified evaluation across sender demographics, transaction-size bands, and geography before going live.
- **Investigator capacity is not a model parameter.** Sweeping the operating threshold matters: at the cost-optimum, LightGBM fires 1,459 alerts on 56K transactions; at the 0.5% alert budget the model fires 281–508. Either way, an alert means a human reviewer. The cost ratios used here (`$500 / $10`) are placeholders, not calibrated values.

## Limitations

- Trained on a 20% stratified sample of real PaySim; full-data run is on the roadmap (memory-bound by the ablation phase, not the model).
- No hyperparameter tuning (defaults).
- No graph/network features (in/out degree, PageRank) — tabular only.
- No drift monitoring; in production, fraud labels arrive weeks to months late and lag-aware evaluation is required.
- LightGBM CV is unstable (PR-AUC std 0.40). The test PR-AUC of 0.88 is much closer to fold 2 (0.83) than to the CV mean (0.37) — suggests temporal regime shift or insufficient training data in early folds. Optuna tuning or a larger `min_train_window` would tighten this.

## Maintenance

This is a portfolio / research project, not a maintained service. Issues and pull requests are welcome on [GitHub](https://github.com/silasyakalim/fraud-detection-mobile-money) but there is no SLA.
