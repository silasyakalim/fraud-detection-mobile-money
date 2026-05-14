# Results

> Status: numbers below are from a 500K-row synthetic PaySim-shaped sample (0.13% fraud rate). Re-running on the real ~6.3M-row file is on the roadmap. The methodology and pipeline structure are the same either way.

## Methodology summary

Three things are enforced in code:

**Temporal hold-out** — the last 30% of `step` (150K rows, 201 fraud cases) is held out as a test set and touched exactly once at the end. The first 70% (350K rows, 449 fraud cases) is the train+val pool.

**Walk-forward validation within train+val** — three folds with expanding training windows; each validation window sits strictly after its training window in time. No information from the future leaks into model selection. CV mean ± std is what we report as the *expected* generalization gap; the held-out test is the *measured* number.

**Test set is touched once** at the end. No early stopping, no hyperparameter selection, no feature ablation uses it for training decisions. Every chart labeled "test" is a single forward pass over data the model never saw.

## Cost curve

The single most decision-relevant chart for a deployed fraud system: sweep the alert threshold, plot the fraction of transactions flagged on the x-axis, plot total expected loss on the y-axis. The minimum is the cost-optimal operating point.

![Cost curve](images/06_cost_curve.png)

LightGBM's optimum on the held-out test: alert about 0.17% of transactions for ~$3,600 expected loss — that's 256 alerts out of 150,115 transactions, small enough for a single reviewer. At the configured 0.5% alert rate, expected loss is higher (~$16K) but still manageable.

## Model performance

### Validation vs test

![Validation vs test](images/05_model_comparison.png)

| Model | Val PR-AUC (CV mean) | Test PR-AUC | Val P@0.5% | Test P@0.5% |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.945 ± 0.029 | **0.978** | 0.257 ± 0.021 | 0.266 |
| LightGBM | 0.734 ± 0.041 | **0.760** | 0.258 ± 0.031 | 0.268 |

Test numbers are slightly *better* than the validation CV means for both models — typical for expanding-window CV, where later folds get more training data. Validation std (~0.03) is a usable estimate of run-to-run uncertainty.

### Confusion matrices on the test set

Both models at the 0.5% alert threshold:

![Confusion matrices](images/08_confusion_matrix.png)

- LR catches 200/201 fraud (99.5% recall) with 551 false alerts. Fraud ops would review ~750 cases, ~27% of which are real fraud.
- LightGBM catches 201/201 (100% recall) with 1,632 false alerts — same recall, ~3× the alert volume.
- The marginal fraud LightGBM catches that LR misses is one transaction. The cost is ~1,000 extra false positives. For this dataset and these cost ratios, LR is the better operational choice.

### Comparison to the legacy rule

The dataset includes `isFlaggedFraud`, the existing rule-based system's output (TRANSFER amount > 200,000 → flag).

| System | Recall | Precision |
| --- | --- | --- |
| `isFlaggedFraud` rule | 49% | **100%** |
| Logistic regression @ 0.5% | **99.5%** | 27% |
| LightGBM @ 0.5% | **100%** | 11% |

The legacy rule has perfect precision because it only fires on a narrow, unambiguous case (very large transfers) — but it misses half of all fraud. ML systems trade precision for recall: catching nearly all fraud at the cost of more reviews. Whether that's the right trade depends on investigator capacity and the cost ratio.

## Feature importance and ablation

LightGBM's top features by gain on the final test:

![Feature importance](images/07_feature_importance.png)

Three features dominate, all encoding the destination-balance quirk and the source-account drainage pattern: `oldbalanceDest`, `newbalanceDest`, and `orig_balance_delta`.

Ablation says these aren't spurious:

![Ablation](images/10_ablation.png)

| Setting | Test PR-AUC | Δ vs all features |
| --- | --- | --- |
| All features | 0.760 | — |
| Drop `oldbalanceDest` | 0.253 | -0.508 |
| Drop `orig_balance_delta` | 0.405 | -0.355 |
| Drop `newbalanceDest` | 0.633 | -0.128 |

`oldbalanceDest` is the most load-bearing feature — dropping it cuts PR-AUC by two thirds. That's consistent with the documented PaySim quirk where destination balance fields stay near zero for fraud, making it a strong binary signal. Every drop hurts; nothing improves the model when removed, which is what we'd want to see.

## Calibration

A side-finding worth surfacing:

![Calibration](images/09_calibration.png)

Both models are overconfident. When LR predicts 23% fraud probability, the actual rate is ~2%. When LightGBM predicts 17%, actual is ~11%. This is a known consequence of using `scale_pos_weight` for class imbalance — the loss is biased toward predicting positives, so output probabilities end up inflated.

So: don't use these probabilities as probabilities. Use them for ranking only (which is what precision@k and the threshold metrics already do). For deployments that need calibrated probabilities, fit Platt scaling or isotonic regression on the validation set before serving.

## Data quality

A separate report runs in `scripts/run_data_quality.py`.

| Check | Status |
| --- | --- |
| Schema and dtypes match expected | Pass |
| No missing values in any column | Pass |
| No duplicate rows | Pass |
| All 5 transaction types present | Pass |
| Account ID prefix conventions (C/M) | Pass |
| `isFlaggedFraud` is a strict subset of `isFraud` | Pass |
| No negative amounts or balances | Pass |
| Balance arithmetic | 28% of rows inconsistent — informational |

The 28% balance inconsistency is concentrated in legitimate "underflow" transactions (where `amount > oldbalanceOrg` and `newbalanceOrig` got clamped to 0). In real PaySim, this inconsistency concentrates in *fraud* rows because of the destination-balance quirk — that's an artifact of how this synthetic data was generated. Either way, the modeling code captures both patterns via `dest_balance_zero` and `balance_drained`. See [data_quality.md](data_quality.md) for the full breakdown.

## Limitations

- **Synthetic data.** Real PaySim has noisier balances and adversarial drift. Test PR-AUC on this synthetic version is likely an upper bound vs the real dataset.
- **Three folds is too few** for tight std estimates at 0.13% fraud rate. The roadmap moves to 5 folds with Optuna tuning.
- **No graph features yet.** Tabular only. Adding network features (week 2) is expected to close or invert the LR/LightGBM gap.
- **No hyperparameter tuning.** Both models use defaults. Optuna sweeps are week 2.
- **Cost ratios are placeholders.** $500 per missed fraud, $10 per investigator review — calibrated from finance in a real deployment.
- **Labels are clean.** In production, fraud labels arrive with weeks-to-months of lag and are noisy. A real system would need lag-aware evaluation.

## Reproducibility

```bash
python scripts/generate_synthetic_paysim.py --rows 500000 --seed 42
python scripts/run_data_quality.py        # 9 checks, ~5s
python scripts/build_notebook.py          # rebuilds the notebook with executed outputs
```

Per-fold raw numbers in `docs/training_results.json`. Data quality findings in `docs/data_quality.json`. Charts in `docs/images/`.

## Roadmap

| Metric | Current (week 1) | Target (week 4) |
| --- | --- | --- |
| Test PR-AUC (best model) | 0.978 | > 0.98 with graph + tuning |
| Test P@0.5% | 0.27 | > 0.40 with full feature set |
| Cross-dataset PR-AUC on MoMTSim | n/a | reported honestly |
| Calibration | Poor (overconfident) | Recalibrated via isotonic regression |
| Hyperparameter tuning | None (defaults) | Optuna with walk-forward CV objective |
