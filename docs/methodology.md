# Methodology

Why each modeling choice is what it is.

## Walk-forward validation

The default in a lot of notebooks is `train_test_split(shuffle=True)`. For PaySim — and any time-ordered data — that's wrong in a way that doesn't show up in the metric.

Random splits let the model train on hour 600 and validate on hour 100. In production, the model is always trained on the past and applied to the future. If fraud patterns evolve at all (and they do, because fraudsters adapt), a model evaluated on shuffled data overstates how well it'll do once it's deployed.

Walk-forward validation simulates production retraining. Each fold trains on data up to time *t* and validates on a window strictly after *t*. The training window expands fold by fold so later folds see more data, mirroring how a deployed model accumulates examples.

The cost is more pessimistic numbers. A walk-forward PR-AUC of 0.85 reflects what the model would do in production. A random-split PR-AUC of 0.99 doesn't reflect anything useful.

## Cost-sensitive evaluation

At a 0.13% positive rate, accuracy is meaningless — predicting "not fraud" everywhere gets 99.87%. ROC-AUC is better but still hides the question that actually matters.

What fraud ops cares about:

- Total $ recovered from caught fraud, minus
- $ cost of investigator time on false alerts.

That's expected loss at a threshold:

```
expected_loss(threshold) = FN(threshold) × cost_FN + FP(threshold) × cost_FP
```

The `cost_FN` and `cost_FP` defaults in `config.py` ($500 and $10) are placeholders — in a real deployment they'd come from finance. The framework accepts overrides per evaluation.

The chart that goes with this is the cost curve: expected loss on the y-axis, alert volume on the x-axis, swept across thresholds. The optimal point is where the curve bottoms out, but ops teams often have a hard cap on alert volume (an investigator can only review N alerts per day), so the chart shows the cost at any feasible volume.

Primary metrics:

- PR-AUC — better than ROC-AUC for rare events because both axes care about positives.
- Precision@k at k = 0.1%, 0.5%, 1% — matches realistic alert volumes.
- Expected loss at a target alert rate — the bottom line.

## Cost-aware class weighting (no SMOTE)

We deliberately don't use SMOTE or other resampling. Two reasons:

1. Research consistently shows oversampling underperforms proper class weighting on highly imbalanced fraud detection.
2. SMOTE applied before splitting leaks synthetic positives across folds. SMOTE per-fold is fine but adds complexity for no gain.

Instead, set `scale_pos_weight` in LightGBM to:

```
scale_pos_weight = (n_negative / n_positive) × (cost_FN / cost_FP)
```

The first term corrects for class imbalance. The second injects business cost asymmetry directly into the loss. The model treats missing a fraud as `cost_FN/cost_FP` times worse than flagging a legit transaction, baked into the gradients.

## Comparison to the legacy rule

PaySim ships an existing `isFlaggedFraud` column — the simple rule the original system used (TRANSFER amount > 200,000 → flag). We score it alongside the ML models because a portfolio number with no reference point is meaningless. The legacy rule has perfect precision and ~50% recall; the ML models trade precision for recall. Whether that trade is good depends on investigator capacity, which is a finance decision, not a modeling one. The cost curve makes the trade-off explicit.
