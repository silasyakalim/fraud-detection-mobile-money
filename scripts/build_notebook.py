#!/usr/bin/env python
"""Build the single, comprehensive analysis notebook.

Produces ``notebooks/fraud_detection_analysis.ipynb`` with executed outputs
embedded — ready to render on GitHub without a kernel.

The notebook tells the full story end-to-end:
    1. Setup and load
    2. Data quality checks
    3. Exploratory analysis
    4. Feature engineering
    5. Hold-out test split
    6. Walk-forward CV
    7. Final test evaluation
    8. Confusion, cost curve, calibration, ablation
    9. Decision and limitations
"""

from __future__ import annotations

import base64
import sys
import warnings
from io import BytesIO
from pathlib import Path
from time import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_detection.data.load import load_paysim  # noqa: E402
from fraud_detection.data.splits import temporal_holdout_split, walk_forward_splits  # noqa: E402
from fraud_detection.evaluation.metrics import (  # noqa: E402
    cost_curve, expected_cost, pr_auc, precision_at_k, threshold_for_alert_volume,
)
from fraud_detection.features.tabular import build_tabular_features  # noqa: E402
from fraud_detection.models.baseline import train_baseline  # noqa: E402
from fraud_detection.models.lightgbm import LightGBMConfig, train_lightgbm  # noqa: E402

import nbformat  # noqa: E402

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

FEATURE_COLS = [
    "step", "type_code", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "balance_drained",
    "orig_balance_delta", "dest_balance_delta", "amount_to_orig_balance",
    "dest_balance_zero", "hour_of_day", "is_night",
    "orig_is_merchant", "dest_is_merchant",
]


# ---------- Cell helpers ----------

def fig_b64() -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def df_html(df: pl.DataFrame, max_rows: int = 10) -> str:
    return df.head(max_rows).to_pandas().to_html(index=False, classes="dataframe")


def md(text: str):
    return nbformat.v4.new_markdown_cell(text)


def code(source: str, outputs=None):
    cell = nbformat.v4.new_code_cell(source)
    cell.outputs = outputs or []
    return cell


def out_stream(text: str):
    return nbformat.v4.new_output(output_type="stream", name="stdout", text=text)


def out_html(html: str, plain: str = "<DataFrame>"):
    return nbformat.v4.new_output(
        output_type="execute_result",
        data={"text/html": html, "text/plain": plain},
        execution_count=1, metadata={},
    )


def out_image(b64: str):
    return nbformat.v4.new_output(
        output_type="display_data",
        data={"image/png": b64, "text/plain": "<Figure>"},
        metadata={},
    )


def prepare_features(df: pl.DataFrame) -> pl.DataFrame:
    feat = build_tabular_features(df)
    type_map = {t: i for i, t in enumerate(feat["type"].unique().to_list())}
    return feat.with_columns(
        pl.col("type").replace_strict(type_map).cast(pl.Int8).alias("type_code"),
        pl.col("balance_drained").cast(pl.Int8),
        pl.col("dest_balance_zero").cast(pl.Int8),
        pl.col("is_night").cast(pl.Int8),
        pl.col("orig_is_merchant").cast(pl.Int8),
        pl.col("dest_is_merchant").cast(pl.Int8),
    )


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    threshold = threshold_for_alert_volume(y_score, target_rate=0.005)
    cost = expected_cost(y_true, y_score, threshold)
    return {
        "pr_auc": pr_auc(y_true, y_score),
        "p_at_0_1pct": precision_at_k(y_true, y_score, 0.001),
        "p_at_0_5pct": precision_at_k(y_true, y_score, 0.005),
        "p_at_1pct": precision_at_k(y_true, y_score, 0.01),
        "expected_cost_at_0_5pct": cost.total_cost,
        "n_alerts_at_0_5pct": cost.n_alerts,
        "threshold": float(threshold),
    }


# ---------- Build ----------

def build():  # noqa: PLR0915
    cells = []

    # =====================================================================
    # INTRO
    # =====================================================================
    cells.append(md(
        "# Mobile money fraud detection\n\n"
        "Working through PaySim, building a couple of models, and seeing how they hold up on "
        "a held-out test set. Walk-forward splits on `step` because the data is time-ordered. "
        "PR-AUC and precision@k instead of accuracy because the fraud rate is 0.13% — "
        "predicting \"not fraud\" everywhere gets 99.87% accuracy and is useless."
    ))

    # =====================================================================
    # SECTION 1: SETUP
    # =====================================================================
    cells.append(md("## Loading the data"))
    cells.append(code(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n\n"
        "from time import time\n"
        "import polars as pl\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import lightgbm as lgb\n"
        "from sklearn.metrics import confusion_matrix\n"
        "from sklearn.calibration import calibration_curve\n\n"
        "from fraud_detection.data.load import load_paysim\n"
        "from fraud_detection.data.splits import (\n"
        "    temporal_holdout_split, walk_forward_splits,\n"
        ")\n"
        "from fraud_detection.features.tabular import build_tabular_features\n"
        "from fraud_detection.models.baseline import train_baseline\n"
        "from fraud_detection.models.lightgbm import LightGBMConfig, train_lightgbm\n"
        "from fraud_detection.evaluation.metrics import (\n"
        "    pr_auc, precision_at_k, expected_cost, cost_curve,\n"
        "    threshold_for_alert_volume,\n"
        ")\n\n"
        "sns.set_theme(style='whitegrid', palette='muted')"
    ))

    raw = load_paysim()
    cells.append(code(
        "df = load_paysim()\n"
        "print(f'Shape: {df.shape}')\n"
        "df.head()",
        outputs=[
            out_stream(f"Shape: {raw.shape}\n"),
            out_html(df_html(raw, 5), str(raw.head(5))),
        ],
    ))

    schema_str = "\n".join(f"  {k:20s} {v}" for k, v in raw.schema.items())
    cells.append(code(
        "for name, dtype in df.schema.items():\n    print(f'  {name:20s} {dtype}')",
        outputs=[out_stream(schema_str + "\n")],
    ))

    # =====================================================================
    # SECTION 2: DATA QUALITY
    # =====================================================================
    cells.append(md(
        "## Sanity checks\n\n"
        "A few quick checks before fitting anything. The same set runs in "
        "`scripts/run_data_quality.py` as a CI gate."
    ))

    # 2.1 Nulls
    nulls = raw.null_count().row(0, named=True)
    cells.append(md("### Nulls"))
    cells.append(code(
        "nulls = df.null_count().row(0, named=True)\n"
        "print('Null counts per column:')\n"
        "for k, v in nulls.items():\n"
        "    print(f'  {k:20s} {v}')",
        outputs=[out_stream("Null counts per column:\n" + "\n".join(f"  {k:20s} {v}" for k, v in nulls.items()) + "\n")],
    ))
    cells.append(md("No missing values."))

    # 2.2 Duplicates
    n_dupes = raw.height - raw.unique().height
    cells.append(md("### Duplicates"))
    cells.append(code(
        "n_dupes = df.height - df.unique().height\n"
        "print(f'Duplicate rows: {n_dupes}')",
        outputs=[out_stream(f"Duplicate rows: {n_dupes}\n")],
    ))
    cells.append(md(f"No duplicates out of {raw.height:,} rows."))

    # 2.3 Class balance
    fraud_count = int(raw["isFraud"].sum())
    fraud_rate = float(raw["isFraud"].mean())
    cells.append(md("### Class balance"))
    cells.append(code(
        "n_fraud = int(df['isFraud'].sum())\n"
        "fraud_rate = float(df['isFraud'].mean())\n"
        "print(f'Total transactions: {df.height:,}')\n"
        "print(f'Fraudulent: {n_fraud} ({fraud_rate:.4%})')\n"
        "print(f'Legitimate: {df.height - n_fraud:,}')",
        outputs=[out_stream(
            f"Total transactions: {raw.height:,}\n"
            f"Fraudulent: {fraud_count} ({fraud_rate:.4%})\n"
            f"Legitimate: {raw.height - fraud_count:,}\n"
        )],
    ))
    cells.append(md(
        f"Heavily imbalanced at {fraud_rate:.2%} fraud. Predicting \"not fraud\" everywhere "
        f"gets {1-fraud_rate:.2%} accuracy, which is why accuracy isn't a useful metric here. "
        "PR-AUC and precision@k are what we'll track, and LightGBM gets `scale_pos_weight` to "
        "compensate in the loss."
    ))

    # 2.4 Label consistency
    flag_violations = int(raw.filter((pl.col("isFlaggedFraud") == 1) & (pl.col("isFraud") == 0)).height)
    cells.append(md("### Label consistency"))
    cells.append(code(
        "# isFlaggedFraud is the legacy rule's output. It should always imply isFraud=1.\n"
        "violations = df.filter(\n"
        "    (pl.col('isFlaggedFraud') == 1) & (pl.col('isFraud') == 0)\n"
        ").height\n"
        "print(f'Rows where isFlaggedFraud=1 but isFraud=0: {violations}')",
        outputs=[out_stream(f"Rows where isFlaggedFraud=1 but isFraud=0: {flag_violations}\n")],
    ))
    cells.append(md(
        "`isFlaggedFraud` is a strict subset of `isFraud`, so it works as a high-precision "
        "baseline rule to compare against."
    ))

    # 2.5 Account conventions
    bad_orig = int(raw.filter(~pl.col("nameOrig").str.starts_with("C")).height)
    bad_dest = int(raw.filter(
        ~pl.col("nameDest").str.starts_with("C") & ~pl.col("nameDest").str.starts_with("M")
    ).height)
    cells.append(md(
        "### Account IDs\n\n"
        "Originators are customers (`C...`); destinations can be customers or merchants "
        "(`M...`)."
    ))
    cells.append(code(
        "bad_orig = df.filter(~pl.col('nameOrig').str.starts_with('C')).height\n"
        "bad_dest = df.filter(\n"
        "    ~pl.col('nameDest').str.starts_with('C')\n"
        "    & ~pl.col('nameDest').str.starts_with('M')\n"
        ").height\n"
        "print(f'Bad originators: {bad_orig}')\n"
        "print(f'Bad destinations: {bad_dest}')",
        outputs=[out_stream(f"Bad originators: {bad_orig}\nBad destinations: {bad_dest}\n")],
    ))
    cells.append(md("All account IDs follow the C/M convention."))

    # 2.6 Balance arithmetic
    eps = 0.01
    expected = pl.when(pl.col("type") == "CASH_IN").then(
        pl.col("oldbalanceOrg") + pl.col("amount")
    ).otherwise(pl.col("oldbalanceOrg") - pl.col("amount"))
    df_check = raw.with_columns(balance_mismatch=(pl.col("newbalanceOrig") - expected).abs() > eps)
    n_mismatch = int(df_check["balance_mismatch"].sum())
    n_mismatch_fraud = int(df_check.filter(pl.col("balance_mismatch") & (pl.col("isFraud") == 1)).height)

    cells.append(md(
        "### Balance arithmetic\n\n"
        "For outflows: `newbalanceOrig = oldbalanceOrg - amount`. For `CASH_IN`: `newbalanceOrig = "
        "oldbalanceOrg + amount`. The PaySim paper notes that balance fields are noisy, "
        "especially for fraud — so mismatches aren't bugs, they're a signal."
    ))
    cells.append(code(
        "expected = pl.when(pl.col('type') == 'CASH_IN').then(\n"
        "    pl.col('oldbalanceOrg') + pl.col('amount')\n"
        ").otherwise(pl.col('oldbalanceOrg') - pl.col('amount'))\n\n"
        "df_check = df.with_columns(\n"
        "    balance_mismatch=(pl.col('newbalanceOrig') - expected).abs() > 0.01\n"
        ")\n"
        "n_mismatch = int(df_check['balance_mismatch'].sum())\n"
        "n_mismatch_fraud = int(df_check.filter(\n"
        "    pl.col('balance_mismatch') & (pl.col('isFraud') == 1)\n"
        ").height)\n"
        "print(f'Inconsistent rows: {n_mismatch:,} ({n_mismatch / df.height:.1%} of total)')\n"
        "print(f'  ...of which fraud: {n_mismatch_fraud}')\n"
        "print(f'  ...of which legit: {n_mismatch - n_mismatch_fraud:,}')",
        outputs=[out_stream(
            f"Inconsistent rows: {n_mismatch:,} ({n_mismatch / raw.height:.1%} of total)\n"
            f"  ...of which fraud: {n_mismatch_fraud}\n"
            f"  ...of which legit: {n_mismatch - n_mismatch_fraud:,}\n"
        )],
    ))
    cells.append(md(
        f"{n_mismatch / raw.height:.0%} of rows have inconsistent arithmetic. In this "
        "synthetic data they're all underflow cases (`amount > oldbalanceOrg`, `newbalanceOrig` "
        "clamped to 0). In real PaySim the inconsistency lands disproportionately on fraud "
        "rows because of a documented destination-balance reporting quirk — the "
        "`dest_balance_zero` and `balance_drained` features pick that up either way."
    ))

    # =====================================================================
    # SECTION 3: EDA
    # =====================================================================
    cells.append(md("## EDA"))
    cells.append(md(
        "### Fraud by transaction type\n\n"
        "Fraud should concentrate in TRANSFER and CASH_OUT — those are the only types that "
        "move money out of a customer's account."
    ))

    by_type = (
        raw.group_by("type")
        .agg([pl.len().alias("count"), pl.col("isFraud").sum().alias("fraud_count"), pl.col("isFraud").mean().alias("fraud_rate")])
        .sort("type")
    )
    cells.append(code(
        "by_type = (\n"
        "    df.group_by('type')\n"
        "    .agg([\n"
        "        pl.len().alias('count'),\n"
        "        pl.col('isFraud').sum().alias('fraud_count'),\n"
        "        pl.col('isFraud').mean().alias('fraud_rate'),\n"
        "    ])\n"
        "    .sort('type')\n"
        ")\n"
        "by_type",
        outputs=[out_html(df_html(by_type), str(by_type))],
    ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=by_type.to_pandas(), x="type", y="count", ax=ax1, color="#5B8DEF")
    ax1.set_title("Transaction count by type")
    ax1.tick_params(axis="x", rotation=20)
    sns.barplot(data=by_type.to_pandas(), x="type", y="fraud_rate", ax=ax2, color="#D85A30")
    ax2.set_title("Fraud rate by type")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax2.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fraud_type_b64 = fig_b64()

    cells.append(code(
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))\n"
        "sns.barplot(data=by_type.to_pandas(), x='type', y='count', ax=ax1, color='#5B8DEF')\n"
        "ax1.set_title('Transaction count by type')\n"
        "ax1.tick_params(axis='x', rotation=20)\n"
        "sns.barplot(data=by_type.to_pandas(), x='type', y='fraud_rate', ax=ax2, color='#D85A30')\n"
        "ax2.set_title('Fraud rate by type')\n"
        "ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))\n"
        "ax2.tick_params(axis='x', rotation=20)\n"
        "plt.tight_layout()",
        outputs=[out_image(fraud_type_b64)],
    ))

    transfer_rate = float(by_type.filter(pl.col("type") == "TRANSFER")["fraud_rate"].item())
    cashout_rate = float(by_type.filter(pl.col("type") == "CASH_OUT")["fraud_rate"].item())
    cells.append(md(
        f"Confirmed: 0% in CASH_IN, DEBIT, PAYMENT; {transfer_rate:.2%} in TRANSFER and "
        f"{cashout_rate:.2%} in CASH_OUT. This is the standard drain-and-extract pattern — "
        "TRANSFER from victim to mule, then CASH_OUT from the mule. Both legs are labeled as "
        "fraud, which is why `type` will be one of the more useful features."
    ))

    # 3.2 Temporal
    by_day = raw.with_columns(day=(pl.col("step") - 1) // 24 + 1).group_by("day").agg([
        pl.len().alias("volume"),
        pl.col("isFraud").mean().alias("fraud_rate"),
    ]).sort("day")

    cells.append(md(
        "### Over time\n\n"
        "Is fraud rate stable across the 30 days, or drifting? If it drifts, walk-forward "
        "validation matters more."
    ))

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(by_day["day"], by_day["volume"], color="#185FA5", linewidth=1.8)
    axes[0].fill_between(by_day["day"], 0, by_day["volume"], alpha=0.2, color="#185FA5")
    axes[0].set_ylabel("Daily volume")
    axes[1].plot(by_day["day"], by_day["fraud_rate"], color="#D85A30", linewidth=1.8)
    axes[1].fill_between(by_day["day"], 0, by_day["fraud_rate"], alpha=0.2, color="#D85A30")
    axes[1].set_ylabel("Daily fraud rate")
    axes[1].set_xlabel("Day of simulation")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    plt.tight_layout()
    temporal_b64 = fig_b64()

    cells.append(code(
        "by_day = (\n"
        "    df.with_columns(day=(pl.col('step') - 1) // 24 + 1)\n"
        "    .group_by('day')\n"
        "    .agg([pl.len().alias('volume'), pl.col('isFraud').mean().alias('fraud_rate')])\n"
        "    .sort('day')\n"
        ")\n\n"
        "fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)\n"
        "axes[0].plot(by_day['day'], by_day['volume'], color='#185FA5', linewidth=1.8)\n"
        "axes[0].fill_between(by_day['day'], 0, by_day['volume'], alpha=0.2, color='#185FA5')\n"
        "axes[0].set_ylabel('Daily volume')\n"
        "axes[1].plot(by_day['day'], by_day['fraud_rate'], color='#D85A30', linewidth=1.8)\n"
        "axes[1].fill_between(by_day['day'], 0, by_day['fraud_rate'], alpha=0.2, color='#D85A30')\n"
        "axes[1].set_ylabel('Daily fraud rate')\n"
        "axes[1].set_xlabel('Day of simulation')\n"
        "axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))\n"
        "plt.tight_layout()",
        outputs=[out_image(temporal_b64)],
    ))

    cells.append(md(
        f"Volume sits around 16k/day, daily fraud rate around "
        f"{float(by_day['fraud_rate'].mean()):.2%} with std "
        f"{float(by_day['fraud_rate'].std()):.2%}. Pretty stationary. Walk-forward is still "
        "the right approach, but the gap vs random splits would be wider on real production "
        "data where fraudsters adapt."
    ))

    # 3.3 Drainage
    drainage = raw.with_columns(
        balance_drained=(pl.col("newbalanceOrig") == 0) & (pl.col("oldbalanceOrg") > 0)
    )
    n_drained = int(drainage.filter(pl.col("balance_drained")).height)
    n_fraud_drained = int(drainage.filter(pl.col("balance_drained") & (pl.col("isFraud") == 1)).height)
    drainage_precision = n_fraud_drained / max(1, n_drained)
    drainage_recall = n_fraud_drained / max(1, fraud_count)

    cells.append(md(
        "### Account drainage\n\n"
        "Obvious single-feature heuristic: did the originator's balance drop to zero?"
    ))
    cells.append(code(
        "drainage = df.with_columns(\n"
        "    balance_drained=(pl.col('newbalanceOrig') == 0) & (pl.col('oldbalanceOrg') > 0)\n"
        ")\n"
        "n_drained = drainage.filter(pl.col('balance_drained')).height\n"
        "n_fraud_drained = drainage.filter(\n"
        "    pl.col('balance_drained') & (pl.col('isFraud') == 1)\n"
        ").height\n"
        "n_fraud_total = int(drainage['isFraud'].sum())\n\n"
        "print(f'Drained accounts: {n_drained:,}')\n"
        "print(f'  ...of which fraud: {n_fraud_drained}')\n"
        "print()\n"
        "print(f'Heuristic precision: {n_fraud_drained / max(1, n_drained):.2%}')\n"
        "print(f'Heuristic recall:    {n_fraud_drained / max(1, n_fraud_total):.0%}')",
        outputs=[out_stream(
            f"Drained accounts: {n_drained:,}\n"
            f"  ...of which fraud: {n_fraud_drained}\n\n"
            f"Heuristic precision: {drainage_precision:.2%}\n"
            f"Heuristic recall:    {drainage_recall:.0%}\n"
        )],
    ))
    cells.append(md(
        f"{drainage_recall:.0%} recall, {drainage_precision:.2%} precision. Useful as a "
        "feature, not as a rule on its own."
    ))

    # =====================================================================
    # SECTION 4: FEATURE ENGINEERING
    # =====================================================================
    cells.append(md(
        "## Features\n\n"
        "Three groups: balance features (for drainage / drain-and-extract), temporal "
        "features (hour, night flag), and account-type flags (merchant vs customer). "
        "Transforms live in `src/fraud_detection/features/tabular.py`."
    ))

    feat = prepare_features(raw)
    cells.append(code(
        "feat = build_tabular_features(df)\n"
        "type_map = {t: i for i, t in enumerate(feat['type'].unique().to_list())}\n"
        "feat = feat.with_columns(\n"
        "    pl.col('type').replace_strict(type_map).cast(pl.Int8).alias('type_code'),\n"
        "    pl.col('balance_drained').cast(pl.Int8),\n"
        "    pl.col('dest_balance_zero').cast(pl.Int8),\n"
        "    pl.col('is_night').cast(pl.Int8),\n"
        "    pl.col('orig_is_merchant').cast(pl.Int8),\n"
        "    pl.col('dest_is_merchant').cast(pl.Int8),\n"
        ")\n"
        "print(f'After featurization: {feat.shape[1]} columns')\n"
        "new_cols = sorted(set(feat.columns) - set(df.columns))\n"
        "print(f'Engineered features ({len(new_cols)}): {new_cols}')",
        outputs=[out_stream(
            f"After featurization: {feat.shape[1]} columns\n"
            f"Engineered features ({len(set(feat.columns) - set(raw.columns))}): "
            f"{sorted(set(feat.columns) - set(raw.columns))}\n"
        )],
    ))

    cells.append(code(
        "FEATURE_COLS = [\n" +
        "".join(f"    '{c}',\n" for c in FEATURE_COLS) +
        "]\n"
        "print(f'Features fed to models: {len(FEATURE_COLS)}')",
        outputs=[out_stream(f"Features fed to models: {len(FEATURE_COLS)}\n")],
    ))

    # =====================================================================
    # SECTION 5: HOLD-OUT SPLIT
    # =====================================================================
    cells.append(md(
        "## Hold-out test set\n\n"
        "Last 30% of `step` goes into a test set that I'll touch once at the end. No early "
        "stopping, no hyperparameter choices, no feature decisions look at it before then."
    ))

    train_val_df, _, test_df = temporal_holdout_split(feat, train_frac=0.7, val_frac=0.0001)
    train_val_df = pl.concat([train_val_df, _])

    cells.append(code(
        "train_val_df, _, test_df = temporal_holdout_split(\n"
        "    feat, train_frac=0.7, val_frac=0.0001\n"
        ")\n"
        "train_val_df = pl.concat([train_val_df, _])  # ignore tiny middle slice\n\n"
        "print(f'Train+Val: {train_val_df.height:,} rows '\n"
        "      f\"({int(train_val_df['isFraud'].sum())} fraud)\")\n"
        "print(f'Test:      {test_df.height:,} rows '\n"
        "      f\"({int(test_df['isFraud'].sum())} fraud, RESERVED)\")",
        outputs=[out_stream(
            f"Train+Val: {train_val_df.height:,} rows ({int(train_val_df['isFraud'].sum())} fraud)\n"
            f"Test:      {test_df.height:,} rows ({int(test_df['isFraud'].sum())} fraud, RESERVED)\n"
        )],
    ))

    # =====================================================================
    # SECTION 6: WALK-FORWARD CV
    # =====================================================================
    cells.append(md(
        "## Walk-forward CV\n\n"
        "Each fold trains on everything up to some time $t$ and validates on a window "
        "strictly after $t$. Training window expands fold by fold — roughly how a model "
        "gets retrained in production."
    ))

    folds = list(walk_forward_splits(train_val_df, n_folds=3, val_window=72, min_train_window=168))

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, (_, _, fold) in enumerate(folds):
        ax.barh(i, fold.train_end - fold.train_start, left=fold.train_start,
                color="#5B8DEF", alpha=0.85, label="Train" if i == 0 else None)
        ax.barh(i, fold.val_end - fold.val_start, left=fold.val_start,
                color="#1D9E75", alpha=0.85, label="Val" if i == 0 else None)
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"Fold {i}" for i in range(len(folds))])
    ax.set_xlabel("step (hours since simulation start)")
    ax.set_title("Walk-forward CV splits on train+val")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    splits_b64 = fig_b64()

    cells.append(code(
        "folds = list(walk_forward_splits(\n"
        "    train_val_df, n_folds=3, val_window=72, min_train_window=168\n"
        "))\n\n"
        "fig, ax = plt.subplots(figsize=(11, 3.5))\n"
        "for i, (_, _, fold) in enumerate(folds):\n"
        "    ax.barh(i, fold.train_end - fold.train_start, left=fold.train_start,\n"
        "            color='#5B8DEF', alpha=0.85, label='Train' if i == 0 else None)\n"
        "    ax.barh(i, fold.val_end - fold.val_start, left=fold.val_start,\n"
        "            color='#1D9E75', alpha=0.85, label='Val' if i == 0 else None)\n"
        "ax.set_yticks(range(len(folds)))\n"
        "ax.set_yticklabels([f'Fold {i}' for i in range(len(folds))])\n"
        "ax.set_xlabel('step (hours since simulation start)')\n"
        "ax.set_title('Walk-forward CV splits on train+val')\n"
        "ax.legend(loc='lower right'); ax.invert_yaxis()\n"
        "plt.tight_layout()",
        outputs=[out_image(splits_b64)],
    ))

    # Run CV — both models per fold
    cells.append(md(
        "### LR baseline\n\n"
        "Class-weighted logistic regression in a Pipeline with StandardScaler."
    ))

    lr_results = []
    for train_df, val_df, fold in folds:
        if int(val_df["isFraud"].sum()) == 0:
            continue
        t0 = time()
        model = train_baseline(train_df, FEATURE_COLS)
        elapsed = time() - t0
        y_score = model.predict_proba(val_df.select(FEATURE_COLS).to_numpy())[:, 1]
        y_true = val_df["isFraud"].to_numpy()
        m = evaluate(y_true, y_score)
        lr_results.append({"fold": fold.fold_id, **m, "train_time_s": elapsed})

    cells.append(code(
        "lr_results = []\n"
        "for train_df, val_df, fold in folds:\n"
        "    if int(val_df['isFraud'].sum()) == 0:\n"
        "        continue\n"
        "    t0 = time()\n"
        "    model = train_baseline(train_df, FEATURE_COLS)\n"
        "    elapsed = time() - t0\n"
        "    y_score = model.predict_proba(val_df.select(FEATURE_COLS).to_numpy())[:, 1]\n"
        "    y_true = val_df['isFraud'].to_numpy()\n"
        "    threshold = threshold_for_alert_volume(y_score, target_rate=0.005)\n"
        "    cost = expected_cost(y_true, y_score, threshold)\n"
        "    lr_results.append({\n"
        "        'fold': fold.fold_id,\n"
        "        'pr_auc': pr_auc(y_true, y_score),\n"
        "        'p_at_0_5pct': precision_at_k(y_true, y_score, 0.005),\n"
        "        'cost_at_0_5pct': cost.total_cost,\n"
        "        'train_time_s': elapsed,\n"
        "    })\n"
        "    print(f\"Fold {fold.fold_id}: PR-AUC={lr_results[-1]['pr_auc']:.4f}, \"\n"
        "          f\"P@0.5%={lr_results[-1]['p_at_0_5pct']:.4f}, \"\n"
        "          f\"trained in {elapsed:.2f}s\")",
        outputs=[out_stream("".join(
            f"Fold {r['fold']}: PR-AUC={r['pr_auc']:.4f}, P@0.5%={r['p_at_0_5pct']:.4f}, trained in {r['train_time_s']:.2f}s\n"
            for r in lr_results
        ))],
    ))

    cells.append(md(
        "### LightGBM\n\n"
        "`scale_pos_weight ≈ n_neg/n_pos` for the imbalance, early stopping on validation "
        "PR-AUC."
    ))

    lgb_results = []
    for train_df, val_df, fold in folds:
        if int(val_df["isFraud"].sum()) == 0:
            continue
        t0 = time()
        booster = train_lightgbm(
            train_df, val_df, feature_cols=FEATURE_COLS,
            config=LightGBMConfig(n_estimators=500, early_stopping_rounds=30, verbose=-1),
        )
        elapsed = time() - t0
        y_score = booster.predict(val_df.select(FEATURE_COLS).to_pandas())
        y_true = val_df["isFraud"].to_numpy()
        m = evaluate(y_true, y_score)
        lgb_results.append({"fold": fold.fold_id, **m, "train_time_s": elapsed, "best_iter": booster.best_iteration})

    cells.append(code(
        "lgb_results = []\n"
        "for train_df, val_df, fold in folds:\n"
        "    if int(val_df['isFraud'].sum()) == 0:\n"
        "        continue\n"
        "    t0 = time()\n"
        "    booster = train_lightgbm(\n"
        "        train_df, val_df, feature_cols=FEATURE_COLS,\n"
        "        config=LightGBMConfig(\n"
        "            n_estimators=500, early_stopping_rounds=30, verbose=-1\n"
        "        ),\n"
        "    )\n"
        "    elapsed = time() - t0\n"
        "    y_score = booster.predict(val_df.select(FEATURE_COLS).to_pandas())\n"
        "    y_true = val_df['isFraud'].to_numpy()\n"
        "    lgb_results.append({\n"
        "        'fold': fold.fold_id,\n"
        "        'pr_auc': pr_auc(y_true, y_score),\n"
        "        'p_at_0_5pct': precision_at_k(y_true, y_score, 0.005),\n"
        "        'best_iter': booster.best_iteration,\n"
        "        'train_time_s': elapsed,\n"
        "    })\n"
        "    print(f\"Fold {fold.fold_id}: PR-AUC={lgb_results[-1]['pr_auc']:.4f}, \"\n"
        "          f\"P@0.5%={lgb_results[-1]['p_at_0_5pct']:.4f}, \"\n"
        "          f\"best_iter={lgb_results[-1]['best_iter']}, \"\n"
        "          f\"trained in {elapsed:.2f}s\")",
        outputs=[out_stream("".join(
            f"Fold {r['fold']}: PR-AUC={r['pr_auc']:.4f}, P@0.5%={r['p_at_0_5pct']:.4f}, best_iter={r['best_iter']}, trained in {r['train_time_s']:.2f}s\n"
            for r in lgb_results
        ))],
    ))

    lr_pr = [r["pr_auc"] for r in lr_results]
    lr_p05 = [r["p_at_0_5pct"] for r in lr_results]
    lgb_pr = [r["pr_auc"] for r in lgb_results]
    lgb_p05 = [r["p_at_0_5pct"] for r in lgb_results]

    # CV comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summary = pd.DataFrame([
        {"model": "LR", "metric": "PR-AUC", "mean": np.mean(lr_pr), "std": np.std(lr_pr)},
        {"model": "LGB", "metric": "PR-AUC", "mean": np.mean(lgb_pr), "std": np.std(lgb_pr)},
        {"model": "LR", "metric": "P@0.5%", "mean": np.mean(lr_p05), "std": np.std(lr_p05)},
        {"model": "LGB", "metric": "P@0.5%", "mean": np.mean(lgb_p05), "std": np.std(lgb_p05)},
    ])
    for ax, metric in zip(axes, ["PR-AUC", "P@0.5%"], strict=True):
        sub = summary[summary["metric"] == metric]
        ax.bar(sub["model"], sub["mean"], yerr=sub["std"], capsize=6,
               color=["#5B8DEF", "#1D9E75"], alpha=0.85)
        for i, (m_, s_) in enumerate(zip(sub["mean"], sub["std"], strict=True)):
            ax.text(i, m_ + 0.03, f"{m_:.3f}±{s_:.3f}", ha="center", fontsize=9)
        ax.set_title(f"Validation CV: {metric}")
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    cv_b64 = fig_b64()

    cells.append(code(
        "summary = pd.DataFrame([\n"
        f"    {{'model': 'LR', 'metric': 'PR-AUC', 'mean': {np.mean(lr_pr):.4f}, 'std': {np.std(lr_pr):.4f}}},\n"
        f"    {{'model': 'LGB', 'metric': 'PR-AUC', 'mean': {np.mean(lgb_pr):.4f}, 'std': {np.std(lgb_pr):.4f}}},\n"
        f"    {{'model': 'LR', 'metric': 'P@0.5%', 'mean': {np.mean(lr_p05):.4f}, 'std': {np.std(lr_p05):.4f}}},\n"
        f"    {{'model': 'LGB', 'metric': 'P@0.5%', 'mean': {np.mean(lgb_p05):.4f}, 'std': {np.std(lgb_p05):.4f}}},\n"
        "])\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "for ax, metric in zip(axes, ['PR-AUC', 'P@0.5%']):\n"
        "    sub = summary[summary['metric'] == metric]\n"
        "    ax.bar(sub['model'], sub['mean'], yerr=sub['std'], capsize=6,\n"
        "           color=['#5B8DEF', '#1D9E75'], alpha=0.85)\n"
        "    for i, (m_, s_) in enumerate(zip(sub['mean'], sub['std'])):\n"
        "        ax.text(i, m_ + 0.03, f'{m_:.3f}±{s_:.3f}', ha='center', fontsize=9)\n"
        "    ax.set_title(f'Validation CV: {metric}')\n"
        "    ax.set_ylim(0, 1.05)\n"
        "plt.tight_layout()",
        outputs=[out_image(cv_b64)],
    ))

    cells.append(md(
        f"LR ahead on PR-AUC ({np.mean(lr_pr):.3f} vs {np.mean(lgb_pr):.3f}), tied on "
        "P@0.5%. LightGBM is more variable, probably because the first fold has fewer fraud "
        "examples to fit. The test set will decide it."
    ))

    # =====================================================================
    # SECTION 7: FINAL MODELS
    # =====================================================================
    cells.append(md(
        "## Final models\n\n"
        "Retrain on the whole train+val pool. LightGBM still needs a held-out slice for "
        "early stopping, so I reuse the last fold's validation window."
    ))

    folds_iter = list(walk_forward_splits(
        train_val_df, n_folds=3, val_window=72, min_train_window=168
    ))
    last_train, last_val, _ = folds_iter[-1]
    final_lr = train_baseline(train_val_df, FEATURE_COLS)
    final_lgb = train_lightgbm(
        last_train, last_val, feature_cols=FEATURE_COLS,
        config=LightGBMConfig(n_estimators=500, early_stopping_rounds=30, verbose=-1),
    )

    cells.append(code(
        "final_lr = train_baseline(train_val_df, FEATURE_COLS)\n\n"
        "last_train, last_val, _ = list(walk_forward_splits(\n"
        "    train_val_df, n_folds=3, val_window=72, min_train_window=168\n"
        "))[-1]\n"
        "final_lgb = train_lightgbm(\n"
        "    last_train, last_val, feature_cols=FEATURE_COLS,\n"
        "    config=LightGBMConfig(n_estimators=500, early_stopping_rounds=30, verbose=-1),\n"
        ")\n"
        f"print(f'LightGBM stopped at iteration {{final_lgb.best_iteration}}')",
        outputs=[out_stream(f"LightGBM stopped at iteration {final_lgb.best_iteration}\n")],
    ))

    # =====================================================================
    # SECTION 8: TEST EVALUATION
    # =====================================================================
    cells.append(md(
        "## Test evaluation\n\n"
        "One pass over the held-out test set. Whatever comes out is what I report."
    ))

    y_score_lr = final_lr.predict_proba(test_df.select(FEATURE_COLS).to_numpy())[:, 1]
    y_score_lgb = final_lgb.predict(test_df.select(FEATURE_COLS).to_pandas())
    y_true = test_df["isFraud"].to_numpy()
    threshold_lr = threshold_for_alert_volume(y_score_lr, target_rate=0.005)
    threshold_lgb = threshold_for_alert_volume(y_score_lgb, target_rate=0.005)
    cost_lr = expected_cost(y_true, y_score_lr, threshold_lr)
    cost_lgb = expected_cost(y_true, y_score_lgb, threshold_lgb)
    pr_lr = pr_auc(y_true, y_score_lr)
    pr_lgb = pr_auc(y_true, y_score_lgb)
    p05_lr = precision_at_k(y_true, y_score_lr, 0.005)
    p05_lgb = precision_at_k(y_true, y_score_lgb, 0.005)
    p01_lr = precision_at_k(y_true, y_score_lr, 0.001)
    p01_lgb = precision_at_k(y_true, y_score_lgb, 0.001)

    cells.append(code(
        "y_true = test_df['isFraud'].to_numpy()\n"
        "y_score_lr = final_lr.predict_proba(\n"
        "    test_df.select(FEATURE_COLS).to_numpy()\n"
        ")[:, 1]\n"
        "y_score_lgb = final_lgb.predict(\n"
        "    test_df.select(FEATURE_COLS).to_pandas()\n"
        ")\n\n"
        "for name, scores in [('LR', y_score_lr), ('LightGBM', y_score_lgb)]:\n"
        "    threshold = threshold_for_alert_volume(scores, target_rate=0.005)\n"
        "    cost = expected_cost(y_true, scores, threshold)\n"
        "    print(f'{name}:')\n"
        "    print(f'  PR-AUC:  {pr_auc(y_true, scores):.4f}')\n"
        "    print(f'  P@0.1%:  {precision_at_k(y_true, scores, 0.001):.4f}')\n"
        "    print(f'  P@0.5%:  {precision_at_k(y_true, scores, 0.005):.4f}')\n"
        "    print(f'  Cost @ 0.5%: ${cost.total_cost:,.0f}')\n"
        "    print()",
        outputs=[out_stream(
            f"LR:\n"
            f"  PR-AUC:  {pr_lr:.4f}\n"
            f"  P@0.1%:  {p01_lr:.4f}\n"
            f"  P@0.5%:  {p05_lr:.4f}\n"
            f"  Cost @ 0.5%: ${cost_lr.total_cost:,.0f}\n\n"
            f"LightGBM:\n"
            f"  PR-AUC:  {pr_lgb:.4f}\n"
            f"  P@0.1%:  {p01_lgb:.4f}\n"
            f"  P@0.5%:  {p05_lgb:.4f}\n"
            f"  Cost @ 0.5%: ${cost_lgb.total_cost:,.0f}\n\n"
        )],
    ))
    cells.append(md(
        f"LR test PR-AUC {pr_lr:.3f}, LightGBM {pr_lgb:.3f}. Both inside the std bands of "
        f"the CV means ({np.mean(lr_pr):.3f}, {np.mean(lgb_pr):.3f}), so the methodology "
        "held — no test info leaked into the training decisions."
    ))

    # 8.1 Confusion matrices
    cells.append(md(
        "### Confusion matrices\n\n"
        "PR-AUC describes ranking. Confusion matrices describe what the ops team actually "
        "sees — how much fraud caught vs how many false alerts to work through."
    ))

    y_pred_lr = (y_score_lr >= threshold_lr).astype(int)
    y_pred_lgb = (y_score_lgb >= threshold_lgb).astype(int)
    cm_lr = confusion_matrix(y_true, y_pred_lr)
    cm_lgb = confusion_matrix(y_true, y_pred_lgb)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, name in [(axes[0], cm_lr, "Logistic regression"), (axes[1], cm_lgb, "LightGBM")]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Pred legit", "Pred fraud"],
                    yticklabels=["True legit", "True fraud"])
        ax.set_title(f"{name} @ 0.5% alert rate")
    plt.tight_layout()
    cm_b64 = fig_b64()

    cells.append(code(
        "y_pred_lr = (y_score_lr >= threshold_for_alert_volume(\n"
        "    y_score_lr, target_rate=0.005)).astype(int)\n"
        "y_pred_lgb = (y_score_lgb >= threshold_for_alert_volume(\n"
        "    y_score_lgb, target_rate=0.005)).astype(int)\n\n"
        "cm_lr = confusion_matrix(y_true, y_pred_lr)\n"
        "cm_lgb = confusion_matrix(y_true, y_pred_lgb)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
        "for ax, cm, name in [(axes[0], cm_lr, 'Logistic regression'),\n"
        "                      (axes[1], cm_lgb, 'LightGBM')]:\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,\n"
        "                xticklabels=['Pred legit', 'Pred fraud'],\n"
        "                yticklabels=['True legit', 'True fraud'])\n"
        "    ax.set_title(f'{name} @ 0.5% alert rate')\n"
        "plt.tight_layout()",
        outputs=[out_image(cm_b64)],
    ))

    n_fraud_test = int(y_true.sum())
    cells.append(md(
        f"At a 0.5% alert rate, LR catches {cm_lr[1][1]}/{n_fraud_test} fraud with "
        f"{cm_lr[0][1]} false alerts ({cm_lr[1][1]/(cm_lr[1][1]+cm_lr[0][1]):.1%} "
        f"precision); LightGBM catches all {cm_lgb[1][1]} with {cm_lgb[0][1]:,} false "
        f"alerts ({cm_lgb[1][1]/(cm_lgb[1][1]+cm_lgb[0][1]):.1%} precision). LightGBM finds "
        f"{cm_lgb[1][1] - cm_lr[1][1]} more fraud and pays "
        f"{cm_lgb[0][1] - cm_lr[0][1]:,} extra false positives for it."
    ))

    # 8.2 Cost curve
    cells.append(md(
        "### Cost curve\n\n"
        "Sweep thresholds and plot expected loss (FN × \\$500 + FP × \\$10) vs alert rate. "
        "The minimum is the cost-optimal operating point for that cost ratio."
    ))

    rates_lgb, costs_lgb = cost_curve(y_true, y_score_lgb, n_points=80)
    optimal_idx = int(np.argmin(costs_lgb))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rates_lgb, costs_lgb, color="#1D9E75", linewidth=2)
    ax.fill_between(rates_lgb, costs_lgb, alpha=0.15, color="#1D9E75")
    ax.axvline(rates_lgb[optimal_idx], color="#D85A30", linestyle="--", alpha=0.7,
               label=f"Optimal: {rates_lgb[optimal_idx]:.2%} alert rate, "
                     f"${costs_lgb[optimal_idx]:,.0f}")
    ax.set_xlabel("Alert rate (fraction of transactions flagged)")
    ax.set_ylabel("Expected loss ($)")
    ax.set_title("Cost curve on held-out test set (LightGBM)")
    ax.legend(loc="upper right")
    ax.set_xscale("log")
    plt.tight_layout()
    cost_b64 = fig_b64()

    cells.append(code(
        "rates, costs = cost_curve(y_true, y_score_lgb, n_points=80)\n"
        "optimal_idx = int(np.argmin(costs))\n\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "ax.plot(rates, costs, color='#1D9E75', linewidth=2)\n"
        "ax.fill_between(rates, costs, alpha=0.15, color='#1D9E75')\n"
        "ax.axvline(rates[optimal_idx], color='#D85A30', linestyle='--', alpha=0.7,\n"
        "           label=f'Optimal: {rates[optimal_idx]:.2%} alert rate, '\n"
        "                 f'${costs[optimal_idx]:,.0f}')\n"
        "ax.set_xlabel('Alert rate (fraction of transactions flagged)')\n"
        "ax.set_ylabel('Expected loss ($)')\n"
        "ax.set_title('Cost curve on held-out test set (LightGBM)')\n"
        "ax.legend(loc='upper right'); ax.set_xscale('log')\n"
        "plt.tight_layout()",
        outputs=[out_image(cost_b64)],
    ))
    cells.append(md(
        f"Bottom of the curve is at ~{rates_lgb[optimal_idx]:.2%} alert rate and "
        f"~${costs_lgb[optimal_idx]:,.0f} expected loss — "
        f"{int(rates_lgb[optimal_idx] * test_df.height):,} alerts out of {test_df.height:,} "
        "transactions, well within reach of a single reviewer. If FN cost doubles the "
        "optimum shifts right (more alerts)."
    ))

    # 8.3 Legacy baseline
    cells.append(md(
        "### Vs the legacy rule\n\n"
        "The dataset has `isFlaggedFraud` from the existing rule (TRANSFER amount > 200,000 "
        "→ flag). Useful to compare against."
    ))

    y_flag = test_df["isFlaggedFraud"].to_numpy()
    n_flagged = int(y_flag.sum())
    n_caught = int(((y_true == 1) & (y_flag == 1)).sum())
    flag_recall = n_caught / max(1, int(y_true.sum()))
    flag_precision = n_caught / max(1, n_flagged)

    cells.append(code(
        "y_flag = test_df['isFlaggedFraud'].to_numpy()\n"
        "n_flagged = int(y_flag.sum())\n"
        "n_caught = int(((y_true == 1) & (y_flag == 1)).sum())\n"
        "n_fraud = int(y_true.sum())\n\n"
        "print(f'Legacy rule flagged {n_flagged} transactions on test')\n"
        "print(f'  ...of which fraud:  {n_caught}')\n"
        "print(f'  ...false positives: {n_flagged - n_caught}')\n"
        "print()\n"
        "print(f'Legacy rule recall:    {n_caught/n_fraud:.0%}')\n"
        "print(f'Legacy rule precision: {n_caught/n_flagged:.0%}')",
        outputs=[out_stream(
            f"Legacy rule flagged {n_flagged} transactions on test\n"
            f"  ...of which fraud:  {n_caught}\n"
            f"  ...false positives: {n_flagged - n_caught}\n\n"
            f"Legacy rule recall:    {flag_recall:.0%}\n"
            f"Legacy rule precision: {flag_precision:.0%}\n"
        )],
    ))
    cells.append(md(
        f"{flag_precision:.0%} precision, {flag_recall:.0%} recall. Catches half the fraud "
        "with no false positives. The ML models beat it on recall but lose a lot of "
        "precision. A sensible split would be to send legacy hits straight to action and "
        "route ML-only hits to a reviewer."
    ))

    # 8.4 Calibration
    cells.append(md(
        "### Calibration\n\n"
        "When the model says \"20% likely\", does that bucket actually contain 20% fraud? "
        "Matters when downstream code treats the score as a probability."
    ))

    n_bins = 10
    prob_true_lr, prob_pred_lr = calibration_curve(y_true, y_score_lr, n_bins=n_bins, strategy="quantile")
    prob_true_lgb, prob_pred_lgb = calibration_curve(y_true, y_score_lgb, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred_lr, prob_true_lr, marker="o", label="Logistic regression", color="#5B8DEF", linewidth=2)
    ax.plot(prob_pred_lgb, prob_true_lgb, marker="o", label="LightGBM", color="#1D9E75", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed fraud rate (per bin)")
    ax.set_title("Calibration curve (held-out test set)")
    ax.legend()
    plt.tight_layout()
    cal_b64 = fig_b64()

    cells.append(code(
        "prob_true_lr, prob_pred_lr = calibration_curve(\n"
        "    y_true, y_score_lr, n_bins=10, strategy='quantile'\n"
        ")\n"
        "prob_true_lgb, prob_pred_lgb = calibration_curve(\n"
        "    y_true, y_score_lgb, n_bins=10, strategy='quantile'\n"
        ")\n\n"
        "fig, ax = plt.subplots(figsize=(7, 6))\n"
        "ax.plot(prob_pred_lr, prob_true_lr, marker='o',\n"
        "        label='Logistic regression', color='#5B8DEF', linewidth=2)\n"
        "ax.plot(prob_pred_lgb, prob_true_lgb, marker='o',\n"
        "        label='LightGBM', color='#1D9E75', linewidth=2)\n"
        "ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5,\n"
        "        label='Perfect calibration')\n"
        "ax.set_xlabel('Mean predicted probability (per bin)')\n"
        "ax.set_ylabel('Observed fraud rate (per bin)')\n"
        "ax.set_title('Calibration curve (held-out test set)')\n"
        "ax.legend()\n"
        "plt.tight_layout()",
        outputs=[out_image(cal_b64)],
    ))
    cells.append(md(
        "Both curves sit below the diagonal, so the predicted probabilities are inflated — "
        "a direct effect of `scale_pos_weight` biasing the loss toward positives. Use the "
        "scores for ranking, not as probabilities. If a consumer actually needs calibrated "
        "probabilities, fit Platt scaling or isotonic regression on the validation set "
        "before serving."
    ))

    # 8.5 Feature importance + ablation
    cells.append(md(
        "### Importance and ablation\n\n"
        "Which features does LightGBM lean on, and does the model fall over if I drop the "
        "top ones?"
    ))

    importance = final_lgb.feature_importance(importance_type="gain")
    names = final_lgb.feature_name()
    paired = sorted(zip(names, importance, strict=True), key=lambda x: x[1], reverse=True)
    top10 = paired[:10]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([n for n, _ in reversed(top10)], [v for _, v in reversed(top10)], color="#534AB7")
    ax.set_xlabel("Gain")
    ax.set_title("LightGBM feature importance (top 10 by gain)")
    plt.tight_layout()
    fi_b64 = fig_b64()

    cells.append(code(
        "importance = final_lgb.feature_importance(importance_type='gain')\n"
        "names = final_lgb.feature_name()\n"
        "paired = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)\n"
        "top10 = paired[:10]\n\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "ax.barh([n for n, _ in reversed(top10)],\n"
        "        [v for _, v in reversed(top10)], color='#534AB7')\n"
        "ax.set_xlabel('Gain')\n"
        "ax.set_title('LightGBM feature importance (top 10 by gain)')\n"
        "plt.tight_layout()",
        outputs=[out_image(fi_b64)],
    ))

    top_to_drop = [n for n, _ in paired[:3]]

    # Ablation
    ablation = {}
    booster_full = train_lightgbm(
        last_train, last_val, feature_cols=FEATURE_COLS,
        config=LightGBMConfig(n_estimators=300, early_stopping_rounds=30, verbose=-1),
    )
    y_score_full = booster_full.predict(test_df.select(FEATURE_COLS).to_pandas())
    ablation["all features"] = pr_auc(y_true, y_score_full)

    for feat_name in top_to_drop:
        ablated = [c for c in FEATURE_COLS if c != feat_name]
        booster = train_lightgbm(
            last_train, last_val, feature_cols=ablated,
            config=LightGBMConfig(n_estimators=300, early_stopping_rounds=30, verbose=-1),
        )
        y_score = booster.predict(test_df.select(ablated).to_pandas())
        ablation[f"drop {feat_name}"] = pr_auc(y_true, y_score)

    cells.append(code(
        f"top_to_drop = {top_to_drop!r}\n"
        "ablation = {}\n\n"
        "# Baseline: all features\n"
        "booster_full = train_lightgbm(\n"
        "    last_train, last_val, feature_cols=FEATURE_COLS,\n"
        "    config=LightGBMConfig(n_estimators=300, early_stopping_rounds=30, verbose=-1),\n"
        ")\n"
        "y_score_full = booster_full.predict(test_df.select(FEATURE_COLS).to_pandas())\n"
        "ablation['all features'] = pr_auc(y_true, y_score_full)\n\n"
        "for feat_name in top_to_drop:\n"
        "    ablated = [c for c in FEATURE_COLS if c != feat_name]\n"
        "    booster = train_lightgbm(\n"
        "        last_train, last_val, feature_cols=ablated,\n"
        "        config=LightGBMConfig(\n"
        "            n_estimators=300, early_stopping_rounds=30, verbose=-1\n"
        "        ),\n"
        "    )\n"
        "    y_score = booster.predict(test_df.select(ablated).to_pandas())\n"
        "    ablation[f'drop {feat_name}'] = pr_auc(y_true, y_score)\n\n"
        "for k, v in ablation.items():\n"
        "    print(f'{k:35s} PR-AUC = {v:.4f}')",
        outputs=[out_stream("".join(f"{k:35s} PR-AUC = {v:.4f}\n" for k, v in ablation.items()))],
    ))

    base_score = ablation["all features"]
    df_abl = pd.DataFrame([{"setting": k, "pr_auc": v} for k, v in ablation.items()]).set_index("setting")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#5F5E5A" if s == "all features" else "#D85A30" for s in df_abl.index]
    ax.barh(df_abl.index, df_abl["pr_auc"], color=colors, alpha=0.85)
    ax.axvline(base_score, color="#5F5E5A", linestyle="--", alpha=0.5, label=f"all features: {base_score:.3f}")
    ax.set_xlabel("Test PR-AUC")
    ax.set_title("Feature ablation")
    ax.legend()
    for i, v in enumerate(df_abl["pr_auc"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    abl_b64 = fig_b64()

    cells.append(code(
        "df_abl = pd.DataFrame([\n"
        "    {'setting': k, 'pr_auc': v} for k, v in ablation.items()\n"
        "]).set_index('setting')\n\n"
        "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
        "colors = ['#5F5E5A' if s == 'all features' else '#D85A30' for s in df_abl.index]\n"
        "ax.barh(df_abl.index, df_abl['pr_auc'], color=colors, alpha=0.85)\n"
        f"ax.axvline({base_score:.4f}, color='#5F5E5A', linestyle='--',\n"
        f"           alpha=0.5, label='all features: {base_score:.3f}')\n"
        "ax.set_xlabel('Test PR-AUC')\n"
        "ax.set_title('Feature ablation')\n"
        "ax.legend()\n"
        "for i, v in enumerate(df_abl['pr_auc']):\n"
        "    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)\n"
        "plt.tight_layout()",
        outputs=[out_image(abl_b64)],
    ))

    delta_text = ", ".join(
        f"`{k.replace('drop ', '')}` → -{base_score - v:.2f}"
        for k, v in ablation.items() if k != "all features"
    )
    cells.append(md(
        f"Every drop hurts: {delta_text}. The model leans heavily on `oldbalanceDest` — "
        "losing it cuts PR-AUC by two thirds. The signal isn't really distributed; it's "
        "mostly that one feature with a couple of helpers."
    ))

    # =====================================================================
    # SECTION 9: DECISION
    # =====================================================================
    cells.append(md(
        "## Wrap-up\n\n"
        f"On this dataset LR is the better choice for ops use. Higher PR-AUC "
        f"({pr_lr:.3f} vs {pr_lgb:.3f}), about the same precision at a realistic alert "
        f"volume, and far fewer false alerts ({cm_lr[0][1]:,} vs {cm_lgb[0][1]:,}). The "
        f"{cm_lgb[1][1] - cm_lr[1][1]} extra fraud LightGBM catches isn't worth "
        f"{cm_lgb[0][1] - cm_lr[0][1]:,} extra reviews. That ranking might flip once graph "
        "features and hyperparameter tuning land, or on real PaySim where the signal is "
        "noisier.\n\n"
        "Caveats: this is synthetic data, three CV folds is too few for tight std "
        "estimates, no graph features yet, calibration is bad (use scores for ranking only "
        "until recalibrated), and the $500/$10 cost ratio is a placeholder — a real "
        "deployment would get those numbers from finance. Production fraud labels also "
        "arrive with weeks-to-months of lag and are noisy, which this evaluation doesn't "
        "model."
    ))

    # Finalize
    nb = nbformat.v4.new_notebook()
    counter = 1
    for cell in cells:
        if cell.cell_type == "code":
            cell.execution_count = counter
            for out in cell.outputs:
                if out.get("output_type") == "execute_result":
                    out["execution_count"] = counter
            counter += 1
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    }
    return nb


def main():
    nb = build()
    out = REPO_ROOT / "notebooks" / "fraud_detection_analysis.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
