"""
step03_train_val_split.py — Time-Aware Train / Validation Split
================================================================
Step 3 of the GRIDSHIELD Forecast Risk Advisory Pipeline.

Split strategy:
  - Training  : April 2013 – September 2019  (model fitting)
  - Dev       : October 2019 – December 2019  (early-stopping only — never reported)
  - Validation: January 2020 – April 2021     (reporting, includes COVID period)

NO random shuffling. Strict temporal ordering preserved.

Why a separate dev set?
  Early stopping in LightGBM uses the eval_set to decide when to stop adding trees.
  If the same set is used for stopping AND reporting, performance is slightly inflated
  (the model tuned its stopping to that set). Using a separate dev set ensures the
  reported validation metrics are on data the model truly never influenced.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import set_plot_style, save_plot
from step02_feature_engineering import ALL_FEATURES

# ──────────────────────────────────────────────────────────────────────────────
# Three-way split boundaries
TRAIN_END = "2019-09-30 23:59:59"    # Training   : Apr 2013 → Sep 2019
DEV_START = "2019-10-01 00:00:00"    # Dev        : Oct 2019 → Dec 2019 (early-stopping only)
DEV_END   = "2019-12-31 23:59:59"    # Dev end
VAL_START = "2020-01-01 00:00:00"    # Validation : Jan 2020 → Apr 2021 (reporting)
TARGET    = "LOAD"


def split_features(df: pd.DataFrame):
    """
    [LEGACY] Two-way split: Train (Apr 2013–Sep 2019) + Validation (Jan 2020–Apr 2021).
    Kept for backward compatibility with steps that don't do early stopping.
    For model training with clean early stopping, use split_with_dev() instead.
    """
    train_df = df[df["DateTime"] <= TRAIN_END].copy()
    val_df   = df[df["DateTime"] >= VAL_START].copy()

    assert len(train_df) > 0, "Training set is empty!"
    assert len(val_df)   > 0, "Validation set is empty!"

    feats = [f for f in ALL_FEATURES if f in df.columns]

    X_train = train_df[feats].values
    y_train = train_df[TARGET].values
    X_val   = val_df[feats].values
    y_val   = val_df[TARGET].values

    print(f"\n  Train : {train_df['DateTime'].min().date()} → {train_df['DateTime'].max().date()} "
          f"| {len(train_df):,} rows")
    print(f"  Val   : {val_df['DateTime'].min().date()} → {val_df['DateTime'].max().date()} "
          f"| {len(val_df):,} rows")
    print(f"  Features used: {len(feats)}")

    return X_train, y_train, X_val, y_val, val_df, feats


def split_with_dev(df: pd.DataFrame):
    """
    Three-way temporal split with a dedicated early-stopping dev set.

        Training  : Apr 2013 – Sep 2019  (model fitting)
        Dev       : Oct 2019 – Dec 2019  (early stopping — NEVER reported)
        Validation: Jan 2020 – Apr 2021  (reporting — NEVER touched during training)

    Returns
    -------
    X_train, y_train  : training arrays
    X_dev,   y_dev    : dev arrays for LightGBM early stopping
    X_val,   y_val    : validation arrays for unbiased reporting
    val_df            : full validation DataFrame (with all columns)
    feats             : list of feature column names
    """
    train_df = df[df["DateTime"] <= TRAIN_END].copy()
    dev_df   = df[(df["DateTime"] >= DEV_START) & (df["DateTime"] <= DEV_END)].copy()
    val_df   = df[df["DateTime"] >= VAL_START].copy()

    # Sanity checks
    assert len(train_df) > 0, "Training set is empty!"
    assert len(dev_df)   > 0, "Dev set is empty!"
    assert len(val_df)   > 0, "Validation set is empty!"
    assert train_df["DateTime"].max() < pd.Timestamp(DEV_START), \
        "DATA LEAKAGE: training overlaps dev set!"
    assert dev_df["DateTime"].max() < pd.Timestamp(VAL_START), \
        "DATA LEAKAGE: dev set overlaps validation set!"

    feats = [f for f in ALL_FEATURES if f in df.columns]

    X_train = train_df[feats].values;  y_train = train_df[TARGET].values
    X_dev   = dev_df[feats].values;    y_dev   = dev_df[TARGET].values
    X_val   = val_df[feats].values;    y_val   = val_df[TARGET].values

    print(f"\n  Train : {train_df['DateTime'].min().date()} → {train_df['DateTime'].max().date()} "
          f"| {len(train_df):,} rows")
    print(f"  Dev   : {dev_df['DateTime'].min().date()} → {dev_df['DateTime'].max().date()} "
          f"| {len(dev_df):,} rows  [early-stopping only — not reported]")
    print(f"  Val   : {val_df['DateTime'].min().date()} → {val_df['DateTime'].max().date()} "
          f"| {len(val_df):,} rows")
    print(f"  Features used: {len(feats)}")

    return X_train, y_train, X_dev, y_dev, X_val, y_val, val_df, feats


def plot_split(df: pd.DataFrame):
    """Visual confirmation of 3-way train / dev / val split."""
    set_plot_style()

    daily      = df.set_index("DateTime")["LOAD"].resample("D").mean()
    train_mask = daily.index <= TRAIN_END
    dev_mask   = (daily.index > TRAIN_END) & (daily.index < pd.Timestamp(VAL_START))
    val_mask   = daily.index >= pd.Timestamp(VAL_START)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(daily.index[train_mask], daily.values[train_mask],
            lw=0.8, color="#2E86AB", label="Training (Apr 2013–Sep 2019)")
    ax.plot(daily.index[dev_mask],   daily.values[dev_mask],
            lw=0.8, color="#F4A261", label="Dev / Early-stop (Oct–Dec 2019)")
    ax.plot(daily.index[val_mask],   daily.values[val_mask],
            lw=0.8, color="#E84855", label="Validation (Jan 2020–Apr 2021, incl. COVID)")
    ax.axvline(pd.Timestamp(DEV_START), color="#F4A261", lw=1.5, linestyle="--",
               label="Train→Dev boundary")
    ax.axvline(pd.Timestamp(VAL_START), color="black",   lw=2,   linestyle="--",
               label="Dev→Val boundary (report line)")
    ax.set_title("Time-Aware 3-Way Split: Train / Dev / Validation", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Average Load (kW)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_plot(fig, "08_train_val_split.png")


def run(df: pd.DataFrame = None):
    print("=" * 60)
    print("  STEP 3 — Train / Validation Split")
    print("=" * 60)

    if df is None:
        features_path = r"c:\hackathons\DecodeX\outputs\features.parquet"
        if os.path.exists(features_path):
            df = pd.read_parquet(features_path)
        else:
            from step02_feature_engineering import build_features
            df = build_features()

    X_train, y_train, X_val, y_val, val_df, feats = split_features(df)
    plot_split(df)

    print("\n  ✔ STEP 3 COMPLETE — strict temporal split verified, no data leakage.")
    print("\n  KEY FINDINGS:")
    print("  • Validation set covers the COVID disruption period — a key stress test.")
    print("  • ~80/20 train/val split by time ensures no future information contaminates training.")

    print("\n  BUSINESS IMPLICATION FOR LUMINA ENERGY:")
    print("  • Validation performance on 2020–2021 (abnormal period) provides a worst-case")
    print("    estimate of penalty — conservative risk management.")

    return X_train, y_train, X_val, y_val, val_df, feats, df


if __name__ == "__main__":
    run()
