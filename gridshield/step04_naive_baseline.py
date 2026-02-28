"""
step04_naive_baseline.py — Naive Baseline Model
================================================
Step 4 of the GRIDSHIELD Forecast Risk Advisory Pipeline.

Naive forecast: predict = load at same 15-min slot exactly 7 days ago (lag_672).
This is the standard industry benchmark for short-term load forecasting.

Outputs:
  - penalty breakdown (total, peak, off-peak, bias, 95th pct deviation, RMSE)
  - plot: actual vs naive forecast for 2 representative weeks
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import compute_penalty, penalty_table_row, set_plot_style, save_plot


def evaluate_naive(val_df: pd.DataFrame) -> dict:
    """
    Evaluate naive (lag_672) forecast on validation set.
    Returns penalty metrics dict.
    """
    actual   = val_df["LOAD"].values
    forecast = val_df["lag_672"].values          # 7-days-ago load
    is_peak  = val_df["is_peak_hour"].values

    row = penalty_table_row("Naive Baseline (lag_672)", actual, forecast, is_peak)

    print(f"\n  ── Naive Baseline Results ──────────────────────────────")
    for k, v in row.items():
        print(f"  {k:<25}: {v:>15,.2f}" if isinstance(v, float) else f"  {k:<25}: {v}")
    print(f"  ─────────────────────────────────────────────────────\n")

    return row, actual, forecast, is_peak


def plot_naive_forecast(val_df: pd.DataFrame):
    """Plot actual vs naive forecast for 2 representative weeks."""
    set_plot_style()

    # Pick the first full week in validation (non-COVID)
    week1_start = pd.Timestamp("2020-09-07")   # normal week
    week1_end   = pd.Timestamp("2020-09-20")   # 2 weeks

    week_df = val_df[(val_df["DateTime"] >= week1_start) &
                     (val_df["DateTime"] <= week1_end)].copy()

    if len(week_df) == 0:
        # Fall back to first available window
        week_df = val_df.iloc[:14*96].copy()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(week_df["DateTime"], week_df["LOAD"],
            label="Actual Load", color="#2E86AB", lw=1.5)
    ax.plot(week_df["DateTime"], week_df["lag_672"],
            label="Naive Forecast (lag_672)", color="#E84855",
            lw=1.5, linestyle="--")

    delta = week_df["LOAD"] - week_df["lag_672"]
    under = delta.clip(lower=0)
    over  = (-delta).clip(lower=0)
    ax.fill_between(week_df["DateTime"], week_df["lag_672"], week_df["LOAD"],
                    where=delta > 0, alpha=0.2, color="#E84855", label="Under-forecast (₹4/kWh)")
    ax.fill_between(week_df["DateTime"], week_df["lag_672"], week_df["LOAD"],
                    where=delta < 0, alpha=0.2, color="#3BB273", label="Over-forecast (₹2/kWh)")

    ax.set_title("Naive Baseline: Actual vs Forecast — 2-Week Sample (Sep 2020)",
                 fontweight="bold")
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Load (kW)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_plot(fig, "09_naive_baseline_forecast.png")


def run(val_df: pd.DataFrame = None):
    print("=" * 60)
    print("  STEP 4 — Naive Baseline Model")
    print("=" * 60)

    if val_df is None:
        features_path = r"c:\hackathons\DecodeX\outputs\features.parquet"
        if os.path.exists(features_path):
            df = pd.read_parquet(features_path)
        else:
            from step02_feature_engineering import build_features
            df = build_features()
        val_df = df[df["DateTime"] >= "2020-01-01"].copy()

    row, actual, forecast, is_peak = evaluate_naive(val_df)
    plot_naive_forecast(val_df)

    print("\n  ✔ STEP 4 COMPLETE — naive baseline evaluated.")
    print("\n  KEY FINDINGS:")
    print("  • A 7-day lag captures weekly cyclicity but misses holiday & COVID anomalies.")
    print("  • Naive model systematically under-forecasts during high-demand days → higher ₹ penalties.")
    print("  • p95 deviation is high — extreme events drive disproportionate financial risk.")

    print("\n  BUSINESS IMPLICATION FOR LUMINA ENERGY:")
    print("  • Naive baseline total penalty forms the benchmark to beat.")
    print("  • Under-forecast risk (₹4/kWh) dominates over-forecast risk (₹2/kWh).")
    print("  • Intelligent model targeting τ=0.667 quantile directly minimises expected penalty.")

    return row, val_df


if __name__ == "__main__":
    run()
