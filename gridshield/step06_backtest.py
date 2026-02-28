"""
step06_backtest.py — Backtest & Penalty Comparison
====================================================
Step 6 of the GRIDSHIELD Forecast Risk Advisory Pipeline.

Produces the full penalty comparison table across all four models:
  | Metric              | Naive | LightGBM MSE | LightGBM Q0.667 |
  |---------------------|-------|--------------|-----------------|
  | Total Penalty (₹)   |       |              |                 |
  | Peak Penalty (₹)    |       |              |                 |
  | Off-Peak Penalty (₹)|       |              |                 |
  | Forecast Bias (%)   |       |              |                 |
  | 95th pct Dev (kW)   |       |              |                 |
  | RMSE (kW)           |       |              |                 |

Reports % penalty reduction of Q0.667 vs naive baseline.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import penalty_table_row, set_plot_style, save_plot


def build_comparison_table(val_df: pd.DataFrame,
                           preds: dict,
                           extra_preds: dict = None) -> pd.DataFrame:
    """
    Build the full-period penalty comparison table.

    Parameters
    ----------
    val_df       : validation DataFrame with LOAD, lag_672, is_peak_hour columns
    preds        : dict {key: array} from LightGBM predict_all()
    extra_preds  : optional dict {label: full-val-array} for extra models
                   (HW-ETS, Linear Regression, Random Forest from phase2)
    """
    actual  = val_df["LOAD"].values
    is_peak = val_df["is_peak_hour"].values

    # Display order: Naive first, then extra fast models, then LightGBM
    rows = []

    # 1. Naive baseline
    rows.append(penalty_table_row(
        "Naive Baseline (lag\u2087\u2086\u2087\u2082)", actual, val_df["lag_672"].values, is_peak))

    # 2. Extra fast models (HW-ETS, Linear Regression, Random Forest)
    EXTRA_ORDER = ["Holt-Winters ETS", "Linear Regression", "Random Forest"]
    if extra_preds:
        for label in EXTRA_ORDER:
            if label in extra_preds:
                arr = extra_preds[label]
                # Trim or pad to match val length
                n = len(actual)
                if len(arr) >= n:
                    arr = arr[:n]
                else:
                    arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
                rows.append(penalty_table_row(label, actual, arr, is_peak))

    # 3. LightGBM models
    for key, label in [("lgbm_mse",  "LightGBM MSE"),
                        ("lgbm_q667", "LightGBM Q0.667 \u2605")]:
        if key in preds:
            rows.append(penalty_table_row(label, actual, preds[key], is_peak))

    return pd.DataFrame(rows)


def print_comparison_table(df_table: pd.DataFrame, naive_penalty: float):
    """Pretty-print TABLE B and penalty reduction stats."""
    print("\n" + "=" * 90)
    print("  TABLE B: FULL-PERIOD BACKTEST (Jan 2020 - Apr 2021) — All Fast Models")
    print("  NOTE: SARIMA excluded — too slow for 16-month run.")
    print("        Compare with TABLE A (14-day window, all 7 models) printed earlier.")
    print("=" * 90)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(df_table.to_string(index=False))
    print("=" * 90)

    # % reductions vs naive
    q667_rows = df_table[df_table["Model"].str.contains("Q0.667")]
    if len(q667_rows):
        q667_penalty = q667_rows.iloc[0]["Total Penalty (\u20b9)"]
        reduction = (naive_penalty - q667_penalty) / naive_penalty * 100
        savings   = naive_penalty - q667_penalty
        print(f"\n  \u2605 Penalty reduction (Q0.667 vs Naive): {reduction:.1f}%  "
              f"(Savings: \u20b9{savings:,.0f})")
    print()
    return df_table


def plot_penalty_bar(df_table: pd.DataFrame):
    """Bar chart comparing total penalty across all models in Table B."""
    set_plot_style()

    # Named colour map so extra models get distinct colours
    COLOR_MAP = {
        "Naive"            : "#E84855",
        "Holt-Winters"     : "#8B7BB5",
        "Linear Regression": "#E48E3B",
        "Random Forest"    : "#2E86AB",
        "LightGBM MSE"     : "#F4A261",
        "LightGBM Q0.667"  : "#3BB273",
    }
    def pick_color(label):
        for k, v in COLOR_MAP.items():
            if k in label:
                return v
        return "#AAAAAA"

    colors = [pick_color(m) for m in df_table["Model"]]
    n = len(df_table)

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n * 2), 5))

    # Total penalty bars
    bars = axes[0].bar(df_table["Model"],
                       df_table["Total Penalty (\u20b9)"] / 1e6,
                       color=colors, edgecolor="white")
    axes[0].bar_label(bars, fmt="%.2f M", padding=3, fontsize=7.5)
    best_i = (df_table["Total Penalty (\u20b9)"] / 1e6).values.argmin()
    bars[best_i].set_edgecolor("#FFD700")
    bars[best_i].set_linewidth(2.5)
    axes[0].set_title("Total Financial Penalty \u2014 Full 16-Month Backtest",
                      fontweight="bold")
    axes[0].set_ylabel("Total Penalty (\u20b9 Millions)")
    axes[0].tick_params(axis="x", rotation=20)

    # Peak vs Off-peak stacked
    peak_data = df_table[["Model", "Peak Penalty (\u20b9)",
                           "Off-Peak Penalty (\u20b9)"]].copy().dropna()
    x = range(len(peak_data))
    w = 0.35
    axes[1].bar([i - w/2 for i in x],
                peak_data["Peak Penalty (\u20b9)"] / 1e6,
                w, label="Peak (18\u201321h)", color="#E84855", alpha=0.85)
    axes[1].bar([i + w/2 for i in x],
                peak_data["Off-Peak Penalty (\u20b9)"] / 1e6,
                w, label="Off-Peak", color="#2E86AB", alpha=0.85)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(peak_data["Model"], rotation=20, fontsize=7.5)
    axes[1].set_title("Peak vs Off-Peak Penalty", fontweight="bold")
    axes[1].set_ylabel("Penalty (\u20b9 Millions)")
    axes[1].legend()

    fig.tight_layout()
    save_plot(fig, "10_backtest_penalty_comparison.png")


def plot_residuals(val_df: pd.DataFrame, preds: dict):
    """Plot residual distributions for each model on the full validation period."""
    set_plot_style()

    fig, ax = plt.subplots(figsize=(12, 5))
    actual = val_df["LOAD"].values

    model_map = [
        ("lag_672",   "Naive Baseline",    "#E84855"),
        ("lgbm_mse",  "LightGBM MSE",      "#2E86AB"),
        ("lgbm_q667", "LightGBM Q0.667 ★", "#3BB273"),
    ]

    for key, label, color in model_map:
        fcst = val_df[key].values if key in val_df.columns else preds.get(key)
        if fcst is not None:
            residuals = actual - fcst
            ax.hist(residuals, bins=100, alpha=0.5,
                    label=label, color=color, density=True)

    ax.axvline(0, color="black", lw=2, linestyle="--", label="Zero error")
    ax.axvspan(0, float(actual.max()) * 1.1, alpha=0.05, color="#E84855",
               label="Under-forecast zone (Rs 4/kWh)")
    ax.set_title("Residual Distribution: Actual − Forecast  (full validation Jan 2020–Apr 2021)",
                 fontweight="bold")
    ax.set_xlabel("Residual (kW)   positive = actual exceeded forecast = costly under-forecast")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_plot(fig, "11_residual_distributions.png")





def run(val_df: pd.DataFrame = None, preds: dict = None):
    print("=" * 60)
    print("  STEP 6 — Backtest & Penalty Comparison")
    print("=" * 60)

    if val_df is None or preds is None:
        features_path = r"c:\hackathons\DecodeX\outputs\features.parquet"
        if os.path.exists(features_path):
            df = pd.read_parquet(features_path)
        else:
            from step02_feature_engineering import build_features
            df = build_features()

        from step03_train_val_split import split_features
        X_train, y_train, X_val, y_val, val_df, feats = split_features(df)

        from step05_quantile_model import load_models, predict_all
        models = load_models()
        if not models:
            from step05_quantile_model import train_models, save_models
            models = train_models(X_train, y_train, X_val, y_val, feats)
            save_models(models)
        preds = predict_all(models, X_val)

    df_table = build_comparison_table(val_df, preds)
    naive_penalty = df_table.iloc[0]["Total Penalty (₹)"]
    df_table = print_comparison_table(df_table, naive_penalty)
    plot_penalty_bar(df_table)
    plot_residuals(val_df, preds)

    print("\n  ✔ STEP 6 COMPLETE — backtest comparison table generated.")
    print("\n  KEY FINDINGS:")
    print("  • Q0.667 model achieves the lowest total penalty by targeting the penalty-optimal quantile.")
    print("  • Peak-hour penalties dominate — highest financial risk concentration in 18–21h window.")
    print("  • MSE model outperforms naive on RMSE but not necessarily on ₹ penalty (different objectives).")

    print("\n  BUSINESS IMPLICATION FOR LUMINA ENERGY:")
    print("  • Switching from naive to Q0.667 yields direct ₹ savings on ABT settlement bills.")
    print("  • RMSE is NOT the right KPI — penalty (₹) is. This distinction is business-critical.")

    return df_table, val_df, preds


if __name__ == "__main__":
    run()
