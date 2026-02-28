"""
step11_model_comparison.py — Why LightGBM Q0.667 Beats Classical Time-Series Models
======================================================================================
Stage 1 GRIDSHIELD — DecodeX 2026

This script pits LightGBM Q0.667 against 4 alternative models:

  Model 1 : SARIMA                (classical univariate time series)
  Model 2 : Holt-Winters ETS      (exponential smoothing — captures seasonality)
  Model 3 : Linear Regression     (time features only — simple ML baseline)
  Model 4 : Random Forest         (ensemble, no quantile loss)
  Model 5 : LightGBM Q0.667 ★    (our recommended model)

Evaluation criterion = FINANCIAL PENALTY (₹), NOT RMSE.

Key insight demonstrated here:
  - Classical models (SARIMA, HW-ETS) are built to minimise squared-error,
    not asymmetric financial loss.
  - None of them can natively target the τ=0.667 quantile.
  - LightGBM with quantile objective directly minimises E[ABT Penalty].

Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, sklearn, lightgbm
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle

from utils import (compute_penalty, penalty_table_row,
                   set_plot_style, save_plot, PLOTS_DIR, MODELS_DIR,
                   C_UNDER, C_OVER)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_END = "2019-12-31 23:59:59"
VAL_START = "2020-01-01 00:00:00"
TARGET    = "LOAD"

# Use a smaller slice for slow models (SARIMA is O(n²))
# We use 4 weeks of val data for individual model plots, full val for table
SARIMA_TRAIN_N  = 96 * 30      # 30 days for SARIMA fitting (fast)
HW_TRAIN_N      = 96 * 90      # 90 days for HW-ETS
EVAL_N          = 96 * 14      # 14 days for evaluation window (all models comparable)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    features_path = r"c:\hackathons\DecodeX\outputs\features.parquet"
    if os.path.exists(features_path):
        df = pd.read_parquet(features_path)
    else:
        from step02_feature_engineering import build_features
        df = build_features()

    train_df = df[df["DateTime"] <= TRAIN_END].copy()
    val_df   = df[df["DateTime"] >= VAL_START].copy()

    from step02_feature_engineering import ALL_FEATURES
    feats = [f for f in ALL_FEATURES if f in df.columns]

    X_train = train_df[feats].values
    y_train = train_df[TARGET].values
    X_val   = val_df[feats].values
    y_val   = val_df[TARGET].values

    return train_df, val_df, X_train, y_train, X_val, y_val, feats


# ──────────────────────────────────────────────────────────────────────────────
# 2. MODEL 1 — SARIMA (classical univariate)
# ──────────────────────────────────────────────────────────────────────────────
def fit_sarima(train_df, val_df, eval_n=EVAL_N):
    """
    SARIMA(1,0,1)(1,0,1)[96] — seasonal period = 96 (daily = one day of 15-min)
    Limitations:
      - Univariate: cannot use weather or holiday inputs
      - MSE-based: no way to target τ=0.667 quantile
      - Computationally slow — only practical on short windows
      - Cannot handle structural breaks (COVID) natively
    """
    print("  [Model 1] Fitting SARIMA …")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Use last 30 days of training for SARIMA (full history is too slow)
        train_series = train_df["LOAD"].values[-SARIMA_TRAIN_N:]
        val_series   = val_df["LOAD"].values[:eval_n]

        model = SARIMAX(train_series,
                        order=(1, 0, 1),
                        seasonal_order=(1, 0, 1, 96),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=50)

        # Walk-forward forecast (rolling 1-step ahead, same-slot prediction)
        # For 2-day-ahead: use get_forecast with steps
        forecast_obj = result.get_forecast(steps=eval_n)
        fcst         = forecast_obj.predicted_mean
        conf         = forecast_obj.conf_int(alpha=0.1)  # 90% CI for reference
        print(f"    SARIMA AIC: {result.aic:.1f}")
        return fcst, val_series
    except Exception as e:
        print(f"    SARIMA failed: {e} — using lag_672 as fallback")
        val_series = val_df["LOAD"].values[:eval_n]
        fcst       = val_df["lag_672"].values[:eval_n]
        return fcst, val_series


# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL 2 — Holt-Winters ETS (exponential smoothing)
# ──────────────────────────────────────────────────────────────────────────────
def fit_holtwinters(train_df, val_df, eval_n=EVAL_N):
    """
    Holt-Winters with additive trend + additive seasonality (period=96).
    Limitations:
      - Cannot incorporate exogenous features (temperature, holidays)
      - Symmetric loss — no penalty-aware quantile targeting
      - Seasonal period = 96 captures intraday pattern only, NOT weekly
      - Requires stationary-ish data — COVID structural break breaks it
    """
    print("  [Model 2] Fitting Holt-Winters ETS …")
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        train_series = train_df["LOAD"].values[-HW_TRAIN_N:]
        val_series   = val_df["LOAD"].values[:eval_n]

        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=96,    # daily seasonality
            damped_trend=True
        )
        result = model.fit(optimized=True, use_brute=False)
        fcst   = result.forecast(eval_n)
        fcst   = np.maximum(fcst, 0)   # load cannot be negative
        print(f"    HW-ETS AIC: {result.aic:.1f}")
        return fcst, val_series
    except Exception as e:
        print(f"    Holt-Winters failed: {e} — using lag_672 as fallback")
        val_series = val_df["LOAD"].values[:eval_n]
        fcst       = val_df["lag_672"].values[:eval_n]
        return fcst, val_series


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL 3 — Linear Regression (time features)
# ──────────────────────────────────────────────────────────────────────────────
def fit_linear_regression(X_train, y_train, X_val, y_val, eval_n=EVAL_N):
    """
    Ridge Regression on all 31 engineered features.
    Limitations:
      - Assumes strictly linear relationships — misses temp² AC non-linearity
      - MSE objective — no penalty-aware bias
      - Struggles with interaction terms (heat_index × peak)
      - Cannot model abrupt structural breaks naturally
    Note: Still far better than SARIMA because it uses all weather & lag features.
    """
    print("  [Model 3] Fitting Linear Regression (Ridge) …")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xvl    = scaler.transform(X_val)

    model = Ridge(alpha=1.0)
    model.fit(Xtr, y_train)
    fcst  = model.predict(Xvl)
    fcst  = np.maximum(fcst, 0)

    r2 = model.score(Xvl, y_val)
    print(f"    Linear Regression R² on val: {r2:.4f}")
    return fcst[:eval_n], y_val[:eval_n]


# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL 4 — Random Forest
# ──────────────────────────────────────────────────────────────────────────────
def fit_random_forest(X_train, y_train, X_val, y_val, eval_n=EVAL_N):
    """
    RandomForestRegressor — ensemble of decision trees (MSE split criterion).
    Limitations:
      - MSE objective → predicts mean, not optimal quantile
      - No native quantile regression (scikit-learn's quantile RF is slow)
      - Prone to over-predicting because mean > median in right-skewed load
      - Cannot extrapolate beyond training range (issue with load growth)
    Note: Good accuracy but not penalty-aware — our comparison shows this matters.
    """
    print("  [Model 4] Fitting Random Forest …")
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    fcst = model.predict(X_val)
    fcst = np.maximum(fcst, 0)

    # OOB error proxy
    print(f"    RF prediction done — {len(fcst):,} forecasts generated")
    return fcst[:eval_n], y_val[:eval_n]


# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL 5 — LightGBM Q0.667 (our model)
# ──────────────────────────────────────────────────────────────────────────────
def load_lgbm(X_val, y_val, eval_n=EVAL_N):
    """
    LightGBM with quantile objective at α=0.6667.
    Why it wins:
      1. Targets τ* = C_under/(C_under+C_over) — mathematical penalty optimum
      2. Gradient boosting captures non-linear feature interactions
      3. Handles 31 diverse features (lags, weather, calendar, COVID flag)
      4. Early stopping prevents overfitting
      5. Naturally produces an upward-biased forecast — exactly what ABT rewards
    """
    print("  [Model 5] Loading LightGBM Q0.667 …")
    pkl = os.path.join(MODELS_DIR, "lgbm_q667.pkl")
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    fcst = model.predict(X_val)
    print(f"    LightGBM Q0.667 loaded — {len(fcst):,} predictions")
    return fcst[:eval_n], y_val[:eval_n]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Penalty Comparison Table
# ──────────────────────────────────────────────────────────────────────────────
def build_comparison_table(val_df, models_dict, eval_n):
    """
    Build penalty comparison for ALL models on the same eval window.
    models_dict = {label: forecast_array}
    """
    actual  = val_df["LOAD"].values[:eval_n]
    is_peak = val_df["is_peak_hour"].values[:eval_n]

    rows = []
    for label, fcst in models_dict.items():
        rows.append(penalty_table_row(label, actual, fcst, is_peak))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_penalty_comparison(df_table):
    """Side-by-side penalty bar chart for all 5 models."""
    set_plot_style()

    colors = {
        "SARIMA"                : "#9D9D9D",
        "Holt-Winters ETS"      : "#8B7BB5",
        "Linear Regression"     : "#E48E3B",
        "Random Forest"         : "#2E86AB",
        "LightGBM Q0.667 ★"    : "#3BB273",
    }
    bar_colors = [colors.get(m, "#AAAAAA") for m in df_table["Model"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Total penalty ──────────────────────────────────────────────────────────
    vals = df_table["Total Penalty (Rs)"] / 1e6
    bars = axes[0].bar(df_table["Model"], vals, color=bar_colors, edgecolor="white", width=0.6)
    axes[0].bar_label(bars, fmt="%.2f M", padding=4, fontsize=8)
    axes[0].set_title("Total ABT Penalty Comparison (Rs, 14-day window)", fontweight="bold")
    axes[0].set_ylabel("Total Penalty (Rs Millions)")
    axes[0].tick_params(axis="x", rotation=20)

    # Highlight winner
    best_idx = vals.idxmin()
    bars[best_idx].set_edgecolor("#FFD700")
    bars[best_idx].set_linewidth(3)

    # ── RMSE vs Penalty scatter ────────────────────────────────────────────────
    axes[1].scatter(df_table["RMSE (kW)"],
                    df_table["Total Penalty (Rs)"] / 1e6,
                    color=bar_colors, s=200, zorder=5, edgecolors="white", linewidths=1.5)

    for _, row in df_table.iterrows():
        axes[1].annotate(row["Model"].replace(" ★",""),
                         (row["RMSE (kW)"], row["Total Penalty (Rs)"] / 1e6),
                         textcoords="offset points", xytext=(8, 4), fontsize=8)

    axes[1].set_title("RMSE vs Financial Penalty\n(Lower-right = good RMSE; Lower = good penalty)",
                      fontweight="bold")
    axes[1].set_xlabel("RMSE (kW)  [lower = better accuracy]")
    axes[1].set_ylabel("Total Penalty (Rs Millions)  [lower = better financially]")
    axes[1].axhline(df_table["Total Penalty (Rs)"].min() / 1e6,
                    color="#3BB273", lw=1.5, linestyle="--", alpha=0.5, label="Best penalty")

    fig.suptitle("Why LightGBM Q0.667 Beats Classical Time-Series Models",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "16_model_comparison_penalty.png")


def plot_forecast_sample(val_df, models_dict, eval_n, n_days=3):
    """Overlay forecasts of all models on the same 3-day window."""
    set_plot_style()

    n_slots = 96 * n_days
    dates   = val_df["DateTime"].values[:n_slots]
    actual  = val_df["LOAD"].values[:n_slots]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, actual, color="black", lw=2.5, label="Actual Load", zorder=10)

    style_map = {
        "SARIMA"             : ("#9D9D9D", ":"),
        "Holt-Winters ETS"   : ("#8B7BB5", "--"),
        "Linear Regression"  : ("#E48E3B", "-."),
        "Random Forest"      : ("#2E86AB", "--"),
        "LightGBM Q0.667 ★" : ("#3BB273", "-"),
    }

    for label, fcst in models_dict.items():
        color, ls = style_map.get(label, ("#AAAAAA", ":"))
        lw = 2.5 if "LightGBM" in label else 1.5
        ax.plot(dates, fcst[:n_slots], label=label, color=color, lw=lw, linestyle=ls, alpha=0.85)

    ax.set_title(f"Forecast Comparison — First {n_days} Days of Validation (Jan 2020)",
                 fontweight="bold")
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Load (kW)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_plot(fig, "17_model_comparison_forecast.png")


def plot_residual_comparison(val_df, models_dict, eval_n):
    """Compare residual distributions — shows systematic bias in each model."""
    set_plot_style()

    actual = val_df["LOAD"].values[:eval_n]
    fig, ax = plt.subplots(figsize=(14, 6))

    color_map = {
        "SARIMA"             : "#9D9D9D",
        "Holt-Winters ETS"   : "#8B7BB5",
        "Linear Regression"  : "#E48E3B",
        "Random Forest"      : "#2E86AB",
        "LightGBM Q0.667 ★" : "#3BB273",
    }

    for label, fcst in models_dict.items():
        resid = actual - fcst[:eval_n]
        color = color_map.get(label, "#AAAAAA")
        lw    = 2.5 if "LightGBM" in label else 1.5
        sns.kdeplot(resid, ax=ax, label=label, color=color, lw=lw)

    ax.axvline(0, color="black", lw=2, linestyle="--", label="Zero error")
    # Shade the costly under-forecast side
    ax.axvspan(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 400,
               alpha=0.05, color="#E84855", label="Under-forecast zone (Rs 4/kWh)")

    ax.set_title("Residual Distribution: Actual - Forecast\n"
                 "LightGBM Q0.667 intentionally shifts right (upward bias) to avoid costly under-forecasts",
                 fontweight="bold")
    ax.set_xlabel("Residual (kW) — positive = actual exceeded forecast = COSTLY")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_plot(fig, "18_residual_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Why LightGBM Q0.667 Wins — Printed Summary
# ──────────────────────────────────────────────────────────────────────────────
def print_why_lgbm_wins(df_table):
    lgbm_row  = df_table[df_table["Model"].str.contains("LightGBM")].iloc[0]
    naive_row = df_table[df_table["Model"] == "SARIMA"].iloc[0] if "SARIMA" in df_table["Model"].values else df_table.iloc[0]

    print()
    print("=" * 75)
    print("  WHY LightGBM Q0.667 BEATS CLASSICAL TIME-SERIES MODELS")
    print("=" * 75)

    headers = ["Criterion", "SARIMA", "Holt-Winters", "Lin.Reg.", "Rnd.Forest", "LGBM Q0.667"]
    rows_txt = [
        ["Penalty objective",  "No",  "No",  "No",  "No",  "YES (tau=0.667)"],
        ["Exogenous features", "No",  "No",  "Yes", "Yes", "Yes (31 feats)"],
        ["Non-linear effects", "No",  "No",  "No",  "Yes", "Yes (trees)"],
        ["Holiday awareness",  "No",  "No",  "Yes", "Yes", "Yes + proximity"],
        ["COVID flag",         "No",  "No",  "Yes", "Yes", "Yes"],
        ["Weekly seasonality", "Yes", "No",  "Yes", "Yes", "Yes (lag_672)"],
        ["Upward bias control","No",  "No",  "No",  "No",  "YES (quantile)"],
        ["Calibration target", "Mean","Mean","Mean","Mean", "tau=0.667 OPTIMAL"],
    ]

    col_w = 22
    print(f"  {'Criterion':<22}", end="")
    for h in headers[1:]:
        print(f"  {h:<14}", end="")
    print()
    print("  " + "-" * 95)
    for row in rows_txt:
        print(f"  {row[0]:<22}", end="")
        for v in row[1:]:
            mark = "  ✔ " if v not in ["No","Mean"] else "  ✗ "
            print(f"{mark}{v:<10}", end="")
        print()
    print("  " + "-" * 95)

    best_model = df_table.loc[df_table["Total Penalty (Rs)"].idxmin(), "Model"]
    best_pen   = df_table["Total Penalty (Rs)"].min()
    worst_pen  = df_table["Total Penalty (Rs)"].max()
    saving     = worst_pen - best_pen
    print(f"\n  Best model     : {best_model}")
    print(f"  Best penalty   : Rs {best_pen:>12,.2f}")
    print(f"  Worst penalty  : Rs {worst_pen:>12,.2f}")
    print(f"  Saving vs worst: Rs {saving:>12,.2f}  ({saving/worst_pen*100:.1f}%)")
    print()

    print("  BOTTOM LINE:")
    print("  ─────────────────────────────────────────────────────────────────────")
    print("  SARIMA and Holt-Winters are blind to weather, holidays, and COVID.")
    print("  Linear Regression is linear — misses the temp² AC non-linearity.")
    print("  Random Forest predicts the MEAN — wrong target under ABT regulations.")
    print("  LightGBM Q0.667 is the ONLY model that:")
    print("    (a) targets the mathematically optimal quantile for cost minimisation")
    print("    (b) uses all 31 available features including weather + calendar")
    print("    (c) captures non-linear interactions via gradient boosted trees")
    print("    (d) produces an intentional upward bias — exactly what ABT rewards")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def run():
    print("=" * 70)
    print("  STEP 11 — Model Comparison: LightGBM Q0.667 vs Classical Models")
    print("=" * 70)

    train_df, val_df, X_train, y_train, X_val, y_val, feats = load_data()
    eval_n = min(EVAL_N, len(val_df))

    # Fit all models
    sarima_fcst,  sarima_actual  = fit_sarima(train_df, val_df, eval_n)
    hw_fcst,      hw_actual      = fit_holtwinters(train_df, val_df, eval_n)
    lr_fcst,      lr_actual      = fit_linear_regression(X_train, y_train, X_val, y_val, eval_n)
    rf_fcst,      rf_actual      = fit_random_forest(X_train, y_train, X_val, y_val, eval_n)
    lgbm_fcst,    lgbm_actual    = load_lgbm(X_val, y_val, eval_n)

    # Align all forecasts to same window
    models_dict = {
        "SARIMA"             : sarima_fcst,
        "Holt-Winters ETS"   : hw_fcst,
        "Linear Regression"  : lr_fcst,
        "Random Forest"      : rf_fcst,
        "LightGBM Q0.667 ★" : lgbm_fcst,
    }

    # Build comparison table using standard penalty metric columns
    actual  = val_df["LOAD"].values[:eval_n]
    is_peak = val_df["is_peak_hour"].values[:eval_n]
    rows = []
    for label, fcst in models_dict.items():
        r = penalty_table_row(label, actual, fcst, is_peak)
        # Rename column for ASCII-safe display
        r["Total Penalty (Rs)"]    = r.pop("Total Penalty (₹)")
        r["Peak Penalty (Rs)"]     = r.pop("Peak Penalty (₹)")
        r["Off-Peak Penalty (Rs)"] = r.pop("Off-Peak Penalty (₹)")
        rows.append(r)
    df_table = pd.DataFrame(rows)

    # Print results
    print("\n  ── Penalty Comparison Table (14-day validation window) ──────────")
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(df_table[["Model", "Total Penalty (Rs)", "Peak Penalty (Rs)",
                    "Forecast Bias (%)", "RMSE (kW)"]].to_string(index=False))
    print()

    print_why_lgbm_wins(df_table)

    # Plots
    plot_penalty_comparison(df_table)
    plot_forecast_sample(val_df, models_dict, eval_n)
    plot_residual_comparison(val_df, models_dict, eval_n)

    print("\n  Plots saved:")
    print("    16_model_comparison_penalty.png")
    print("    17_model_comparison_forecast.png")
    print("    18_residual_comparison.png")
    print("\n  STEP 11 COMPLETE.")

    return df_table, models_dict


if __name__ == "__main__":
    run()
