"""
utils.py — Shared utilities for GRIDSHIELD Forecast Risk Advisory Pipeline
==========================================================================
Contains:
  - DateTime parser for SAS-style format "01APR2013:00:15:00"
  - Asymmetric ABT penalty calculator
  - Standard penalty report generator
  - Common plot style setup
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR    = r"c:\hackathons\DecodeX"
OUTPUT_DIR  = r"c:\hackathons\DecodeX\outputs"
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")

LOAD_FILE    = os.path.join(DATA_DIR, "Electric_Load_Data_Train.csv")
WEATHER_FILE = os.path.join(DATA_DIR, "External_Factor_Data_Train.csv")
EVENTS_FILE  = os.path.join(DATA_DIR, "Events_Data.csv")

# ──────────────────────────────────────────────────────────────────────────────
# PENALTY CONSTANTS (Maharashtra ABT)
# ──────────────────────────────────────────────────────────────────────────────
C_UNDER = 4.0   # ₹ per kWh — actual > forecast (under-forecast)
C_OVER  = 2.0   # ₹ per kWh — forecast > actual (over-forecast)

# Optimal quantile for asymmetric penalty: τ* = C_under / (C_under + C_over)
OPTIMAL_QUANTILE = C_UNDER / (C_UNDER + C_OVER)   # 0.6667
PEAK_QUANTILE    = 0.75                            # extra buffer for peak hours


# ──────────────────────────────────────────────────────────────────────────────
# DATETIME PARSER
# ──────────────────────────────────────────────────────────────────────────────
def parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse SAS-format datetime strings like '01APR2013:00:15:00'
    into pandas Timestamps.
    """
    return pd.to_datetime(series, format="%d%b%Y:%H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# PENALTY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def compute_penalty(actual: np.ndarray, forecast: np.ndarray,
                    is_peak: np.ndarray = None) -> dict:
    """
    Compute asymmetric ABT financial penalty.
    
    Under-forecast: actual > forecast → penalty = C_UNDER * (actual - forecast) kWh
    Over-forecast : forecast > actual → penalty = C_OVER  * (forecast - actual) kWh
    
    Note: Load is in kW at 15-min resolution → kWh = kW × 0.25
    
    Parameters
    ----------
    actual   : np.ndarray, actual load (kW)
    forecast : np.ndarray, forecasted load (kW)
    is_peak  : np.ndarray (bool), mask for peak hours 18–21
    
    Returns
    -------
    dict with penalty breakdown
    """
    delta = actual - forecast   # positive => under-forecast

    under_kw = np.maximum(delta, 0.0)
    over_kw  = np.maximum(-delta, 0.0)

    # Convert kW → kWh (15-min slots)
    under_kwh = under_kw * 0.25
    over_kwh  = over_kw  * 0.25

    total_penalty  = C_UNDER * under_kwh.sum() + C_OVER * over_kwh.sum()
    under_penalty  = C_UNDER * under_kwh.sum()
    over_penalty   = C_OVER  * over_kwh.sum()

    # Bias: positive = under-forecast on average
    bias_pct = (forecast.mean() - actual.mean()) / actual.mean() * 100

    result = {
        "total_penalty_INR"  : round(total_penalty, 2),
        "under_penalty_INR"  : round(under_penalty, 2),
        "over_penalty_INR"   : round(over_penalty, 2),
        "forecast_bias_pct"  : round(bias_pct, 4),
        "p95_abs_dev_kW"     : round(np.percentile(np.abs(delta), 95), 2),
        "rmse_kW"            : round(np.sqrt(np.mean(delta**2)), 2),
    }

    if is_peak is not None:
        mask = is_peak.astype(bool)
        result["peak_penalty_INR"]     = round(
            C_UNDER * (under_kwh[mask]).sum() + C_OVER * (over_kwh[mask]).sum(), 2)
        result["offpeak_penalty_INR"]  = round(
            C_UNDER * (under_kwh[~mask]).sum() + C_OVER * (over_kwh[~mask]).sum(), 2)

    return result


def penalty_table_row(name: str, actual: np.ndarray, forecast: np.ndarray,
                       is_peak: np.ndarray) -> dict:
    """Return a flat dict suitable for building a comparison DataFrame."""
    p = compute_penalty(actual, forecast, is_peak)
    return {
        "Model"               : name,
        "Total Penalty (₹)"   : p["total_penalty_INR"],
        "Peak Penalty (₹)"    : p.get("peak_penalty_INR", np.nan),
        "Off-Peak Penalty (₹)": p.get("offpeak_penalty_INR", np.nan),
        "Forecast Bias (%)"   : p["forecast_bias_pct"],
        "95th pct Dev (kW)"   : p["p95_abs_dev_kW"],
        "RMSE (kW)"           : p["rmse_kW"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# PLOT STYLE HELPER
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# SHARED LightGBM BASE PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
BASE_PARAMS = dict(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 127,
    min_child_samples = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
    random_state      = 42,
)


def set_plot_style():
    """Apply a clean, professional Seaborn-based plot style."""
    import seaborn as sns
    sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi"     : 120,
        "figure.figsize" : (14, 5),
        "axes.titlesize" : 14,
        "axes.labelsize" : 12,
    })


def save_plot(fig, filename: str, subfolder: str = ""):
    """Save a matplotlib figure to outputs/plots/ with optional subfolder."""
    folder = os.path.join(PLOTS_DIR, subfolder) if subfolder else PLOTS_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved plot → {path}")
