"""
step12_sldc_submission.py — 2-Day Ahead Forecast for SLDC Submission
=====================================================================
Step 12 (Final) of the GRIDSHIELD Forecast Risk Advisory Pipeline.

Under Maharashtra ABT regulations, Lumina Energy must submit a 2-day ahead
load forecast at 15-minute resolution (96 slots/day x 2 = 192 rows) to the
State Load Dispatch Centre (SLDC) before 10:00 AM of the schedule day.

This step:
  1. Determines the forecast horizon: the 2 days AFTER the last historical date
  2. Builds all 31 features for those 192 future slots using:
       - Known lags from historical data (lag_192, lag_288, lag_672)
       - Rolling statistics from the last 7 days of history
       - Climatological (same-month) weather averages as forward weather inputs
       - Calendar/temporal features (exact — no uncertainty)
       - Holiday flags from the Events calendar
  3. Runs LightGBM Q0.667 (penalty-optimal model) for the point forecast
  4. Runs P10 / P90 quantile models for uncertainty bands
  5. Applies peak-hour buffer: uses Q0.75 for slots 18:00–21:00
  6. Outputs the SLDC dispatch schedule as:
       outputs/SLDC_Forecast_YYYY-MM-DD_to_YYYY-MM-DD.csv
  7. Saves a forecast chart (Plot 19)

Output CSV columns (SLDC format):
  Date          — YYYY-MM-DD
  TimeSlot      — HH:MM (start of each 15-min slot)
  SlotNo        — 1–96 (slot number within the day, as per ABT)
  Forecast_kW   — Q0.667 point forecast (rounded to 1 decimal)
  P10_kW        — Lower bound (10th percentile)
  P90_kW        — Upper bound (90th percentile)
  IsPeakHour    — 1 if 18:00–21:00, else 0
  Quantile_Used — 0.667 (normal) or 0.750 (peak-hour buffer)
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import set_plot_style, save_plot, EVENTS_FILE

MODELS_DIR = r"c:\hackathons\DecodeX\outputs\models"
OUTPUTS_DIR = r"c:\hackathons\DecodeX\outputs"

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE LIST (must match the 31 features used in training)
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "hour", "minute", "day_of_week", "month", "year", "quarter",
    "is_weekend", "is_peak_hour", "slot",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month",
    "is_holiday", "days_to_next_holiday", "days_since_last_holiday",
    "is_covid_period",
    "lag_192", "lag_288", "lag_672",
    "rolling_mean_672", "rolling_std_672",
    "ACT_TEMP", "ACT_HEAT_INDEX", "ACT_HUMIDITY", "ACT_RAIN", "COOL_FACTOR",
    "temp_squared", "heat_index_x_peak",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load models
# ──────────────────────────────────────────────────────────────────────────────
def load_forecast_models():
    """Load Q0.667 (point), P10, P90, Q0.75 (peak buffer) models."""
    models = {}
    needed = {
        "lgbm_q667": "LightGBM Q0.667 (point forecast)",
        "lgbm_q10":  "LightGBM P10 (lower bound)",
        "lgbm_q90":  "LightGBM P90 (upper bound)",
        "lgbm_q75":  "LightGBM Q0.75 (peak-hour buffer)",
    }
    for key, label in needed.items():
        path = os.path.join(MODELS_DIR, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
            print(f"  ✔ Loaded  {label}")
        else:
            print(f"  ✗ MISSING {path}  — {label} will be unavailable")
    return models


# ──────────────────────────────────────────────────────────────────────────────
# 2. Build future feature matrix (192 slots = 2 days)
# ──────────────────────────────────────────────────────────────────────────────
def build_future_features(historical_df: pd.DataFrame,
                           df_events: pd.DataFrame,
                           n_slots: int = 192) -> pd.DataFrame:
    """
    Construct the feature matrix for the next n_slots time slots
    (default 192 = 2 days × 96 slots/day).

    Lag strategy (all lags are ≥192 slots so NO leakage):
      lag_192  — from 2 days back  (already in history)
      lag_288  — from 3 days back
      lag_672  — from 7 days back (same weekday, same slot)

    Weather strategy:
      Uses the climatological average for the same month + same slot
      computed from the historical data. In production this would be
      replaced by IMD 2-day weather forecast API inputs.
    """
    hist = historical_df.copy()
    last_dt = hist["DateTime"].max()

    # Future timestamps: next n_slots × 15 min
    freq = pd.tseries.offsets.Minute(15)
    future_dts = pd.date_range(start=last_dt + freq,
                               periods=n_slots, freq="15min")
    future = pd.DataFrame({"DateTime": future_dts})

    # ── Temporal features ────────────────────────────────────────────────────
    dt = future["DateTime"]
    future["hour"]        = dt.dt.hour
    future["minute"]      = dt.dt.minute
    future["day_of_week"] = dt.dt.dayofweek
    future["month"]       = dt.dt.month
    future["year"]        = dt.dt.year
    future["quarter"]     = dt.dt.quarter
    future["slot"]        = future["hour"] * 4 + future["minute"] // 15
    future["is_weekend"]   = (future["day_of_week"] >= 5).astype(np.int8)
    future["is_peak_hour"] = ((future["hour"] >= 18) & (future["hour"] <= 21)).astype(np.int8)  # matches training (18:00–21:59)

    # Cyclical
    future["sin_hour"]  = np.sin(2 * np.pi * future["hour"]        / 24)
    future["cos_hour"]  = np.cos(2 * np.pi * future["hour"]        / 24)
    future["sin_dow"]   = np.sin(2 * np.pi * future["day_of_week"] / 7)
    future["cos_dow"]   = np.cos(2 * np.pi * future["day_of_week"] / 7)
    future["sin_month"] = np.sin(2 * np.pi * future["month"]       / 12)
    future["cos_month"] = np.cos(2 * np.pi * future["month"]       / 12)

    # ── COVID flag ───────────────────────────────────────────────────────────
    future["is_covid_period"] = 0   # forecast is post-COVID

    # ── Holiday features ─────────────────────────────────────────────────────
    df_events_clean = df_events.copy()
    holiday_dates = sorted(
        df_events_clean[df_events_clean["Holiday_Ind"] == 1]["Date"]
        .dt.normalize().unique()
    )
    future_dates = future["DateTime"].dt.normalize()
    holiday_set  = set(holiday_dates)
    future["is_holiday"] = future_dates.isin(holiday_set).astype(np.int8)

    def proximity(d, hols):
        ts = pd.Timestamp(d)
        past   = [h for h in hols if h <= ts]
        future_h = [h for h in hols if h >= ts]
        dsince = (ts - past[-1]).days   if past    else 365
        dto    = (future_h[0] - ts).days if future_h else 365
        return dsince, dto

    prox_list = [proximity(d, holiday_dates) for d in future_dates]
    future["days_since_last_holiday"] = [p[0] for p in prox_list]
    future["days_to_next_holiday"]    = [p[1] for p in prox_list]

    # ── Lag features — pulled directly from history ──────────────────────────
    # Index the history by DateTime for fast lookup
    hist_indexed = hist.set_index("DateTime")["LOAD"]

    def lag_lookup(future_dt, lag_slots):
        target_dt = future_dt - pd.Timedelta(minutes=15 * lag_slots)
        if target_dt in hist_indexed.index:
            return hist_indexed[target_dt]
        # Fallback: same slot, nearest available date
        return hist_indexed.iloc[-lag_slots] if lag_slots <= len(hist_indexed) else np.nan

    future["lag_192"] = [lag_lookup(dt, 192) for dt in future_dts]
    future["lag_288"] = [lag_lookup(dt, 288) for dt in future_dts]
    future["lag_672"] = [lag_lookup(dt, 672) for dt in future_dts]

    # Rolling stats from the last 672 slots of history (7 days)
    last_672_load = hist_indexed.iloc[-672:].values
    future["rolling_mean_672"] = last_672_load.mean()
    future["rolling_std_672"]  = last_672_load.std()

    # ── Weather: climatological monthly average by slot ──────────────────────
    # "What does the weather typically look like in May at 14:15?"
    forecast_month = future["month"].iloc[0]
    weather_cols   = ["ACT_TEMP", "ACT_HEAT_INDEX", "ACT_HUMIDITY",
                      "ACT_RAIN", "COOL_FACTOR"]
    # Compute slot-level climatological means for this month
    hist_month = hist[hist["DateTime"].dt.month == forecast_month]
    if len(hist_month) == 0:
        # Fallback: use last 30 days
        hist_month = hist.iloc[-30 * 96:]

    clim = hist_month.groupby("slot")[weather_cols].mean()

    for col in weather_cols:
        if col in clim.columns:
            future[col] = future["slot"].map(clim[col])
            future[col] = future[col].fillna(hist[col].mean() if col in hist.columns else 30.0)
        else:
            default = {"ACT_TEMP": 35.0, "ACT_HEAT_INDEX": 38.0,
                       "ACT_HUMIDITY": 70.0, "ACT_RAIN": 0.0, "COOL_FACTOR": 25.0}
            future[col] = default.get(col, 0.0)

    future["temp_squared"]      = future["ACT_TEMP"] ** 2
    future["heat_index_x_peak"] = future["ACT_HEAT_INDEX"] * future["is_peak_hour"]

    return future


# ──────────────────────────────────────────────────────────────────────────────
# 3. Generate forecasts
# ──────────────────────────────────────────────────────────────────────────────
def generate_sldc_forecast(future: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Run all four models on the future feature matrix.
    Applies peak-hour buffer: Q0.75 replaces Q0.667 for 18:00–21:00 slots.
    """
    X = future[FEATURE_COLS].values

    out = future[["DateTime", "slot", "is_peak_hour"]].copy()

    # Point forecast: Q0.667 (or Q0.75 peak buffer)
    q667 = np.maximum(models["lgbm_q667"].predict(X), 0) if "lgbm_q667" in models else np.full(len(X), np.nan)
    q75  = np.maximum(models["lgbm_q75"].predict(X),  0) if "lgbm_q75"  in models else q667
    p10  = np.maximum(models["lgbm_q10"].predict(X),  0) if "lgbm_q10"  in models else q667 * 0.85
    p90  = np.maximum(models["lgbm_q90"].predict(X),  0) if "lgbm_q90"  in models else q667 * 1.15

    peak_mask = out["is_peak_hour"].values.astype(bool)
    point = q667.copy()
    point[peak_mask] = q75[peak_mask]   # peak-hour buffer

    out["Forecast_kW"]   = np.round(point, 1)
    out["P10_kW"]        = np.round(p10, 1)
    out["P90_kW"]        = np.round(p90, 1)
    out["Quantile_Used"] = np.where(peak_mask, 0.750, 0.667)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. Format as SLDC dispatch schedule
# ──────────────────────────────────────────────────────────────────────────────
def format_sldc_csv(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format forecast into SLDC submission format.
    SlotNo is 1-indexed per day (1=00:00, 96=23:45), as per ABT convention.
    """
    df = forecast_df.copy()
    df["Date"]        = df["DateTime"].dt.strftime("%Y-%m-%d")
    df["TimeSlot"]    = df["DateTime"].dt.strftime("%H:%M")
    df["SlotNo"]      = (df["slot"] + 1).astype(int)   # 1-indexed per ABT
    df["IsPeakHour"]  = df["is_peak_hour"].astype(int)

    sldc = df[["Date", "TimeSlot", "SlotNo",
               "Forecast_kW", "P10_kW", "P90_kW",
               "IsPeakHour", "Quantile_Used"]].reset_index(drop=True)
    return sldc


# ──────────────────────────────────────────────────────────────────────────────
# 5. Plot the 2-day forecast (Plot 19)
# ──────────────────────────────────────────────────────────────────────────────
def plot_sldc_forecast(sldc: pd.DataFrame, historical_df: pd.DataFrame):
    """
    Plot 19 — 2-day ahead SLDC forecast with P10-P90 uncertainty band,
    preceded by the last 3 days of actual history for context.
    """
    set_plot_style()

    # Last 3 days of actual history
    last_3d = historical_df.tail(3 * 96)
    hist_dt   = last_3d["DateTime"]
    hist_load = last_3d["LOAD"]

    fut_dt    = pd.to_datetime(sldc["Date"] + " " + sldc["TimeSlot"])
    fut_fcst  = sldc["Forecast_kW"].values
    fut_p10   = sldc["P10_kW"].values
    fut_p90   = sldc["P90_kW"].values

    fig, ax = plt.subplots(figsize=(18, 6))

    # Historical context
    ax.plot(hist_dt, hist_load, color="#2E86AB", lw=1.8,
            label="Historical Load (last 3 days)", zorder=5)

    # Vertical divider at forecast start
    ax.axvline(fut_dt.iloc[0], color="black", lw=1.5, linestyle=":",
               label="Forecast horizon start")

    # Uncertainty band
    ax.fill_between(fut_dt, fut_p10, fut_p90,
                    alpha=0.20, color="#F4A261", label="P10–P90 uncertainty band")

    # Point forecast
    ax.plot(fut_dt, fut_fcst, color="#E84855", lw=2.2,
            label="LightGBM Q0.667 forecast (Q0.75 at peak hours)", zorder=6)

    # Shade P10 band (under-forecast risk zone)
    ax.fill_between(fut_dt, fut_p10, fut_fcst,
                    alpha=0.10, color="#E84855", hatch="///")

    # Highlight peak windows
    for day in pd.to_datetime(sldc["Date"].unique()):
        peak_s = day + pd.Timedelta(hours=18)
        peak_e = day + pd.Timedelta(hours=22)
        ax.axvspan(peak_s, peak_e, alpha=0.08, color="#FFD700",
                   label="Peak Hours 18-22h (Q0.75 buffer)")

    # Format
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    ax.set_title("GRIDSHIELD — 2-Day Ahead SLDC Dispatch Schedule\n"
                 "Lumina Energy | Maharashtra ABT | Q0.667 Quantile Regression",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Load (kW)")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend(*zip(*unique), fontsize=9, loc="upper left", ncol=2)

    # Annotations
    peak_fcst = fut_fcst[sldc["IsPeakHour"].values == 1]
    if len(peak_fcst):
        ax.annotate(f"Max peak forecast: {peak_fcst.max():,.0f} kW",
                    xy=(fut_dt[sldc["IsPeakHour"].values == 1].iloc[0],
                        peak_fcst.max()),
                    xytext=(fut_dt.iloc[20], peak_fcst.max() + 30),
                    arrowprops=dict(arrowstyle="->", color="#E84855"),
                    color="#E84855", fontsize=8.5)

    fig.tight_layout()
    save_plot(fig, "19_sldc_2day_forecast.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Print submission summary
# ──────────────────────────────────────────────────────────────────────────────
def print_sldc_summary(sldc: pd.DataFrame, csv_path: str):
    dates = sldc["Date"].unique()
    print(f"\n  =========================================================")
    print(f"  SLDC 2-DAY AHEAD DISPATCH SCHEDULE")
    print(f"  =========================================================")
    print(f"  Forecast dates : {dates[0]}  to  {dates[-1]}")
    print(f"  Time slots     : {len(sldc)} rows  (96 slots x {len(dates)} days)")
    print(f"  Model          : LightGBM Q0.667 (Q0.75 at peak hours)")
    print(f"  Output file    : {csv_path}")
    print()

    for date in dates:
        day = sldc[sldc["Date"] == date]
        peak = day[day["IsPeakHour"] == 1]
        print(f"  {date}:")
        print(f"    Total forecast energy : {day['Forecast_kW'].sum() * 0.25 / 1000:,.2f} MWh")
        print(f"    Peak avg load         : {peak['Forecast_kW'].mean():,.0f} kW")
        print(f"    Max slot forecast     : {day['Forecast_kW'].max():,.0f} kW  "
              f"@ {day.loc[day['Forecast_kW'].idxmax(), 'TimeSlot']}")
        print(f"    P10–P90 avg band      : {(day['P90_kW'] - day['P10_kW']).mean():,.0f} kW")
    print()
    print(f"  PENALTY RISK NOTE:")
    print(f"  Under-forecast (actual > forecast) => Rs 4/kWh penalty")
    print(f"  Q0.667 intentionally biases upward by the penalty-optimal")
    print(f"  amount. Q0.75 applied during peak hours (18-21h) for extra buffer.")
    print(f"  =========================================================\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ──────────────────────────────────────────────────────────────────────────────
def run(historical_df: pd.DataFrame = None, feature_df: pd.DataFrame = None):
    """
    Generate 2-day ahead SLDC forecast.

    Parameters
    ----------
    historical_df : Raw DataFrame with DateTime + LOAD columns (for lag lookups).
                    If None, loads from features.parquet.
    feature_df    : Full feature DataFrame (used only for weather climatology).
    """
    print("=" * 60)
    print("  STEP 12 — SLDC 2-Day Ahead Forecast Submission")
    print("=" * 60)

    # Load raw data if not provided
    if historical_df is None:
        features_path = r"c:\hackathons\DecodeX\outputs\features.parquet"
        if os.path.exists(features_path):
            historical_df = pd.read_parquet(features_path)
            print("  Loaded historical data from features.parquet")
        else:
            from step02_feature_engineering import build_features
            historical_df = build_features()

    # Load events for holiday features
    df_events = pd.read_csv(EVENTS_FILE)
    df_events["Date"] = pd.to_datetime(df_events["Date"], dayfirst=True, errors="coerce")
    df_events = df_events.dropna(subset=["Date"])

    # Load models
    print("\n  Loading forecast models ...")
    models = load_forecast_models()
    if "lgbm_q667" not in models:
        print("  ERROR: lgbm_q667 model not found — run pipeline first to train models.")
        return None

    # Determine forecast window
    last_dt      = pd.to_datetime(historical_df["DateTime"]).max()
    forecast_start = last_dt + pd.Timedelta(minutes=15)
    forecast_end   = forecast_start + pd.Timedelta(hours=47, minutes=45)
    print(f"\n  Last historical slot : {last_dt}")
    print(f"  Forecast window      : {forecast_start}  to  {forecast_end}")
    print(f"  Total slots          : 192  ({2} days × 96 slots/day)")

    # Build future features
    print("\n  Building future feature matrix ...")
    future_df = build_future_features(historical_df, df_events, n_slots=192)
    print(f"  Future feature matrix: {future_df.shape}")

    # Check all required features exist
    missing = [c for c in FEATURE_COLS if c not in future_df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")

    # Run models
    print("\n  Generating 192-slot forecast ...")
    forecast_df = generate_sldc_forecast(future_df, models)

    # Format SLDC CSV
    sldc = format_sldc_csv(forecast_df)

    # Save CSV
    date_tag  = f"{sldc['Date'].iloc[0]}_to_{sldc['Date'].iloc[-1]}"
    csv_name  = f"SLDC_Forecast_{date_tag}.csv"
    csv_path  = os.path.join(OUTPUTS_DIR, csv_name)
    sldc.to_csv(csv_path, index=False)
    print(f"\n  ✔ SLDC forecast saved  -> {csv_path}")

    # Print a preview
    print("\n  Sample (first 8 rows):")
    print(sldc.head(8).to_string(index=False))

    # Plot
    print("\n  Generating Plot 19 (2-day forecast chart) ...")
    plot_sldc_forecast(sldc, historical_df)

    # Summary
    print_sldc_summary(sldc, csv_path)

    print("  ✔ STEP 12 COMPLETE — SLDC dispatch schedule ready for submission.")
    print("  SUBMISSION CHECKLIST:")
    print("  [x] 15-minute resolution (96 slots/day)")
    print("  [x] 2-day ahead horizon (ABT regulation compliant)")
    print("  [x] Q0.667 quantile model (penalty-optimal under ABT)")
    print("  [x] Peak-hour buffer Q0.75 applied (18:00–21:00)")
    print("  [x] P10–P90 uncertainty bands included")
    print("  [x] CSV in SLDC date/slot format")
    print(f"\n  Submit file: {csv_path}")

    return sldc


if __name__ == "__main__":
    run()
