"""
step13_stage2.py — Stage 2: Regime Shift & Penalty Escalation
==============================================================
Case GRIDSHIELD | DecodeX 2026 | Stage 2 (effective 28 Feb 2026, 7:00 PM)

DATA NOTE
─────────
Only one load file exists: Electric_Load_Data_Train.csv (Apr 2013 – Apr 2021)
Test set = rows where DateTime >= 2021-01-01 (Jan–Apr 2021, 11,520 slots)
Features loaded from features.parquet with forced datetime64 dtype.

REGULATORY UPDATE
─────────────────
Peak hours (18:00–21:59): under-forecast ₹4 → ₹6 per kWh
Off-peak unchanged. Over-forecast unchanged at ₹2/kWh.
  τ*_peak   = 6/(6+2) = 0.750
  τ*_offpeak = 4/(4+2) = 0.667
"""

import sys, os, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# Force UTF-8 output on Windows to avoid encoding errors with ₹/★/subscript chars
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import (
    compute_penalty, compute_penalty_s2,
    set_plot_style, save_plot,
    PLOTS_DIR, MODELS_DIR,
    OPTIMAL_QUANTILE_PEAK_S2, OPTIMAL_QUANTILE_OFFPEAK_S2,
)
from step02_feature_engineering import ALL_FEATURES

# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR      = r"c:\hackathons\DecodeX"
FEATURES_PATH = os.path.join(BASE_DIR, "outputs", "features.parquet")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TEST_START          = "2021-01-01"
EXPECTED_TEST_ROWS  = 11520   # 120 days × 96 slots


# ──────────────────────────────────────────────────────────────────────────────
# DATETIME VALIDATION GUARD
# ──────────────────────────────────────────────────────────────────────────────
def validate_datetime(df: pd.DataFrame, name: str = "DataFrame"):
    """Raise immediately if datetime parsing failed silently."""
    n_dates = df["DateTime"].dt.date.nunique()
    min_dt  = df["DateTime"].min()
    max_dt  = df["DateTime"].max()
    n_null  = df["DateTime"].isna().sum()

    print(f"\n  {name} DateTime check:")
    print(f"    dtype        : {df['DateTime'].dtype}")
    print(f"    min          : {min_dt}")
    print(f"    max          : {max_dt}")
    print(f"    null count   : {n_null}")
    print(f"    unique dates : {n_dates}")

    if n_null > 0:
        raise ValueError(f"[{name}] {n_null} null datetimes — parsing failed!")
    if n_dates < 10:
        raise ValueError(f"[{name}] Only {n_dates} unique dates — datetime parsing failed!")
    if min_dt.year < 2021:
        raise ValueError(f"[{name}] Test starts {min_dt.year}, expected 2021")
    if max_dt.month != 4 or max_dt.year != 2021:
        raise ValueError(f"[{name}] Test ends {max_dt}, expected Apr 2021")


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD TEST SET FROM features.parquet
# ──────────────────────────────────────────────────────────────────────────────
def load_test_data() -> pd.DataFrame:
    """
    Load features.parquet, FORCE DateTime to datetime64[ns], split on 2021-01-01.
    Returns: test_df (Jan–Apr 2021, 11,520 rows)
    """
    print("  Loading features.parquet …")
    df_all = pd.read_parquet(FEATURES_PATH)

    # CRITICAL FIX: force DateTime to proper Timestamp dtype
    df_all["DateTime"] = pd.to_datetime(df_all["DateTime"])
    print(f"  DateTime dtype after cast: {df_all['DateTime'].dtype}")

    assert pd.api.types.is_datetime64_any_dtype(df_all["DateTime"]), \
        f"DateTime is still {df_all['DateTime'].dtype} — conversion failed"

    df_test = df_all[df_all["DateTime"] >= TEST_START].copy().reset_index(drop=True)

    # ── Verification ──────────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  === VERIFICATION ===")
    print(f"  Test rows  : {len(df_test):,}   (expect {EXPECTED_TEST_ROWS:,})")
    print(f"  Test start : {df_test['DateTime'].min()}")
    print(f"  Test end   : {df_test['DateTime'].max()}")
    print(f"  Test mean load : {df_test['LOAD'].mean():.1f} kW  (expect ~1,151 kW)")
    print(f"  Unique dates   : {df_test['DateTime'].dt.date.nunique()}  (expect 120)")
    print(f"  {'='*60}")

    assert len(df_test) == EXPECTED_TEST_ROWS, \
        f"Expected {EXPECTED_TEST_ROWS} test rows, got {len(df_test)}"

    validate_datetime(df_test, "Test Set (Jan–Apr 2021)")
    return df_test


# ──────────────────────────────────────────────────────────────────────────────
# 2. ROLLING BIAS ESTIMATE FROM PRE-COVID WINDOW
# ──────────────────────────────────────────────────────────────────────────────
def compute_rolling_bias_estimate() -> float:
    """
    Estimate model over-forecast bias using Jan–Feb 2020 (pre-COVID, clean window).
    bias = mean(actual − forecast)  →  negative = model over-forecasts
    Corrected forecast = raw_forecast + bias  (subtracts the over-forecast)
    """
    df_all = pd.read_parquet(FEATURES_PATH)
    df_all["DateTime"] = pd.to_datetime(df_all["DateTime"])

    pre_covid = df_all[
        (df_all["DateTime"] >= "2020-01-01") &
        (df_all["DateTime"] <= "2020-02-28")
    ].copy()

    if len(pre_covid) == 0:
        print("  WARNING: No pre-COVID data found — bias correction = 0")
        return 0.0

    mse_path = os.path.join(MODELS_DIR, "lgbm_mse.pkl")
    if not os.path.exists(mse_path):
        print("  WARNING: lgbm_mse.pkl missing — bias correction = 0")
        return 0.0

    with open(mse_path, "rb") as f:
        mse_model = pickle.load(f)

    feats       = [f for f in ALL_FEATURES if f in pre_covid.columns]
    raw_fcst    = mse_model.predict(pre_covid[feats].values)
    bias        = float((pre_covid["LOAD"].values - raw_fcst).mean())

    print(f"  Bias estimate (Jan–Feb 2020, pre-COVID): {bias:+.1f} kW")
    print(f"  Model {'over' if bias < 0 else 'under'}-forecasts by {abs(bias):.1f} kW on average")
    return bias


# ──────────────────────────────────────────────────────────────────────────────
# 3. LOAD MODELS
# ──────────────────────────────────────────────────────────────────────────────
def load_models_s2() -> dict:
    models = {}
    for name in ["lgbm_q667", "lgbm_q75", "lgbm_q10", "lgbm_q90", "lgbm_mse"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            print(f"  ✔ Loaded {name}")
        else:
            print(f"  ✗ MISSING {name}")
    return models


# ──────────────────────────────────────────────────────────────────────────────
# 4. GENERATE FORECASTS (raw + bias-corrected)
# ──────────────────────────────────────────────────────────────────────────────
def generate_forecasts(test_df: pd.DataFrame,
                        models: dict,
                        bias_correction: float) -> pd.DataFrame:
    """
    Generate raw and bias-corrected forecasts.
    bias_correction: scalar kW (typically negative = model over-forecasts)
    Corrected forecast = raw + bias_correction  (pulls down over-estimating model)
    """
    feats   = [f for f in ALL_FEATURES if f in test_df.columns]
    X       = test_df[feats].values
    is_peak = test_df["is_peak_hour"].values.astype(bool)

    # ── Raw model outputs ─────────────────────────────────────────────────────
    q667_raw = np.maximum(models["lgbm_q667"].predict(X), 0)
    q75_raw  = np.maximum(models["lgbm_q75"].predict(X),  0)
    mse_raw  = np.maximum(models["lgbm_mse"].predict(X),  0)
    p10      = np.maximum(models["lgbm_q10"].predict(X),  0) if "lgbm_q10" in models else q667_raw * 0.85
    p90      = np.maximum(models["lgbm_q90"].predict(X),  0) if "lgbm_q90" in models else q667_raw * 1.15

    # ── Bias-corrected versions ───────────────────────────────────────────────
    q667_corr = np.maximum(q667_raw + bias_correction, 0)
    q75_corr  = np.maximum(q75_raw  + bias_correction, 0)
    mse_corr  = np.maximum(mse_raw  + bias_correction, 0)

    # Approximate Q0.60 by downshifting Q0.667 slightly (no separate model needed)
    q60_corr = np.maximum(q667_raw * 0.993 + bias_correction, 0)

    # ── Hybrid forecasts ──────────────────────────────────────────────────────
    # Raw hybrid (Q0.75 at peak, Q0.667 off-peak)
    hybrid_raw             = q667_raw.copy()
    hybrid_raw[is_peak]    = q75_raw[is_peak]

    # Corrected hybrid — THE RECOMMENDED Stage 2 strategy
    hybrid_corr            = q667_corr.copy()
    hybrid_corr[is_peak]   = q75_corr[is_peak]

    # ── Diagnostics ───────────────────────────────────────────────────────────
    actual_mean = test_df["LOAD"].mean()
    print(f"  Bias correction applied      : {bias_correction:+.1f} kW")
    print(f"  Raw Q0.667 mean forecast     : {q667_raw.mean():.1f} kW")
    print(f"  Corrected Q0.667 mean        : {q667_corr.mean():.1f} kW")
    print(f"  Actual mean load             : {actual_mean:.1f} kW")
    print(f"  Peak slots (Q0.75 applied)   : {is_peak.sum():,}")
    print(f"  Off-peak slots (Q0.667)      : {(~is_peak).sum():,}")

    out = test_df[["DateTime", "LOAD", "is_peak_hour", "lag_672"]].copy()
    out["fcst_naive"]       = np.maximum(out["lag_672"], 0)
    out["fcst_mse_raw"]     = mse_raw
    out["fcst_q667_raw"]    = q667_raw
    out["fcst_hybrid_raw"]  = hybrid_raw
    out["fcst_mse_corr"]    = mse_corr
    out["fcst_q60_corr"]    = q60_corr
    out["fcst_hybrid_corr"] = hybrid_corr   # ← BEST MODEL
    out["p10"]              = p10
    out["p90"]              = p90
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5. COMPUTE ALL PENALTIES
# ──────────────────────────────────────────────────────────────────────────────
def compute_all_penalties(out: pd.DataFrame) -> pd.DataFrame:
    actual  = out["LOAD"].values
    is_peak = out["is_peak_hour"].values

    strategies = {
        "Naive (lag₆₇₂)"               : out["fcst_naive"].values,
        "LightGBM MSE (raw)"            : out["fcst_mse_raw"].values,
        "LightGBM Q0.667 (raw)"         : out["fcst_q667_raw"].values,
        "Hybrid Q0.75 (raw)"            : out["fcst_hybrid_raw"].values,
        "BiasCorr + MSE"                : out["fcst_mse_corr"].values,
        "BiasCorr + Q0.60 [OPTIMAL]"    : out["fcst_q60_corr"].values,
        "BiasCorr + Hybrid Q0.75 ★"     : out["fcst_hybrid_corr"].values,
    }

    rows = []
    for label, fcst in strategies.items():
        s1 = compute_penalty(actual, fcst, is_peak)
        s2 = compute_penalty_s2(actual, fcst, is_peak)
        rows.append({
            "Strategy"           : label,
            "S1 Total (Rs)"      : s1["total_penalty_INR"],
            "S1 Peak (Rs)"       : s1.get("peak_penalty_INR", np.nan),
            "S1 Off-Peak (Rs)"   : s1.get("offpeak_penalty_INR", np.nan),
            "S2 Total (Rs)"      : s2["total_penalty_INR"],
            "S2 Peak (Rs)"       : s2["peak_penalty_INR"],
            "S2 Off-Peak (Rs)"   : s2["offpeak_penalty_INR"],
            "Penalty Shock (Rs)" : s2["total_penalty_INR"] - s1["total_penalty_INR"],
            "Bias (%)"           : s2["forecast_bias_pct"],
            "RMSE (kW)"          : s2["rmse_kW"],
            "95th pct Dev (kW)"  : s2["p95_abs_dev_kW"],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 6. PRINT TABLE
# ──────────────────────────────────────────────────────────────────────────────
def print_tables(df_pen: pd.DataFrame, test_df: pd.DataFrame):
    test_period = (f"{test_df['DateTime'].min().strftime('%b %Y')}"
                   f"–{test_df['DateTime'].max().strftime('%b %Y')}")
    print()
    print("=" * 108)
    print(f"  TABLE C: Stage 2 Test Set Penalty | {test_period} | Peak: ₹4→₹6/kWh")
    print("=" * 108)
    hdr = (f"  {'Strategy':<36}"
           f"{'S1 Total':>13}{'S1 Peak':>12}"
           f"{'S2 Total':>13}{'S2 Peak':>12}"
           f"{'Shock':>11}  {'Bias%':>7}  {'RMSE':>7}")
    print(hdr)
    print("  " + "-" * 106)

    best_s2 = df_pen["S2 Total (Rs)"].min()
    for _, r in df_pen.iterrows():
        star = " ★" if r["S2 Total (Rs)"] == best_s2 else "  "
        print(f"  {r['Strategy'] + star:<36}"
              f"  {r['S1 Total (Rs)']:>11,.0f}"
              f"  {r['S1 Peak (Rs)']:>10,.0f}"
              f"  {r['S2 Total (Rs)']:>11,.0f}"
              f"  {r['S2 Peak (Rs)']:>10,.0f}"
              f"  {r['Penalty Shock (Rs)']:>9,.0f}"
              f"  {r['Bias (%)']:>7.2f}"
              f"  {r['RMSE (kW)']:>7.2f}")
    print("  " + "=" * 106)

    naive_s2  = df_pen.loc[df_pen["Strategy"].str.contains("Naive"), "S2 Total (Rs)"].values[0]
    best_val  = df_pen["S2 Total (Rs)"].min()
    best_name = df_pen.loc[df_pen["S2 Total (Rs)"] == best_val, "Strategy"].values[0]

    print(f"\n  Best strategy (Stage 2): {best_name} → Rs {best_val:,.0f}")
    print(f"  Saving vs Naive : Rs {naive_s2 - best_val:,.0f}  "
          f"({(naive_s2 - best_val) / naive_s2 * 100:.1f}%)")
    print(f"\n  INSIGHT: τ*_peak = 6/(6+2) = 0.750 → Q0.75 is the regulatory optimum.")
    print(f"  Bias correction further reduces over-forecast, minimising ₹2/kWh over-penalty.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 7. CHART 20 — penalty bar chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_penalty_comparison(df_pen: pd.DataFrame, test_df: pd.DataFrame):
    set_plot_style()
    test_period = (f"{test_df['DateTime'].min().strftime('%b %Y')}"
                   f"–{test_df['DateTime'].max().strftime('%b %Y')}")

    strategies = df_pen["Strategy"].tolist()
    x = np.arange(len(strategies))
    w = 0.35

    blues = ["#2E86AB"] * len(strategies)
    reds  = ["#E84855"] * len(strategies)
    best_i = int(df_pen["S2 Total (Rs)"].values.argmin())
    reds[best_i] = "#3BB273"

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    b1 = axes[0].bar(x - w/2, df_pen["S1 Total (Rs)"] / 1e3, w,
                     label="Stage 1 (₹4 peak)", color=blues, alpha=0.85)
    b2 = axes[0].bar(x + w/2, df_pen["S2 Total (Rs)"] / 1e3, w,
                     label="Stage 2 (₹6 peak)", color=reds, alpha=0.85)
    axes[0].bar_label(b1, fmt="%.0f K", padding=3, fontsize=7)
    axes[0].bar_label(b2, fmt="%.0f K", padding=3, fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, rotation=20, ha="right", fontsize=7.5)
    axes[0].set_title(f"Stage 1 vs Stage 2 Total Penalty\n(Test: {test_period})",
                      fontweight="bold")
    axes[0].set_ylabel("Total Penalty (₹ Thousands)")
    axes[0].legend(fontsize=9)

    shock = df_pen["Penalty Shock (Rs)"] / 1e3
    shock_colors = ["#F4A261" if s > 0 else "#3BB273" for s in shock]
    bars = axes[1].bar(strategies, shock, color=shock_colors, alpha=0.9)
    axes[1].bar_label(bars, fmt="%.0f K", padding=3, fontsize=7)
    axes[1].axhline(0, color="black", lw=1.5, linestyle="--")
    axes[1].set_xticks(range(len(strategies)))
    axes[1].set_xticklabels(strategies, rotation=20, ha="right", fontsize=7.5)
    axes[1].set_title("Penalty Shock: Stage 2 − Stage 1\n"
                      "(Additional ₹ from ₹6 peak escalation)",
                      fontweight="bold")
    axes[1].set_ylabel("Additional Penalty (₹ Thousands)")

    fig.suptitle(f"GRIDSHIELD Stage 2 — Regime Shift Impact | {test_period}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "20_stage2_penalty_comparison.png")


def plot_stage2_forecast(out: pd.DataFrame):
    import datetime as _dt
    import matplotlib.patches as mpatches
    set_plot_style()

    # Slice first 3 days (288 slots)
    window = out.iloc[:288].copy().reset_index(drop=True)
    window["DateTime"] = pd.to_datetime(window["DateTime"])

    n_dates = window["DateTime"].dt.date.nunique()
    print(f"  DateTime dtype : {window['DateTime'].dtype}")
    print(f"  Window         : {window['DateTime'].iloc[0]} to {window['DateTime'].iloc[-1]}")
    print(f"  Unique dates   : {n_dates}  (must be 3)")
    if n_dates < 3:
        raise ValueError(f"Window has only {n_dates} unique dates!")

    # Convert ALL timestamps to matplotlib float numbers.
    # This avoids any pandas RangeIndex issues with fill_between/axvspan.
    pydts  = window["DateTime"].dt.to_pydatetime()   # Python datetime objects
    x_num  = mdates.date2num(pydts)                  # float array — safe for all mpl calls

    d0 = window["DateTime"].iloc[0].strftime("%d-%b-%Y")
    d1 = window["DateTime"].iloc[-1].strftime("%d-%b-%Y")

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.xaxis_date()   # tell matplotlib the x-axis is date-space

    # ── Lines (float x, float y) ─────────────────────────────────────────────
    ax.plot(x_num, window["LOAD"].to_numpy(),
            color="black", lw=2, label="Actual Load", zorder=10)
    ax.plot(x_num, window["fcst_naive"].to_numpy(),
            color="#AAAAAA", lw=1.5, linestyle="--", label="Naive (lag-672)", alpha=0.7)
    ax.plot(x_num, window["fcst_hybrid_raw"].to_numpy(),
            color="#E84855", lw=1.5, linestyle=":",
            label="Hybrid Q0.75 (raw)", alpha=0.85)
    ax.plot(x_num, window["fcst_hybrid_corr"].to_numpy(),
            color="#3BB273", lw=2.5, zorder=9,
            label="BiasCorr + Hybrid Q0.75 (RECOMMENDED)")

    # ── P10-P90 band ─────────────────────────────────────────────────────────
    ax.fill_between(x_num, window["p10"].to_numpy(), window["p90"].to_numpy(),
                    alpha=0.15, color="#F4A261", label="P10-P90 interval")

    # ── Peak shading (all floats) ─────────────────────────────────────────────
    for day in sorted(window["DateTime"].dt.date.unique()):
        ps_num = mdates.date2num(_dt.datetime.combine(day, _dt.time(18, 0)))
        pe_num = mdates.date2num(_dt.datetime.combine(day, _dt.time(22, 0)))
        ax.axvspan(ps_num, pe_num, alpha=0.10, color="#E84855", zorder=0)
    peak_patch = mpatches.Patch(facecolor="#E84855", alpha=0.20,
                                label="Peak hours 18-22h (Rs 6/kWh zone)")

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=18))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    ax.set_title(f"Stage 2 Test Set - First 3 Days ({d0} to {d1})\n"
                 "Green: BiasCorr+Hybrid Q0.75 (tau*-optimal) | Red dotted: raw over-forecast",
                 fontweight="bold")
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Load (kW)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [peak_patch], labels + [peak_patch.get_label()],
              fontsize=9, ncol=3, loc="upper left")
    fig.tight_layout()
    save_plot(fig, "21_stage2_forecast.png")
    print("  Chart 21 saved OK")






# ──────────────────────────────────────────────────────────────────────────────
# 9. WRITE stage_2.md
# ──────────────────────────────────────────────────────────────────────────────
def write_stage2_md(df_pen: pd.DataFrame, out: pd.DataFrame, bias: float):
    actual  = out["LOAD"].values
    is_peak = out["is_peak_hour"].values
    best_corr = out["fcst_hybrid_corr"].values
    naive     = out["fcst_naive"].values

    s1_naive  = compute_penalty(actual, naive,      is_peak)
    s2_naive  = compute_penalty_s2(actual, naive,   is_peak)
    s2_best   = compute_penalty_s2(actual, best_corr, is_peak)

    shock_abs  = s2_naive["total_penalty_INR"] - s1_naive["total_penalty_INR"]
    shock_pct  = shock_abs / s1_naive["total_penalty_INR"] * 100
    saving     = s2_naive["total_penalty_INR"] - s2_best["total_penalty_INR"]
    saving_pct = saving / s2_naive["total_penalty_INR"] * 100

    test_start = pd.to_datetime(out["DateTime"].min()).strftime("%B %Y")
    test_end   = pd.to_datetime(out["DateTime"].max()).strftime("%B %Y")
    test_range = f"{test_start}–{test_end}"

    md = f"""# Stage 2 — Regime Shift & Penalty Escalation
### Case GRIDSHIELD | DecodeX 2026 | Effective 28 February 2026, 7:00 PM
### Team: NLD Synapse | N. L. Dalmia Institute of Management Studies & Research

---

## 1. Regulatory Update

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Peak under-forecast penalty | ₹ 4 / kWh | **₹ 6 / kWh** |
| Off-peak under-forecast penalty | ₹ 4 / kWh | ₹ 4 / kWh (unchanged) |
| Over-forecast penalty | ₹ 2 / kWh | ₹ 2 / kWh (unchanged) |
| Test period | Training history | **{test_range} (11,520 slots)** |

---

## 2. Recalibrated Optimal Quantiles

```
τ*_peak   = 6 / (6 + 2) = 0.750  ← peak hours   (Q0.75 is now exact τ*)
τ*_offpk  = 4 / (4 + 2) = 0.667  ← off-peak     (unchanged)
```

**Rolling bias correction:** {bias:+.1f} kW estimated on Jan–Feb 2020 (pre-COVID).
Applied to all corrected models: corrected = raw + ({bias:+.1f}) kW.

---

## 3. Test Set Results ({test_range})

| Strategy | S1 Total (₹) | S2 Total (₹) | Shock (₹) | S2 Peak (₹) | Bias (%) |
|---|---|---|---|---|---|
"""
    for _, r in df_pen.iterrows():
        star = " ★" if r["S2 Total (Rs)"] == df_pen["S2 Total (Rs)"].min() else ""
        md += (f"| {r['Strategy']}{star} "
               f"| {r['S1 Total (Rs)']:,.0f} "
               f"| {r['S2 Total (Rs)']:,.0f} "
               f"| {r['Penalty Shock (Rs)']:+,.0f} "
               f"| {r['S2 Peak (Rs)']:,.0f} "
               f"| {r['Bias (%)']:+.2f}% |\n")

    md += f"""
### Penalty Shock (Naive Baseline)

| | Amount |
|---|---|
| Naive Stage 1 (₹4 peak) | ₹ {s1_naive["total_penalty_INR"]:,.0f} |
| Naive Stage 2 (₹6 peak) | ₹ {s2_naive["total_penalty_INR"]:,.0f} |
| **Shock from escalation** | **₹ {shock_abs:,.0f} (+{shock_pct:.1f}%)** |

### Best Strategy: BiasCorr + Hybrid Q0.75

| Metric | Value |
|---|---|
| Stage 2 Total Penalty | ₹ {s2_best["total_penalty_INR"]:,.0f} |
| Peak Penalty  | ₹ {s2_best["peak_penalty_INR"]:,.0f} |
| Off-Peak Penalty | ₹ {s2_best["offpeak_penalty_INR"]:,.0f} |
| **Saving vs Naive (Stage 2)** | **₹ {saving:,.0f} ({saving_pct:.1f}%)** |
| Forecast Bias | {s2_best["forecast_bias_pct"]:+.2f}% |

---

## 4. Strategy Recalibration

| | Stage 1 | Stage 2 |
|---|---|---|
| Off-peak model | Q0.667 (τ*) | Q0.667 (τ*, unchanged) |
| Peak model | Q0.75 (buffer) | **Q0.75 (τ*-derived, mandatory)** |
| Bias correction | None | −{abs(bias):.1f} kW (pre-COVID window) |
| No retraining needed | ✓ | ✓ |

---

*Stage 2 submission | 28 February 2026 | GRIDSHIELD v2.0*
"""
    md_path = os.path.join(BASE_DIR, "stage_2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  ✔ stage_2.md written → {md_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def run():
    print()
    print("=" * 70)
    print("  GRIDSHIELD — STAGE 2: Regime Shift & Penalty Escalation")
    print("  Test: Jan–Apr 2021 | Peak ₹4→₹6 | τ*_peak=0.75 | bias-corrected")
    print("=" * 70)

    # 1. Load test data (datetime64 verified)
    print("\n  [1/6] Loading test set …")
    test_df = load_test_data()

    # 2. Compute rolling bias correction
    print("\n  [2/6] Computing rolling bias estimate (Jan–Feb 2020) …")
    bias = compute_rolling_bias_estimate()

    # 3. Load models
    print("\n  [3/6] Loading trained models …")
    models = load_models_s2()
    if "lgbm_q667" not in models:
        print("  ERROR: lgbm_q667 not found — run main pipeline first.")
        return

    # 4. Generate forecasts (raw + corrected)
    print("\n  [4/6] Generating forecasts …")
    out = generate_forecasts(test_df, models, bias_correction=bias)

    # ── Verification checks ───────────────────────────────────────────────────
    print(f"\n  === VERIFICATION CHECKS ===")
    print(f"  DateTime dtype     : {out['DateTime'].dtype}")
    window_check = out.iloc[:288]
    print(f"  Window unique dates: {window_check['DateTime'].dt.date.nunique()}  (must be 3)")
    print(f"  Raw MSE mean       : {out['fcst_mse_raw'].mean():.1f} kW")
    print(f"  Corr MSE mean      : {out['fcst_mse_corr'].mean():.1f} kW")
    print(f"  Actual mean        : {out['LOAD'].mean():.1f} kW")

    # 5. Penalties
    print("\n  [5/6] Computing penalties …")
    df_pen = compute_all_penalties(out)
    print_tables(df_pen, test_df)

    # 6. Charts + outputs
    print("\n  [6/6] Generating charts and outputs …")
    plot_penalty_comparison(df_pen, test_df)
    plot_stage2_forecast(out)

    csv_path = os.path.join(OUTPUTS_DIR, "stage2_results.csv")
    df_pen.to_csv(csv_path, index=False)
    print(f"  ✔ Penalty table → {csv_path}")

    fcast_path = os.path.join(OUTPUTS_DIR, "stage2_forecasts.csv")
    out[["DateTime", "LOAD", "is_peak_hour",
         "fcst_naive", "fcst_mse_raw", "fcst_q667_raw",
         "fcst_hybrid_raw", "fcst_mse_corr",
         "fcst_q60_corr", "fcst_hybrid_corr",
         "p10", "p90"]].to_csv(fcast_path, index=False)
    print(f"  ✔ Forecasts     → {fcast_path}")

    write_stage2_md(df_pen, out, bias)

    print()
    print("=" * 70)
    print("  STAGE 2 COMPLETE")
    print(f"  Chart 20 : outputs/plots/20_stage2_penalty_comparison.png")
    print(f"  Chart 21 : outputs/plots/21_stage2_forecast.png")
    print("=" * 70)
    return df_pen, out


if __name__ == "__main__":
    run()
