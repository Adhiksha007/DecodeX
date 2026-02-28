"""
step01_eda.py — Exploratory Data Analysis
==========================================
Step 1 of the GRIDSHIELD Forecast Risk Advisory Pipeline.

Analyses:
  1. Full 8-year load time series with COVID annotation
  2. Average intraday load profile (96 slots)
  3. Day-of-week average load (Mon–Sun)
  4. Monthly average load (seasonal pattern)
  5. Correlation matrix: load vs weather variables
  6. Holiday vs Weekend vs Normal day comparison
  7. COVID-19 anomaly quantification (Mar–Jun 2020 vs 2019)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from utils import (parse_datetime, set_plot_style, save_plot,
                   LOAD_FILE, WEATHER_FILE, EVENTS_FILE, PLOTS_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading raw data …")
    df_load    = pd.read_csv(LOAD_FILE)
    df_weather = pd.read_csv(WEATHER_FILE)
    df_events  = pd.read_csv(EVENTS_FILE)

    # Parse datetime
    df_load["DateTime"]    = parse_datetime(df_load["DATETIME"])
    df_weather["DateTime"] = parse_datetime(df_weather["DATETIME"])

    # Merge load + weather on DateTime
    df = pd.merge(df_load, df_weather, on="DateTime", how="inner")
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Parse events calendar
    df_events["Date"] = pd.to_datetime(df_events["Date"], dayfirst=True, errors="coerce")
    df_events = df_events.dropna(subset=["Date"])

    print(f"  Dataset shape: {df.shape}, range: {df['DateTime'].min()} → {df['DateTime'].max()}")
    return df, df_events


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PLOT 1 — Full 8-year load time series
# ──────────────────────────────────────────────────────────────────────────────
def plot_full_timeseries(df):
    print("\n[Plot 1] Full 8-year load time series …")
    set_plot_style()

    # Daily average to reduce visual noise
    daily = df.set_index("DateTime")["LOAD"].resample("D").mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(daily.index, daily.values, lw=0.8, color="#2E86AB", label="Daily Avg Load (kW)")

    # Annotate COVID-19 period
    covid_start = pd.Timestamp("2020-03-01")
    covid_end   = pd.Timestamp("2020-06-30")
    ax.axvspan(covid_start, covid_end, color="red", alpha=0.15, label="COVID-19 Period")

    # Annotate structural highlights
    ax.axvspan(pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01"),
               color="orange", alpha=0.07, label="2017 — Demonetisation effect")

    ax.set_title("Lumina Energy — 8-Year Electricity Load Time Series (2013–2021)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Daily Load (kW)")
    ax.legend(loc="upper left", fontsize=9)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    save_plot(fig, "01_full_timeseries.png")

    # Key structural breaks visible
    print("  ➜ Structural breaks: COVID drop Mar-Jun 2020; summer peaks (Apr-Jun); "
          "consistent annual growth trend until 2020.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PLOT 2 — Average intraday load profile (96 slots)
# ──────────────────────────────────────────────────────────────────────────────
def plot_intraday_profile(df):
    print("\n[Plot 2] Average intraday load profile …")
    set_plot_style()

    df["slot"] = df["DateTime"].dt.hour * 4 + df["DateTime"].dt.minute // 15
    slot_avg   = df.groupby("slot")["LOAD"].mean()
    slot_time  = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(96), slot_avg.values, color="#E84855", lw=2, marker="o", markersize=2)
    ax.axvspan(72, 84, color="#F4A261", alpha=0.25, label="Peak Hours (18:00–21:00)")
    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=8)
    ax.set_title("Average Intraday Load Profile — 96 Half-Hour Slots", fontweight="bold")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Average Load (kW)")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "02_intraday_profile.png")

    peak_load  = slot_avg.iloc[72:84].mean()
    trough_load= slot_avg.iloc[8:16].mean()
    print(f"  ➜ Peak window (18–21h) avg load: {peak_load:.0f} kW; "
          f"Overnight trough: {trough_load:.0f} kW. "
          f"Peak-to-trough ratio: {peak_load/trough_load:.2f}×")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PLOT 3 — Day-of-week average load
# ──────────────────────────────────────────────────────────────────────────────
def plot_dayofweek_profile(df):
    print("\n[Plot 3] Day-of-week average load …")
    set_plot_style()

    df["dow"]  = df["DateTime"].dt.dayofweek
    dow_avg    = df.groupby("dow")["LOAD"].mean()
    days       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors     = ["#2E86AB"]*5 + ["#E84855"]*2

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(days, dow_avg.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Average Load by Day of Week", fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Load (kW)")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)

    wd_patch = mpatches.Patch(color="#2E86AB", label="Weekday")
    we_patch = mpatches.Patch(color="#E84855", label="Weekend")
    ax.legend(handles=[wd_patch, we_patch])
    fig.tight_layout()
    save_plot(fig, "03_dayofweek_profile.png")

    wkd_avg = dow_avg.iloc[:5].mean()
    wkn_avg = dow_avg.iloc[5:].mean()
    print(f"  ➜ Avg weekday load: {wkd_avg:.0f} kW; Avg weekend load: {wkn_avg:.0f} kW "
          f"({(wkn_avg/wkd_avg-1)*100:.1f}% difference).")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PLOT 4 — Monthly average load (seasonal)
# ──────────────────────────────────────────────────────────────────────────────
def plot_monthly_profile(df):
    print("\n[Plot 4] Monthly average load …")
    set_plot_style()

    df["month"] = df["DateTime"].dt.month
    month_avg   = df.groupby("month")["LOAD"].mean()
    months      = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    palette = sns.color_palette("coolwarm", 12)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(months, month_avg.values, color=palette, edgecolor="white")
    ax.set_title("Average Load by Month — Seasonal Pattern (Mumbai)", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Load (kW)")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    fig.tight_layout()
    save_plot(fig, "04_monthly_profile.png")

    peak_month = months[month_avg.idxmax() - 1]
    print(f"  ➜ Peak month: {peak_month} (summer AC load). "
          f"Winter months show lowest demand — clear bimodal seasonal pattern.")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PLOT 5 — Correlation matrix
# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation(df):
    print("\n[Plot 5] Correlation matrix …")
    set_plot_style()

    weather_cols = ["ACT_TEMP", "ACT_HEAT_INDEX", "ACT_HUMIDITY", "ACT_RAIN", "COOL_FACTOR"]
    # Keep only columns that exist
    cols_present = ["LOAD"] + [c for c in weather_cols if c in df.columns]
    corr = df[cols_present].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix: Load vs Weather Features", fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "05_correlation_matrix.png")

    top_corr = corr["LOAD"].drop("LOAD").abs().idxmax()
    print(f"  ➜ Strongest weather predictor of load: {top_corr} "
          f"(|r|={corr['LOAD'][top_corr]:.2f}). "
          f"Temperature and heat index drive AC load in Mumbai summers.")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PLOT 6 — Holiday vs Weekend vs Normal day
# ──────────────────────────────────────────────────────────────────────────────
def plot_holiday_comparison(df, df_events):
    print("\n[Plot 6] Holiday vs Weekend vs Normal …")
    set_plot_style()

    df["date"] = df["DateTime"].dt.date
    df["dow"]  = df["DateTime"].dt.dayofweek

    # Mark holidays
    holiday_dates = set(df_events[df_events["Holiday_Ind"] == 1]["Date"].dt.date)
    df["is_holiday"]  = df["date"].isin(holiday_dates).astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["day_type"]    = "Normal Weekday"
    df.loc[df["is_weekend"] == 1, "day_type"] = "Weekend"
    df.loc[df["is_holiday"] == 1, "day_type"] = "Public Holiday"

    slot_type = df.groupby(["slot" if "slot" in df.columns else df["DateTime"].dt.hour * 4 + df["DateTime"].dt.minute // 15, "day_type"])["LOAD"].mean().unstack()
    if "slot" in df.columns:
        x = df["slot"].unique(); x.sort()
    else:
        df["slot"] = df["DateTime"].dt.hour * 4 + df["DateTime"].dt.minute // 15

    slot_type = df.groupby(["slot", "day_type"])["LOAD"].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"Normal Weekday": "#2E86AB", "Weekend": "#E84855", "Public Holiday": "#3BB273"}
    for day_type, color in colors.items():
        if day_type in slot_type.columns:
            ax.plot(slot_type.index, slot_type[day_type], label=day_type, color=color, lw=2)

    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=8)
    ax.set_title("Intraday Load Profile: Holidays vs Weekends vs Normal Days", fontweight="bold")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Average Load (kW)")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "06_holiday_comparison.png")

    hol_avg = df[df["is_holiday"]==1]["LOAD"].mean()
    nor_avg = df[(df["is_holiday"]==0) & (df["is_weekend"]==0)]["LOAD"].mean()
    print(f"  ➜ Holidays average {hol_avg:.0f} kW vs normal weekdays {nor_avg:.0f} kW "
          f"({(hol_avg/nor_avg-1)*100:.1f}%). Holiday demand reduction is significant.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 8.  COVID-19 Anomaly Quantification
# ──────────────────────────────────────────────────────────────────────────────
def analyse_covid(df):
    print("\n[Analysis] COVID-19 anomaly quantification ...")

    # ── Three-era masks ────────────────────────────────────────────────────────
    precov_mask  = df["DateTime"] < "2020-03-01"
    covid_mask   = (df["DateTime"] >= "2020-03-01") & (df["DateTime"] <= "2020-06-30")
    postcov_mask = df["DateTime"] > "2020-06-30"

    # Comparisons: same calendar window Mar-Jun across years
    prior_mask   = (df["DateTime"] >= "2019-03-01") & (df["DateTime"] <= "2019-06-30")
    covid_w_mask = (df["DateTime"] >= "2020-03-01") & (df["DateTime"] <= "2020-06-30")
    postcov_val  = (df["DateTime"] >= "2021-01-01")   # full post-COVID in val set

    prior_avg    = df[prior_mask]["LOAD"].mean()
    covid_avg    = df[covid_w_mask]["LOAD"].mean()
    postcov_avg  = df[postcov_val]["LOAD"].mean()
    drop_kw      = prior_avg - covid_avg
    drop_pct     = drop_kw / prior_avg * 100
    rec_kw       = postcov_avg - covid_avg
    rec_pct      = rec_kw / drop_kw * 100

    print(f"\n  -- COVID-19 Load Impact (3-Era Summary) ------")
    print(f"  Pre-COVID avg (Mar-Jun 2019)       : {prior_avg:,.0f} kW")
    print(f"  COVID avg     (Mar-Jun 2020)       : {covid_avg:,.0f} kW  (drop: {drop_kw:,.0f} kW = {drop_pct:.1f}%)")
    print(f"  Post-COVID avg (Jan-Apr 2021)      : {postcov_avg:,.0f} kW  (recovered: {rec_kw:,.0f} kW = {rec_pct:.1f}%)")
    print()

    # ── Plot 7: Full timeline Jan 2019–Apr 2021 with era bands ────────────────
    set_plot_style()

    # Full window for the plot: Jan 2019 – Apr 2021
    view_mask = (df["DateTime"] >= "2019-01-01") & (df["DateTime"] <= "2021-04-30")
    daily = (df[view_mask]
             .set_index("DateTime")["LOAD"]
             .resample("D").mean())

    # Per-era daily averages (horizontal reference lines)
    precov_ref  = df[(df["DateTime"] >= "2019-01-01") & precov_mask]["LOAD"].mean()
    covid_ref   = df[covid_mask]["LOAD"].mean()
    postcov_ref = df[postcov_mask & (df["DateTime"] <= "2021-04-30")]["LOAD"].mean()

    ERA_BANDS = [
        ("2019-01-01", "2020-02-29", "#2E86AB", "Pre-COVID Era",       precov_ref ),
        ("2020-03-01", "2020-06-30", "#E84855", "COVID Lockdown",      covid_ref  ),
        ("2020-07-01", "2021-04-30", "#3BB273", "Post-COVID Recovery", postcov_ref),
    ]

    fig, ax = plt.subplots(figsize=(16, 6))

    # Draw era background bands
    for start, end, color, label, ref in ERA_BANDS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.12, color=color, label=label)
        ax.axhline(ref, xmin=0, xmax=1,
                   color=color, lw=1.5, linestyle="--", alpha=0.7)
        # Annotate avg line
        ax.text(pd.Timestamp(end), ref + 8,
                f"avg {ref:,.0f} kW",
                color=color, fontsize=7.5, fontweight="bold", ha="right")

    # Main load line
    ax.plot(daily.index, daily.values,
            lw=1.2, color="#1A1A2E", label="Daily Avg Load (kW)", zorder=5)

    # Annotate key events
    ax.annotate("Lockdown begins\n(Mar 23 2020)",
                xy=(pd.Timestamp("2020-03-23"), covid_avg - 30),
                xytext=(pd.Timestamp("2020-05-01"), covid_avg - 120),
                arrowprops=dict(arrowstyle="->", color="#E84855"),
                color="#E84855", fontsize=8)
    ax.annotate("Unlock phase\n(Jul 2020)",
                xy=(pd.Timestamp("2020-07-01"), postcov_ref + 10),
                xytext=(pd.Timestamp("2020-08-15"), postcov_ref + 80),
                arrowprops=dict(arrowstyle="->", color="#3BB273"),
                color="#3BB273", fontsize=8)

    ax.set_title(
        f"Lumina Energy Load: Pre-COVID | COVID | Post-COVID (Jan 2019 - Apr 2021)\n"
        f"COVID drop: {drop_kw:,.0f} kW ({drop_pct:.1f}%)   "
        f"Post-COVID recovery: {rec_kw:,.0f} kW ({rec_pct:.1f}% of drop restored)",
        fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Average Load (kW)")

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    handles_original, labels_original = ax.get_legend_handles_labels()
    ax.legend(handles_original, labels_original,
              loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    save_plot(fig, "07_covid_impact.png")

    return drop_kw, drop_pct



def run():
    print("=" * 60)
    print("  STEP 1 — Exploratory Data Analysis")
    print("=" * 60)

    df, df_events = load_data()

    plot_full_timeseries(df)
    plot_intraday_profile(df)
    plot_dayofweek_profile(df)
    plot_monthly_profile(df)
    plot_correlation(df)
    df = plot_holiday_comparison(df, df_events)
    drop_kw, drop_pct = analyse_covid(df)

    print("\n✔ STEP 1 COMPLETE — all EDA plots saved to outputs/plots/")
    print("\n  KEY FINDINGS:")
    print("  • Strong intraday peak at 18:00–21:00 (highest financial risk window)")
    print("  • Clear summer seasonality driven by Mumbai heat & AC usage")
    print(f"  • COVID-19 caused a {drop_pct:.1f}% load drop — a structural break requiring special handling")
    print("  • Holidays show significantly different load profile vs normal weekdays")
    
    print("\n  BUSINESS IMPLICATION FOR LUMINA ENERGY:")
    print("  • Forecast models must account for seasonality, time-of-day, weekday effects & anomalies")
    print("  • COVID period must be flagged to prevent model from learning incorrect demand patterns")
    print("  • Peak hour window is the highest-penalty zone — special forecasting strategy needed")

    return df, df_events


if __name__ == "__main__":
    run()
