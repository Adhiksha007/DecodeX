"""
plot_step12_vs_actual.py
========================
Plots the Step 12 SLDC submission forecast (May 1-2, 2021)
against the actual ground truth load from the Stage 2 test dataset.
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, r"c:\hackathons\DecodeX\gridshield")
from utils import set_plot_style, save_plot

def run():
    print("Loading Step 12 Forecast (SLDC Submission)...")
    fcst_path = r"c:\hackathons\DecodeX\outputs\SLDC_Forecast_2021-05-01_to_2021-05-02.csv"
    if not os.path.exists(fcst_path):
        print(f"Error: Could not find {fcst_path}. Run step12_sldc_submission.py first.")
        return
        
    fcst_df = pd.read_csv(fcst_path)
    # Reconstruct DateTime
    fcst_df['DateTime'] = pd.to_datetime(fcst_df['Date'] + ' ' + fcst_df['TimeSlot'])
    
    print("Loading Actual Test Data...")
    test_path = r"c:\hackathons\DecodeX\Electric_Load_Data_Test.csv"
    test_raw = pd.read_csv(test_path)
    test_raw['DateTime'] = pd.to_datetime(test_raw['DATETIME'], format='%d%b%Y:%H:%M:%S', errors='coerce')
    
    # Filter test data to just May 1-2
    test_2days = test_raw[(test_raw['DateTime'] >= '2021-05-01 00:00:00') & 
                          (test_raw['DateTime'] <= '2021-05-02 23:45:00')].copy()
                          
    if test_2days.empty:
        print("Error: Could not find May 1-2 in test data.")
        return
        
    print(f"Loaded {len(test_2days)} actual records.")
    
    # Merge for plotting
    plot_df = pd.merge(fcst_df, test_2days[['DateTime', 'LOAD']], on='DateTime', how='inner')
    
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Panel 1: Full overlay with bands
    ax1 = axes[0]
    ax1.plot(plot_df['DateTime'], plot_df['LOAD'], color='#2E86AB', lw=2, label='Actual Load (Test Data)', zorder=5)
    ax1.plot(plot_df['DateTime'], plot_df['Forecast_kW'], color='#E84855', lw=2, label='Step 12 Forecast (Q0.667 + Q0.75 Peak Buffer)', ls='--', zorder=6)
    
    ax1.fill_between(plot_df['DateTime'], plot_df['P10_kW'], plot_df['P90_kW'], 
                     color='#F4A261', alpha=0.2, label='P10–P90 Uncertainty Band')
    
    # Highlight peak hours
    for dt in plot_df[plot_df['IsPeakHour'] == 1]['DateTime'].dt.date.unique():
        start = pd.Timestamp(dt) + pd.Timedelta(hours=18)
        end   = pd.Timestamp(dt) + pd.Timedelta(hours=21)
        ax1.axvspan(start, end, alpha=0.1, color='#FFD700', label='Peak Hours (18-21h)' if start == plot_df['DateTime'].min().normalize() + pd.Timedelta(hours=18) else "")
        
    ax1.set_title('Step 12 SLDC Forecast vs Actual Ground Truth — May 1-2, 2021', fontweight='bold')
    ax1.set_ylabel('Load (kW)')
    ax1.legend(loc='upper right')
    
    # Panel 2: Residuals and ABT Penalty visualization
    ax2 = axes[1]
    resids = plot_df['LOAD'] - plot_df['Forecast_kW']
    
    # Red = Underforecast (Rs 4 penalty), Blue = Overforecast (Rs 2 penalty)
    colors = ['#E84855' if r > 0 else '#2E86AB' for r in resids]
    
    ax2.bar(plot_df['DateTime'], resids, width=0.01, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', lw=1.2)
    
    # Calc penalties
    under_pen = resids[resids > 0].sum() * 4
    over_pen  = abs(resids[resids < 0].sum()) * 2
    total_pen = under_pen + over_pen
    
    ax2.text(0.01, 0.95, f"2-Day ABT Penalty (Estimated):\nTotal: Rs {total_pen:,.0f}\nUnder-forecast (Red): Rs {under_pen:,.0f}\nOver-forecast (Blue): Rs {over_pen:,.0f}", 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    ax2.set_title('Residuals (Actual - Forecast) — Red bars show severe ₹4/kWh exposure risk', fontweight='bold')
    ax2.set_ylabel('Error (kW)')
    
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    
    fig.autofmt_xdate()
    fig.tight_layout()
    
    save_plot(fig, '24_step12_vs_actual.png')
    print(f"Done. Plot saved with Total Penalty: Rs {total_pen:,.0f}")

if __name__ == '__main__':
    run()
