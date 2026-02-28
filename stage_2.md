# Stage 2 — Regime Shift & Penalty Escalation
### Case GRIDSHIELD | DecodeX 2026 | Effective 28 February 2026, 7:00 PM
### Team: NLD Synapse | N. L. Dalmia Institute of Management Studies & Research

---

## 1. Regulatory Update

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Peak under-forecast penalty | ₹ 4 / kWh | **₹ 6 / kWh** |
| Off-peak under-forecast penalty | ₹ 4 / kWh | ₹ 4 / kWh (unchanged) |
| Over-forecast penalty | ₹ 2 / kWh | ₹ 2 / kWh (unchanged) |
| Test period | Training history | **January 2021–April 2021 (11,520 slots)** |

---

## 2. Recalibrated Optimal Quantiles

```
τ*_peak   = 6 / (6 + 2) = 0.750  ← peak hours   (Q0.75 is now exact τ*)
τ*_offpk  = 4 / (4 + 2) = 0.667  ← off-peak     (unchanged)
```

**Rolling bias correction:** -26.9 kW estimated on Jan–Feb 2020 (pre-COVID).
Applied to all corrected models: corrected = raw + (-26.9) kW.

---

## 3. Test Set Results (January 2021–April 2021)

| Strategy | S1 Total (₹) | S2 Total (₹) | Shock (₹) | S2 Peak (₹) | Bias (%) |
|---|---|---|---|---|---|
| Naive (lag₆₇₂) | 608,010 | 640,246 | +32,236 | 118,032 | -1.77% |
| LightGBM MSE (raw) | 372,559 | 375,678 | +3,120 | 69,941 | +4.63% |
| LightGBM Q0.667 (raw) | 382,900 | 383,808 | +908 | 68,288 | +5.25% |
| Hybrid Q0.75 (raw) | 390,497 | 391,220 | +723 | 75,699 | +5.37% |
| BiasCorr + MSE | 315,683 | 323,062 | +7,379 | 61,152 | +2.30% |
| BiasCorr + Q0.60 [OPTIMAL] ★ | 292,186 | 298,317 | +6,131 | 54,560 | +2.17% |
| BiasCorr + Hybrid Q0.75 ★ | 306,033 | 308,731 | +2,698 | 57,770 | +3.03% |

### Penalty Shock (Naive Baseline)

| | Amount |
|---|---|
| Naive Stage 1 (₹4 peak) | ₹ 608,010 |
| Naive Stage 2 (₹6 peak) | ₹ 640,246 |
| **Shock from escalation** | **₹ 32,236 (+5.3%)** |

### Best Strategy: BiasCorr + Hybrid Q0.75

| Metric | Value |
|---|---|
| Stage 2 Total Penalty | ₹ 308,731 |
| Peak Penalty  | ₹ 57,770 |
| Off-Peak Penalty | ₹ 250,962 |
| **Saving vs Naive (Stage 2)** | **₹ 331,515 (51.8%)** |
| Forecast Bias | +3.03% |

---

## 4. Strategy Recalibration

| | Stage 1 | Stage 2 |
|---|---|---|
| Off-peak model | Q0.667 (τ*) | Q0.667 (τ*, unchanged) |
| Peak model | Q0.75 (buffer) | **Q0.75 (τ*-derived, mandatory)** |
| Bias correction | None | −26.9 kW (pre-COVID window) |
| No retraining needed | ✓ | ✓ |

---

*Stage 2 submission | 28 February 2026 | GRIDSHIELD v2.0*
