# REGIME DETECTION ANALYSIS REPORT

**Generated:** 2025-11-12 13:44:58  
**Input File:** Nifty_50.csv  
**Analysis Period:** 2007-10-01 to 2025-10-23  
**Total Days:** 4714

---

## CURRENT MARKET STATE

**Date:** 2025-10-23  
**Price:** $25891.40  
**State Vector:** [1, 0]  
**Market Regime:** **Stable Bull**

**Components:**
- Trend (Sensor 1): BULL
- Volatility (Sensor 2): Low (0.66%)

---

## REGIME DISTRIBUTION

| Regime | Days | Percentage |
|--------|------|------------|
| Stable Bull | 3581 | 76.0% |
| Panic/Crash | 667 | 14.1% |
| Volatile Bull | 291 | 6.2% |
| Stable Bear | 175 | 3.7% |


---

## OUTPUT FILES

All results saved in: `results/test_run/`

### Cleaned Data
- `Nifty_50_cleaned.csv` - Cleaned input data

### Sensor 1: Trend (MRS)
- `sensor1_trend_predictions.csv` - BULL/BEAR predictions
- `sensor1_trend_summary.txt` - Model statistics
- `sensor1_trend_analysis.png` - Visualization

### Sensor 2: Volatility (GARCH)
- `sensor2_volatility_regime.csv` - HIGH/LOW volatility regimes
- `sensor2_volatility_model.pkl` - Saved GARCH model
- `sensor2_volatility_analysis.png` - Visualization

### Combined Analysis
- `combined_market_regimes.csv` - **Main output: 4-regime classification**
- `combined_market_regimes_transitions.csv` - Transition probabilities
- `combined_market_regimes_summary.txt` - Summary statistics
- `combined_market_regimes.png` - Complete visualization

---

## CONFIGURATION USED

- Training Ratio: 70% / 30%
- Number of Regimes: 2
- Volatility Percentile: 75th
- Date Column: Date
- Price Column: Adj Close

---

## NEXT STEPS

1. Review the visualizations in the output folder
2. Check `combined_market_regimes.csv` for complete regime history
3. Use current regime (Stable Bull) for trading decisions
4. Monitor for regime transitions using transition matrix

---

**Analysis Complete!** ✅
