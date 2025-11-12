"""
SENSOR 2: Volatility Regime Detection using GARCH
Part of the Market Regime Detection System

This sensor detects High/Low volatility regimes to complement Sensor 1 (Trend)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Configuration
CLEANED_DATA_FILE = "Nifty_IT_Cleaned.csv"
TRAIN_END_DATE = '2020-12-31'
TEST_START_DATE = '2021-01-01'
VOLATILITY_THRESHOLD_PERCENTILE = 75  # 75th percentile = High vol

print("="*60)
print("SENSOR 2: VOLATILITY REGIME DETECTION")
print("="*60)

# 1. Load Data
print("\nLoading data...")
data = pd.read_csv(CLEANED_DATA_FILE, index_col=0, parse_dates=True)
data['Returns'] = data['Log Returns'] * 100  # GARCH works better with percentage returns

# Split
train_data = data.loc[:TRAIN_END_DATE].copy()
test_data = data.loc[TEST_START_DATE:].copy()

print(f"Train: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
print(f"Test: {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")

# 2. Fit GARCH(1,1) Model
print("\n" + "="*60)
print("FITTING GARCH(1,1) MODEL")
print("="*60)

train_returns = train_data['Returns'].dropna()

# GARCH(1,1) specification:
# - p=1: one lag of squared residuals (ARCH term)
# - q=1: one lag of conditional variance (GARCH term)
garch_model = arch_model(
    train_returns,
    vol='Garch',
    p=1,
    q=1,
    mean='Constant',
    dist='normal'
)

print("\nFitting model (this may take a minute)...")
garch_fit = garch_model.fit(disp='off', show_warning=False)

print("\n✓ GARCH model fitted successfully!")
print("\nModel Parameters:")
print(garch_fit.params)

# 3. Extract Conditional Volatility
print("\n" + "="*60)
print("EXTRACTING CONDITIONAL VOLATILITY")
print("="*60)

# Get fitted volatility for training data
train_data['Conditional_Volatility'] = garch_fit.conditional_volatility

# Forecast volatility for test data
print("\nForecasting volatility for test period...")
forecast_horizon = len(test_data)

# Rolling forecast approach (more realistic)
test_volatilities = []
for i in range(len(test_data)):
    # Use all data up to this point
    current_data = pd.concat([train_returns, test_data['Returns'].iloc[:i]])
    
    # Refit model (or use update if you want faster but less accurate)
    if i % 100 == 0:  # Refit every 100 days to save time
        current_model = arch_model(current_data, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        current_fit = current_model.fit(disp='off', show_warning=False)
        test_volatilities.append(current_fit.conditional_volatility.iloc[-1])
    elif i == 0:
        # First prediction from training model
        forecast = garch_fit.forecast(horizon=1)
        test_volatilities.append(np.sqrt(forecast.variance.values[-1, 0]))
    else:
        # Use last fitted model
        test_volatilities.append(test_volatilities[-1])  # Simplified

print("✓ Volatility forecasting complete!")

test_data['Conditional_Volatility'] = test_volatilities[:len(test_data)]

# 4. Define Volatility Regimes
print("\n" + "="*60)
print("DEFINING VOLATILITY REGIMES")
print("="*60)

# Calculate threshold from training data
vol_threshold = np.percentile(train_data['Conditional_Volatility'], 
                               VOLATILITY_THRESHOLD_PERCENTILE)

print(f"\nVolatility Threshold ({VOLATILITY_THRESHOLD_PERCENTILE}th percentile): {vol_threshold:.4f}%")

# Classify regimes
train_data['Volatility_Regime'] = np.where(
    train_data['Conditional_Volatility'] > vol_threshold,
    'High',
    'Low'
)

test_data['Volatility_Regime'] = np.where(
    test_data['Conditional_Volatility'] > vol_threshold,
    'High',
    'Low'
)

print("\nTraining Data - Volatility Regime Distribution:")
print(train_data['Volatility_Regime'].value_counts())
print("\nTest Data - Volatility Regime Distribution:")
print(test_data['Volatility_Regime'].value_counts())

# 5. Analyze Regime Quality
print("\n" + "="*60)
print("VOLATILITY REGIME QUALITY ANALYSIS")
print("="*60)

def analyze_vol_regimes(data, title):
    print(f"\n{title}")
    print("-" * 50)
    
    for regime in ['Low', 'High']:
        regime_data = data[data['Volatility_Regime'] == regime]
        if len(regime_data) == 0:
            continue
        
        returns = regime_data['Returns'] / 100  # Convert back to decimal
        vol = regime_data['Conditional_Volatility']
        
        print(f"\n{regime} Volatility Regime:")
        print(f"  Days: {len(regime_data)} ({len(regime_data)/len(data)*100:.1f}%)")
        print(f"  Avg Volatility: {vol.mean():.4f}%")
        print(f"  Avg Return: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
        print(f"  Sharpe Ratio: {(returns.mean() / returns.std() * np.sqrt(252)):.3f}")

analyze_vol_regimes(train_data, "TRAINING DATA")
analyze_vol_regimes(test_data, "TEST DATA")

# 6. Visualization
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

all_data = pd.concat([train_data, test_data])

# Plot 1: Price with volatility regime shading
ax1 = axes[0]
ax1.plot(all_data.index, all_data['Adj Close'], linewidth=1, color='black')

# Shade high volatility periods
high_vol_mask = all_data['Volatility_Regime'] == 'High'
ax1.fill_between(all_data.index, 
                 all_data['Adj Close'].min(), 
                 all_data['Adj Close'].max(),
                 where=high_vol_mask, 
                 alpha=0.3, 
                 color='red', 
                 label='High Volatility')

ax1.axvline(pd.Timestamp(TRAIN_END_DATE), color='blue', linestyle='--', linewidth=2, label='Train/Test')
ax1.set_title('Nifty IT: Price with Volatility Regimes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Conditional Volatility
ax2 = axes[1]
ax2.plot(all_data.index, all_data['Conditional_Volatility'], linewidth=0.8, color='orange', label='GARCH Vol')
ax2.axhline(vol_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({vol_threshold:.2f}%)')
ax2.axvline(pd.Timestamp(TRAIN_END_DATE), color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_title('GARCH Conditional Volatility', fontsize=12, fontweight='bold')
ax2.set_ylabel('Volatility (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Returns colored by volatility regime
ax3 = axes[2]
low_vol = all_data[all_data['Volatility_Regime'] == 'Low']
high_vol = all_data[all_data['Volatility_Regime'] == 'High']

ax3.scatter(low_vol.index, low_vol['Returns'], c='green', alpha=0.4, s=8, label='Low Vol')
ax3.scatter(high_vol.index, high_vol['Returns'], c='red', alpha=0.4, s=8, label='High Vol')
ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax3.axvline(pd.Timestamp(TRAIN_END_DATE), color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_title('Returns by Volatility Regime', fontsize=12, fontweight='bold')
ax3.set_ylabel('Returns (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Regime timeline
ax4 = axes[3]
regime_numeric = all_data['Volatility_Regime'].map({'Low': 0, 'High': 1})
colors = all_data['Volatility_Regime'].map({'Low': 'green', 'High': 'red'})

ax4.scatter(all_data.index, regime_numeric, c=colors, alpha=0.6, s=10)
ax4.axvline(pd.Timestamp(TRAIN_END_DATE), color='blue', linestyle='--', linewidth=2, label='Train/Test')
ax4.set_title('Volatility Regime Timeline', fontsize=12, fontweight='bold')
ax4.set_ylabel('Regime')
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['Low', 'High'])
ax4.set_ylim(-0.5, 1.5)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('sensor2_volatility_regimes.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved as 'sensor2_volatility_regimes.png'")
plt.show()

# 7. Save Results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output_data = all_data[['Adj Close', 'Returns', 'Conditional_Volatility', 'Volatility_Regime']]
output_data.to_csv('sensor2_volatility_regime.csv')
print("✓ Sensor 2 output saved to 'sensor2_volatility_regime.csv'")

# Save model
import pickle
model_data = {
    'garch_fit': garch_fit,
    'vol_threshold': vol_threshold,
    'percentile': VOLATILITY_THRESHOLD_PERCENTILE
}
with open('sensor2_volatility_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✓ Model saved to 'sensor2_volatility_model.pkl'")

# 8. Summary
print("\n" + "="*60)
print("SENSOR 2: VOLATILITY REGIME DETECTION - COMPLETE")
print("="*60)

print(f"\nModel: GARCH(1,1)")
print(f"Threshold: {vol_threshold:.4f}% ({VOLATILITY_THRESHOLD_PERCENTILE}th percentile)")

print("\nTest Data Summary:")
test_summary = test_data.groupby('Volatility_Regime').agg({
    'Returns': ['mean', 'std', 'count'],
    'Conditional_Volatility': 'mean'
}).round(4)
print(test_summary)

current_vol_regime = test_data['Volatility_Regime'].iloc[-1]
current_vol = test_data['Conditional_Volatility'].iloc[-1]
print(f"\n*** CURRENT VOLATILITY REGIME: {current_vol_regime} ***")
print(f"Current Volatility: {current_vol:.4f}%")

print("\n" + "="*60)
print("✓✓✓ SENSOR 2 READY! ✓✓✓")
print("="*60)

print("\nNext: Combine Sensor 1 (Trend) + Sensor 2 (Volatility)")
print("      → Create final regime aggregation layer")
