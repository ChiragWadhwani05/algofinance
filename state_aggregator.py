"""
STATE AGGREGATOR: Combine Trend (MRS) + Volatility (GARCH) Sensors
Creates unified 4-regime market classification system

Regime Map:
- [1, 0] = Stable Bull    (BULL + Low Vol)
- [1, 1] = Volatile Bull  (BULL + High Vol)
- [0, 0] = Stable Bear    (BEAR + Low Vol)
- [0, 1] = Panic/Crash    (BEAR + High Vol)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class StateAggregator:
    """Combines multiple regime sensors into unified state vector"""
    
    def __init__(self):
        self.regime_map = {
            (1, 0): "Stable Bull",
            (1, 1): "Volatile Bull", 
            (0, 0): "Stable Bear",
            (0, 1): "Panic/Crash"
        }
        
        self.regime_colors = {
            "Stable Bull": "#00AA00",      # Dark green
            "Volatile Bull": "#FFB000",    # Orange
            "Stable Bear": "#FF6B6B",      # Light red
            "Panic/Crash": "#CC0000"       # Dark red
        }
        
    def load_sensor_outputs(self, trend_file, volatility_file):
        """Load outputs from Sensor 1 (Trend) and Sensor 2 (Volatility)"""
        print("="*80)
        print("STATE AGGREGATOR: LOADING SENSOR OUTPUTS")
        print("="*80)
        
        # Load Sensor 1: MRS Trend (BULL/BEAR)
        self.trend_data = pd.read_csv(trend_file, index_col=0, parse_dates=True)
        print(f"\n✓ Loaded Sensor 1 (Trend): {len(self.trend_data)} days")
        print(f"  Columns: {list(self.trend_data.columns)}")
        
        # Load Sensor 2: GARCH Volatility (High/Low)
        self.vol_data = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
        print(f"\n✓ Loaded Sensor 2 (Volatility): {len(self.vol_data)} days")
        print(f"  Columns: {list(self.vol_data.columns)}")
        
        return self
    
    def create_state_vectors(self):
        """Create binary state vectors [trend_bit, vol_bit]"""
        print("\n" + "="*80)
        print("CREATING STATE VECTORS")
        print("="*80)
        
        # Merge data on date index
        self.combined = pd.merge(
            self.trend_data[['Adj Close', 'Regime']].rename(columns={'Regime': 'Trend_Regime'}),
            self.vol_data[['Volatility_Regime', 'Conditional_Volatility']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        print(f"\n✓ Merged data: {len(self.combined)} days (common dates)")
        
        # Create binary encoding
        # Trend: BULL=1, BEAR=0
        self.combined['Trend_Bit'] = (self.combined['Trend_Regime'] == 'BULL').astype(int)
        
        # Volatility: Low=0, High=1
        self.combined['Vol_Bit'] = (self.combined['Volatility_Regime'] == 'High').astype(int)
        
        # Create state vector tuple
        self.combined['State_Vector'] = list(zip(
            self.combined['Trend_Bit'], 
            self.combined['Vol_Bit']
        ))
        
        # Map to regime names
        self.combined['Market_Regime'] = self.combined['State_Vector'].map(self.regime_map)
        
        print("\nState Vector Encoding:")
        print("  Trend: BULL=1, BEAR=0")
        print("  Volatility: Low=0, High=1")
        print("\n✓ State vectors created")
        
        return self
    
    def analyze_regimes(self):
        """Analyze characteristics of each market regime"""
        print("\n" + "="*80)
        print("MARKET REGIME ANALYSIS")
        print("="*80)
        
        # Overall distribution
        print("\nRegime Distribution:")
        regime_counts = self.combined['Market_Regime'].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(self.combined) * 100
            print(f"  {regime:20s}: {count:4d} days ({pct:5.1f}%)")
        
        # Detailed statistics per regime
        print("\n" + "-"*80)
        print("REGIME CHARACTERISTICS")
        print("-"*80)
        
        for state_vector, regime_name in sorted(self.regime_map.items()):
            regime_data = self.combined[self.combined['Market_Regime'] == regime_name]
            
            if len(regime_data) == 0:
                print(f"\n{regime_name} [{state_vector}]: No data")
                continue
            
            # Calculate statistics
            days = len(regime_data)
            pct = days / len(self.combined) * 100
            
            # Price changes
            price_changes = regime_data['Adj Close'].pct_change()
            daily_return = price_changes.mean()
            annual_return = daily_return * 252 * 100
            volatility = price_changes.std() * np.sqrt(252) * 100
            
            if price_changes.std() > 0:
                sharpe = (daily_return * 252) / (price_changes.std() * np.sqrt(252))
            else:
                sharpe = 0
            
            # Average volatility from GARCH
            avg_garch_vol = regime_data['Conditional_Volatility'].mean()
            
            print(f"\n{regime_name} {list(state_vector)}:")
            print(f"  Days: {days} ({pct:.1f}%)")
            print(f"  Annual Return: {annual_return:+.2f}%")
            print(f"  Annual Volatility: {volatility:.2f}%")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            print(f"  Avg GARCH Vol: {avg_garch_vol:.2f}%")
        
        return self
    
    def calculate_transitions(self):
        """Calculate regime transition probabilities"""
        print("\n" + "="*80)
        print("REGIME TRANSITION ANALYSIS")
        print("="*80)
        
        # Create transition matrix
        regimes = list(self.regime_map.values())
        n_regimes = len(regimes)
        
        transition_counts = pd.DataFrame(0, index=regimes, columns=regimes)
        
        # Count transitions
        for i in range(len(self.combined) - 1):
            current = self.combined['Market_Regime'].iloc[i]
            next_regime = self.combined['Market_Regime'].iloc[i + 1]
            transition_counts.loc[current, next_regime] += 1
        
        # Convert to probabilities
        self.transition_matrix = transition_counts.div(
            transition_counts.sum(axis=1), 
            axis=0
        ).fillna(0)
        
        print("\nTransition Probability Matrix:")
        print("(Row = From, Column = To)")
        print("\n" + self.transition_matrix.to_string())
        
        # Persistence (probability of staying in same regime)
        print("\n" + "-"*80)
        print("REGIME PERSISTENCE:")
        for regime in regimes:
            if regime in self.transition_matrix.index:
                persistence = self.transition_matrix.loc[regime, regime]
                avg_duration = 1 / (1 - persistence) if persistence < 1 else float('inf')
                print(f"  {regime:20s}: {persistence:.1%} (avg duration: {avg_duration:.1f} days)")
        
        return self
    
    def get_current_state(self):
        """Get current market state"""
        print("\n" + "="*80)
        print("CURRENT MARKET STATE")
        print("="*80)
        
        latest = self.combined.iloc[-1]
        
        print(f"\nDate: {self.combined.index[-1].date()}")
        print(f"Price: ${latest['Adj Close']:.2f}")
        print(f"\nTrend: {latest['Trend_Regime']}")
        print(f"Volatility: {latest['Volatility_Regime']} ({latest['Conditional_Volatility']:.2f}%)")
        print(f"\nState Vector: {list(latest['State_Vector'])}")
        print(f"Market Regime: {latest['Market_Regime']}")
        
        # Get most likely next regime
        current_regime = latest['Market_Regime']
        if current_regime in self.transition_matrix.index:
            next_probs = self.transition_matrix.loc[current_regime]
            most_likely = next_probs.idxmax()
            prob = next_probs.max()
            
            print(f"\nMost Likely Next: {most_likely} ({prob:.1%} probability)")
        
        return self
    
    def save_results(self, output_file='combined_market_regimes.csv'):
        """Save combined regime data"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Prepare output
        output = self.combined[[
            'Adj Close', 
            'Trend_Regime', 
            'Volatility_Regime',
            'Trend_Bit',
            'Vol_Bit',
            'Market_Regime',
            'Conditional_Volatility'
        ]].copy()
        
        # Add readable state vector
        output['State_Vector'] = output.apply(
            lambda x: f"[{x['Trend_Bit']}, {x['Vol_Bit']}]", 
            axis=1
        )
        
        output.to_csv(output_file)
        print(f"\n✓ Saved combined regimes to: {output_file}")
        
        # Save transition matrix
        trans_file = output_file.replace('.csv', '_transitions.csv')
        self.transition_matrix.to_csv(trans_file)
        print(f"✓ Saved transition matrix to: {trans_file}")
        
        # Save summary statistics
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MARKET REGIME AGGREGATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("REGIME DISTRIBUTION:\n")
            f.write("-"*80 + "\n")
            regime_counts = self.combined['Market_Regime'].value_counts()
            for regime, count in regime_counts.items():
                pct = count / len(self.combined) * 100
                f.write(f"{regime:20s}: {count:4d} days ({pct:5.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("TRANSITION MATRIX:\n")
            f.write("="*80 + "\n")
            f.write(self.transition_matrix.to_string())
            
        print(f"✓ Saved summary to: {summary_file}")
        
        return self
    
    def plot(self, save_file='combined_market_regimes.png'):
        """Create comprehensive visualization"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Price with regime colors (MAIN PLOT)
        ax1 = fig.add_subplot(gs[0:2, :])
        
        for regime, color in self.regime_colors.items():
            mask = self.combined['Market_Regime'] == regime
            if mask.sum() > 0:
                ax1.scatter(self.combined.index[mask], 
                          self.combined['Adj Close'][mask],
                          c=color, s=15, alpha=0.7, label=regime, edgecolors='none')
        
        ax1.plot(self.combined.index, self.combined['Adj Close'], 
                color='black', linewidth=0.5, alpha=0.3, zorder=0)
        ax1.set_title('Market Regimes: Combined Trend + Volatility Analysis', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=11)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime timeline
        ax2 = fig.add_subplot(gs[2, :])
        
        # Map regimes to numbers for plotting
        regime_to_num = {name: i for i, name in enumerate(sorted(self.regime_map.values()))}
        regime_numeric = self.combined['Market_Regime'].map(regime_to_num)
        colors = self.combined['Market_Regime'].map(self.regime_colors)
        
        ax2.scatter(self.combined.index, regime_numeric, 
                   c=colors, s=20, alpha=0.8, edgecolors='none')
        ax2.set_yticks(range(len(regime_to_num)))
        ax2.set_yticklabels(sorted(self.regime_map.values()))
        ax2.set_title('Market Regime Timeline', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Regime distribution
        ax3 = fig.add_subplot(gs[3, 0])
        
        regime_counts = self.combined['Market_Regime'].value_counts()
        colors_list = [self.regime_colors[regime] for regime in regime_counts.index]
        
        bars = ax3.barh(range(len(regime_counts)), regime_counts.values, color=colors_list)
        ax3.set_yticks(range(len(regime_counts)))
        ax3.set_yticklabels(regime_counts.index)
        ax3.set_xlabel('Days')
        ax3.set_title('Regime Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (regime, count) in enumerate(regime_counts.items()):
            pct = count / len(self.combined) * 100
            ax3.text(count, i, f' {pct:.1f}%', va='center', fontsize=9)
        
        # Plot 4: State vector heatmap
        ax4 = fig.add_subplot(gs[3, 1])
        
        # Create 2x2 grid showing regime characteristics
        state_grid = np.zeros((2, 2))
        
        for (trend_bit, vol_bit), regime_name in self.regime_map.items():
            count = (self.combined['Market_Regime'] == regime_name).sum()
            state_grid[1-vol_bit, trend_bit] = count  # Flip vol_bit for better visualization
        
        im = ax4.imshow(state_grid, cmap='YlOrRd', aspect='auto')
        
        # Labels
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['BEAR\n(0)', 'BULL\n(1)'])
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Low Vol\n(0)', 'High Vol\n(1)'])
        ax4.set_xlabel('Trend Bit', fontsize=10)
        ax4.set_ylabel('Volatility Bit', fontsize=10)
        ax4.set_title('State Vector Distribution', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                vol_bit = 1 - i  # Flip back
                trend_bit = j
                regime_name = self.regime_map[(trend_bit, vol_bit)]
                count = int(state_grid[i, j])
                pct = count / len(self.combined) * 100
                
                text = f"{regime_name}\n{count} days\n({pct:.1f}%)"
                ax4.text(j, i, text, ha='center', va='center', 
                        fontsize=9, color='black' if count < state_grid.max()/2 else 'white')
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_file}")
        
        return self


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("STATE AGGREGATOR: COMBINING TREND + VOLATILITY SENSORS")
    print("="*80)
    
    # Initialize aggregator
    aggregator = StateAggregator()
    
    # Run pipeline
    (aggregator
     .load_sensor_outputs(
         trend_file='../mrs_statsmodels_predictions.csv',
         volatility_file='../sensor2_volatility_regime.csv'
     )
     .create_state_vectors()
     .analyze_regimes()
     .calculate_transitions()
     .get_current_state()
     .save_results(output_file='combined_market_regimes.csv')
     .plot(save_file='combined_market_regimes.png')
    )
    
    print("\n" + "="*80)
    print("✅ STATE AGGREGATION COMPLETE!")
    print("="*80)
    print("\nOutput Files:")
    print("  1. combined_market_regimes.csv - Full regime data")
    print("  2. combined_market_regimes_transitions.csv - Transition matrix")
    print("  3. combined_market_regimes_summary.txt - Statistics summary")
    print("  4. combined_market_regimes.png - Visualization")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
