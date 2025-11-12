"""
SENSOR 3: Kalman Filter Trend Detection
Uses Kalman filter to estimate trend direction and velocity

Detects:
- UP trend (positive velocity) = 1
- DOWN trend (negative velocity) = 0

This provides a third independent signal alongside MRS (Sensor 1) and GARCH (Sensor 2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')


class KalmanTrendSensor:
    """Kalman filter-based trend detection sensor"""
    
    def __init__(self, observation_covariance=1.0, transition_covariance=0.01):
        """
        Initialize Kalman trend sensor
        
        Parameters:
        -----------
        observation_covariance : float
            Measurement noise (higher = noisier observations)
        transition_covariance : float
            Process noise (higher = more responsive to changes)
        """
        self.obs_cov = observation_covariance
        self.trans_cov = transition_covariance
        
    def load_data(self, filepath, price_col='Adj Close'):
        """Load cleaned price data"""
        print("="*80)
        print("SENSOR 3: KALMAN TREND FILTER")
        print("="*80)
        
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.price_col = price_col
        
        print(f"\n✓ Loaded data: {len(self.data)} days")
        print(f"  Price column: {price_col}")
        print(f"  Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
        
        return self
    
    def fit_kalman_filter(self):
        """Fit Kalman filter to estimate trend and velocity"""
        print("\n" + "="*80)
        print("FITTING KALMAN FILTER")
        print("="*80)
        
        prices = self.data[self.price_col].values
        n = len(prices)
        
        # State space model: [price, velocity]
        # State transition: price(t+1) = price(t) + velocity(t)
        #                   velocity(t+1) = velocity(t)
        
        transition_matrix = np.array([
            [1, 1],  # price(t+1) = price(t) + velocity(t)
            [0, 1]   # velocity(t+1) = velocity(t)
        ])
        
        # We only observe price, not velocity
        observation_matrix = np.array([[1, 0]])
        
        # Initial state: [first price, 0 velocity]
        initial_state_mean = [prices[0], 0]
        
        # Covariance matrices
        observation_covariance = self.obs_cov
        transition_covariance = self.trans_cov * np.eye(2)
        
        print(f"\nKalman Parameters:")
        print(f"  Observation covariance: {self.obs_cov}")
        print(f"  Transition covariance: {self.trans_cov}")
        
        # Create and fit Kalman filter
        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=initial_state_mean,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance
        )
        
        print("\n✓ Fitting Kalman filter...")
        
        # Filter and smooth
        self.state_means, self.state_covariances = self.kf.filter(prices)
        self.smoothed_state_means, self.smoothed_state_covariances = self.kf.smooth(prices)
        
        # Extract components
        self.data['Price_Kalman'] = self.smoothed_state_means[:, 0]
        self.data['Velocity_Kalman'] = self.smoothed_state_means[:, 1]
        
        print("✓ Kalman filter fitted")
        print(f"  Smoothed price range: ${self.data['Price_Kalman'].min():.2f} - ${self.data['Price_Kalman'].max():.2f}")
        print(f"  Velocity range: {self.data['Velocity_Kalman'].min():.4f} - {self.data['Velocity_Kalman'].max():.4f}")
        
        return self
    
    def detect_trend_regimes(self, velocity_threshold=0.0):
        """
        Detect trend based on Kalman velocity
        
        Parameters:
        -----------
        velocity_threshold : float
            Threshold for classifying UP vs DOWN (default: 0.0)
        """
        print("\n" + "="*80)
        print("DETECTING TREND REGIMES")
        print("="*80)
        
        # Simple classification based on velocity
        self.data['Trend_Regime'] = np.where(
            self.data['Velocity_Kalman'] > velocity_threshold,
            'UP',
            'DOWN'
        )
        
        # Also calculate smoothed returns for additional signal
        self.data['Smoothed_Returns'] = self.data['Price_Kalman'].pct_change()
        
        # Count regimes
        up_days = (self.data['Trend_Regime'] == 'UP').sum()
        down_days = (self.data['Trend_Regime'] == 'DOWN').sum()
        
        print(f"\nVelocity threshold: {velocity_threshold}")
        print(f"\nRegime Distribution:")
        print(f"  UP trend:   {up_days:4d} days ({up_days/len(self.data)*100:.1f}%)")
        print(f"  DOWN trend: {down_days:4d} days ({down_days/len(self.data)*100:.1f}%)")
        
        return self
    
    def calculate_statistics(self):
        """Calculate performance statistics for each regime"""
        print("\n" + "="*80)
        print("REGIME STATISTICS")
        print("="*80)
        
        for regime in ['UP', 'DOWN']:
            regime_data = self.data[self.data['Trend_Regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate returns
            returns = regime_data[self.price_col].pct_change().dropna()
            
            if len(returns) > 0:
                days = len(regime_data)
                daily_return = returns.mean()
                annual_return = daily_return * 252 * 100
                volatility = returns.std() * np.sqrt(252) * 100
                
                if returns.std() > 0:
                    sharpe = (daily_return * 252) / (returns.std() * np.sqrt(252))
                else:
                    sharpe = 0
                
                avg_velocity = regime_data['Velocity_Kalman'].mean()
                
                print(f"\n{regime} Trend:")
                print(f"  Days: {days}")
                print(f"  Annual Return: {annual_return:+.2f}%")
                print(f"  Annual Volatility: {volatility:.2f}%")
                print(f"  Sharpe Ratio: {sharpe:.3f}")
                print(f"  Avg Velocity: {avg_velocity:+.4f}")
        
        return self
    
    def get_current_regime(self):
        """Get current trend regime"""
        print("\n" + "="*80)
        print("CURRENT TREND STATE")
        print("="*80)
        
        latest = self.data.iloc[-1]
        
        print(f"\nDate: {self.data.index[-1].date()}")
        print(f"Price (actual): ${latest[self.price_col]:.2f}")
        print(f"Price (Kalman): ${latest['Price_Kalman']:.2f}")
        print(f"Velocity: {latest['Velocity_Kalman']:+.4f}")
        print(f"\nTrend Regime: {latest['Trend_Regime']}")
        
        return self
    
    def save_results(self, output_file='sensor3_kalman_trend.csv',
                    summary_file='sensor3_kalman_summary.txt'):
        """Save Kalman sensor results"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save main data
        output = self.data[[
            self.price_col,
            'Price_Kalman',
            'Velocity_Kalman',
            'Trend_Regime'
        ]].copy()
        
        output.to_csv(output_file)
        print(f"\n✓ Saved trend predictions to: {output_file}")
        
        # Save summary
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SENSOR 3: KALMAN TREND FILTER SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Days: {len(self.data)}\n")
            f.write(f"Date Range: {self.data.index.min().date()} to {self.data.index.max().date()}\n\n")
            
            f.write("REGIME DISTRIBUTION:\n")
            f.write("-"*80 + "\n")
            
            for regime in ['UP', 'DOWN']:
                count = (self.data['Trend_Regime'] == regime).sum()
                pct = count / len(self.data) * 100
                f.write(f"{regime:10s}: {count:4d} days ({pct:5.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("CURRENT STATE:\n")
            f.write("="*80 + "\n")
            
            latest = self.data.iloc[-1]
            f.write(f"Date: {self.data.index[-1].date()}\n")
            f.write(f"Price (actual): ${latest[self.price_col]:.2f}\n")
            f.write(f"Price (Kalman): ${latest['Price_Kalman']:.2f}\n")
            f.write(f"Velocity: {latest['Velocity_Kalman']:+.4f}\n")
            f.write(f"Trend: {latest['Trend_Regime']}\n")
        
        print(f"✓ Saved summary to: {summary_file}")
        
        return self
    
    def plot(self, save_file='sensor3_kalman_trend.png'):
        """Create visualization"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATION")
        print("="*80)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Actual vs Kalman smoothed price
        axes[0].plot(self.data.index, self.data[self.price_col], 
                    linewidth=1, alpha=0.6, color='gray', label='Actual Price')
        axes[0].plot(self.data.index, self.data['Price_Kalman'],
                    linewidth=2, color='blue', label='Kalman Smoothed')
        axes[0].set_title('Kalman Filter: Price Smoothing', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Velocity (trend strength)
        axes[1].plot(self.data.index, self.data['Velocity_Kalman'],
                    linewidth=0.8, color='purple', label='Kalman Velocity')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].fill_between(self.data.index, 0, self.data['Velocity_Kalman'],
                            where=(self.data['Velocity_Kalman'] > 0),
                            color='green', alpha=0.3, label='Positive Velocity')
        axes[1].fill_between(self.data.index, 0, self.data['Velocity_Kalman'],
                            where=(self.data['Velocity_Kalman'] <= 0),
                            color='red', alpha=0.3, label='Negative Velocity')
        axes[1].set_title('Kalman Velocity (Trend Strength)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Velocity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Trend regime timeline
        regime_numeric = self.data['Trend_Regime'].map({'DOWN': 0, 'UP': 1})
        colors = self.data['Trend_Regime'].map({'DOWN': 'red', 'UP': 'green'})
        
        axes[2].scatter(self.data.index, regime_numeric, c=colors, alpha=0.6, s=10)
        axes[2].set_title('Kalman Trend Regime Timeline', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Trend')
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(['DOWN', 'UP'])
        axes[2].set_ylim(-0.5, 1.5)
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_file}")
        plt.close()
        
        return self


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SENSOR 3: KALMAN TREND FILTER")
    print("="*80)
    
    # Initialize sensor
    sensor = KalmanTrendSensor(
        observation_covariance=1.0,
        transition_covariance=0.01
    )
    
    # Run pipeline
    (sensor
     .load_data('../Nifty_50_Cleaned.csv', price_col='Adj Close')
     .fit_kalman_filter()
     .detect_trend_regimes()
     .calculate_statistics()
     .get_current_regime()
     .save_results(
         output_file='sensor3_kalman_trend.csv',
         summary_file='sensor3_kalman_summary.txt'
     )
     .plot(save_file='sensor3_kalman_trend.png')
    )
    
    print("\n" + "="*80)
    print("✅ KALMAN TREND SENSOR COMPLETE!")
    print("="*80)
    print("\nOutput Files:")
    print("  1. sensor3_kalman_trend.csv - Trend predictions")
    print("  2. sensor3_kalman_summary.txt - Summary statistics")
    print("  3. sensor3_kalman_trend.png - Visualization")


if __name__ == "__main__":
    main()
