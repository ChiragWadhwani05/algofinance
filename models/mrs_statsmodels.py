"""
Markov Regime Switching (MRS) Trend Model - BULL/BEAR Detection
Uses statsmodels MarkovRegression for regime identification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
warnings.filterwarnings('ignore')


class MRSTrendModel:
    """Markov Regime Switching Model using statsmodels"""
    
    def __init__(self, n_regimes=2, train_ratio=0.7):
        self.n_regimes = n_regimes
        self.train_ratio = train_ratio
        self.regime_labels = {}
        
    def load_and_prepare_data(self, filepath, price_col='Adj Close'):
        """Load data and calculate returns"""
        print("="*80)
        print("LOADING & PREPARING DATA")
        print("="*80)
        
        # Load data
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Calculate returns
        if 'Returns' not in self.data.columns:
            if 'Log Returns' in self.data.columns:
                self.data['Returns'] = self.data['Log Returns'] * 100
            else:
                self.data['Returns'] = self.data[price_col].pct_change() * 100
        
        # Add volatility feature
        self.data['Vol_10'] = self.data['Returns'].rolling(10).std()
        self.data = self.data.dropna()
        
        # Split train/test
        split_idx = int(len(self.data) * self.train_ratio)
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        print(f"✓ Loaded {len(self.data)} days")
        print(f"✓ Training: {len(self.train_data)} days ({self.train_ratio*100:.0f}%)")
        print(f"✓ Testing:  {len(self.test_data)} days ({(1-self.train_ratio)*100:.0f}%)")
        
        return self
    
    def fit(self):
        """Fit MRS model on training data"""
        print("\n" + "="*80)
        print("FITTING MRS MODEL")
        print("="*80)
        
        # Fit Markov Switching Model
        model = MarkovRegression(
            endog=self.train_data['Returns'].values,
            k_regimes=self.n_regimes,
            exog=self.train_data[['Vol_10']].values,
            switching_variance=True
        )
        
        self.results = model.fit(maxiter=1000, disp=False)
        
        print(f"✓ Model fitted | Log-Likelihood: {self.results.llf:.2f}")
        print(f"✓ AIC: {self.results.aic:.2f} | BIC: {self.results.bic:.2f}")
        
        return self
    
    def predict(self):
        """Predict regimes for train and test data"""
        print("\n" + "="*80)
        print("PREDICTING REGIMES")
        print("="*80)
        
        # Get train probabilities
        train_probs = pd.DataFrame(self.results.smoothed_marginal_probabilities)
        self.train_data['Regime_ID'] = train_probs.idxmax(axis=1).values
        
        # Predict on full data for test set
        all_endog = np.concatenate([self.train_data['Returns'].values, 
                                    self.test_data['Returns'].values])
        all_exog = np.concatenate([self.train_data[['Vol_10']].values,
                                   self.test_data[['Vol_10']].values])
        
        full_model = MarkovRegression(endog=all_endog, k_regimes=self.n_regimes,
                                      exog=all_exog, switching_variance=True)
        full_results = full_model.fit(start_params=self.results.params, maxiter=50, disp=False)
        
        # Get test probabilities
        all_probs = pd.DataFrame(full_results.smoothed_marginal_probabilities)
        test_probs = all_probs.iloc[len(self.train_data):]
        self.train_data['Regime_ID'] = train_probs.idxmax(axis=1).values
        self.test_data['Regime_ID'] = test_probs.idxmax(axis=1).values
        
        # Add probabilities
        for i in range(self.n_regimes):
            self.train_data[f'Prob_{i}'] = train_probs.iloc[:, i].values
            self.test_data[f'Prob_{i}'] = test_probs.iloc[:, i].values
        
        print("✓ Predictions complete")
        return self
    
    def label_regimes(self):
        """Label regimes as BULL/BEAR based on returns"""
        print("\n" + "="*80)
        print("LABELING REGIMES")
        print("="*80)
        
        # Calculate mean return for each regime
        regime_stats = []
        for i in range(self.n_regimes):
            mask = self.train_data['Regime_ID'] == i
            returns = self.train_data.loc[mask, 'Returns']
            regime_stats.append({
                'id': i,
                'mean_return': returns.mean(),
                'days': mask.sum()
            })
        
        # Sort by return and label
        regime_stats.sort(key=lambda x: x['mean_return'], reverse=True)
        
        if self.n_regimes == 2:
            self.regime_labels = {
                regime_stats[0]['id']: 'BULL',
                regime_stats[1]['id']: 'BEAR'
            }
        
        # Apply labels
        self.train_data['Regime'] = self.train_data['Regime_ID'].map(self.regime_labels)
        self.test_data['Regime'] = self.test_data['Regime_ID'].map(self.regime_labels)
        
        # Get transition matrix
        trans = self.results.regime_transition
        if trans.ndim == 3:
            trans = trans[:, :, 0]
        
        # Print summary
        for stat in regime_stats:
            label = self.regime_labels[stat['id']]
            print(f"\n{label}:")
            print(f"  Daily Return: {stat['mean_return']:.4f}% | Annual: {stat['mean_return']*252:.2f}%")
            print(f"  Days: {stat['days']} ({stat['days']/len(self.train_data)*100:.1f}%)")
            print(f"  Duration: {self.results.expected_durations[stat['id']]:.1f} days")
        
        print(f"\n✓ Transition: BULL→BULL={trans[regime_stats[0]['id'], regime_stats[0]['id']]:.2%}, "
              f"BEAR→BEAR={trans[regime_stats[1]['id'], regime_stats[1]['id']]:.2%}")
        
        return self
    
    def evaluate(self):
        """Evaluate performance on test set"""
        print("\n" + "="*80)
        print("TEST SET PERFORMANCE")
        print("="*80)
        
        for regime in ['BULL', 'BEAR']:
            data = self.test_data[self.test_data['Regime'] == regime]
            if len(data) > 0:
                daily_ret = data['Returns'].mean()
                annual_ret = daily_ret * 252
                sharpe = annual_ret / (data['Returns'].std() * np.sqrt(252))
                
                print(f"\n{regime}:")
                print(f"  Days: {len(data)} ({len(data)/len(self.test_data)*100:.1f}%)")
                print(f"  Annual Return: {annual_ret:.2f}%")
                print(f"  Sharpe Ratio: {sharpe:.3f}")
        
        return self
    
    def get_current_regime(self):
        """Get current market regime"""
        print("\n" + "="*80)
        print("CURRENT REGIME")
        print("="*80)
        
        latest = self.test_data.iloc[-1]
        probs = {self.regime_labels[i]: latest[f'Prob_{i}'] 
                 for i in range(self.n_regimes)}
        
        print(f"\nDate: {self.test_data.index[-1].date()}")
        print(f"Regime: {latest['Regime']} ({max(probs.values()):.1%} confidence)")
        for label, prob in sorted(probs.items()):
            print(f"  {label}: {prob:.1%}")
        
        return self
    
    def save_results(self, prefix='mrs'):
        """Save predictions and summary"""
        # Save predictions
        all_data = pd.concat([self.train_data, self.test_data])
        all_data.to_csv(f'{prefix}_predictions.csv')
        
        # Save summary
        with open(f'{prefix}_summary.txt', 'w') as f:
            f.write(str(self.results.summary()))
        
        print(f"\n✓ Saved: {prefix}_predictions.csv, {prefix}_summary.txt")
        return self
    
    def plot(self, price_col='Adj Close', save_file='mrs_analysis.png'):
        """Create visualization"""
        all_data = pd.concat([self.train_data, self.test_data])
        colors = {'BULL': 'green', 'BEAR': 'red'}
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Price with regimes
        for regime, color in colors.items():
            mask = all_data['Regime'] == regime
            if mask.sum() > 0:
                axes[0].scatter(all_data.index[mask], all_data[price_col][mask],
                              c=color, s=10, alpha=0.6, label=regime)
        
        axes[0].axvline(self.test_data.index[0], color='blue', linestyle='--', 
                       alpha=0.7, label='Train/Test Split')
        axes[0].set_ylabel('Price')
        axes[0].set_title('MRS Model: Price with Regimes')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Regime probabilities
        for i in range(self.n_regimes):
            label = self.regime_labels[i]
            probs = pd.concat([self.train_data[f'Prob_{i}'], 
                              self.test_data[f'Prob_{i}']])
            axes[1].plot(all_data.index, probs, label=label, 
                        color=colors[label], alpha=0.7)
        
        axes[1].axvline(self.test_data.index[0], color='blue', linestyle='--', alpha=0.7)
        axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Regime Probabilities')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_file}")
        
        return self


# Main execution
if __name__ == "__main__":
    
    print("="*80)
    print("MARKOV REGIME SWITCHING - BULL/BEAR TREND MODEL")
    print("="*80)
    
    # Run model
    model = MRSTrendModel(n_regimes=2, train_ratio=0.70)
    
    (model
     .load_and_prepare_data("Nifty_50_Cleaned.csv")
     .fit()
     .predict()
     .label_regimes()
     .evaluate()
     .get_current_regime()
     .save_results(prefix='mrs_statsmodels')
     .plot(save_file='mrs_statsmodels_analysis.png')
    )
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
