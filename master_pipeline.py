"""
MASTER PIPELINE: Complete Regime Detection System
Runs cleaning → MRS → GARCH → State Aggregator → Saves all results

Usage:
    1. Edit pipeline_config.py to set your input CSV file
    2. Run: python3 master_pipeline.py
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Import configuration
import pipeline_config as config

# Import modules
from data_cleaner import DataCleaner
from models.mrs_statsmodels import MRSTrendModel
import warnings
warnings.filterwarnings('ignore')


class MasterPipeline:
    """Orchestrates the complete regime detection pipeline"""
    
    def __init__(self, config_module):
        """Initialize pipeline with configuration"""
        self.config = config_module
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup paths
        self.setup_paths()
        
    def setup_paths(self):
        """Setup all file paths"""
        print("="*80)
        print("MASTER PIPELINE: INITIALIZING")
        print("="*80)
        
        # Base directory
        self.base_dir = Path(__file__).parent
        self.parent_dir = self.base_dir.parent
        
        # Input file (support both relative and absolute paths)
        input_file = Path(self.config.INPUT_CSV_FILE)
        if input_file.is_absolute():
            self.input_csv = input_file
        else:
            # Try current directory first, then parent
            if (self.base_dir / input_file).exists():
                self.input_csv = self.base_dir / input_file
            elif (self.parent_dir / input_file).exists():
                self.input_csv = self.parent_dir / input_file
            else:
                self.input_csv = input_file  # Use as-is, will error later if not found
        
        # Output directory
        self.output_dir = self.base_dir / "results" / self.config.OUTPUT_FOLDER_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cleaned data path
        input_name = self.input_csv.stem
        self.cleaned_csv = self.output_dir / f"{input_name}_cleaned.csv"
        
        # Sensor output paths
        self.mrs_predictions = self.output_dir / "sensor1_trend_predictions.csv"
        self.mrs_summary = self.output_dir / "sensor1_trend_summary.txt"
        self.mrs_plot = self.output_dir / "sensor1_trend_analysis.png"
        
        self.garch_regime = self.output_dir / "sensor2_volatility_regime.csv"
        self.garch_model = self.output_dir / "sensor2_volatility_model.pkl"
        self.garch_plot = self.output_dir / "sensor2_volatility_analysis.png"
        
        # Aggregator output paths
        self.combined_csv = self.output_dir / "combined_market_regimes.csv"
        self.combined_transitions = self.output_dir / "combined_market_regimes_transitions.csv"
        self.combined_summary = self.output_dir / "combined_market_regimes_summary.txt"
        self.combined_plot = self.output_dir / "combined_market_regimes.png"
        self.combined_report = self.output_dir / "REGIME_ANALYSIS_REPORT.md"
        
        print(f"\n✓ Input file: {self.input_csv}")
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Timestamp: {self.timestamp}")
        
    def step1_clean_data(self):
        """Step 1: Clean raw CSV data"""
        print("\n" + "="*80)
        print("STEP 1/4: DATA CLEANING")
        print("="*80)
        
        cleaner = DataCleaner(
            date_col=self.config.DATE_COLUMN,
            price_col=self.config.PRICE_COLUMN,
            skip_rows=self.config.SKIP_ROWS,
            date_format=self.config.DATE_FORMAT,
            fill_method=self.config.FILL_METHOD
        )
        
        self.cleaned_data = cleaner.clean(
            str(self.input_csv),
            str(self.cleaned_csv)
        )
        
        print(f"\n✅ Step 1 complete! Cleaned data saved.")
        return self
        
    def step2_run_mrs(self):
        """Step 2: Run MRS Trend Sensor"""
        print("\n" + "="*80)
        print("STEP 2/4: MRS TREND SENSOR (BULL/BEAR DETECTION)")
        print("="*80)
        
        # Run MRS model
        model = MRSTrendModel(
            n_regimes=self.config.N_REGIMES,
            train_ratio=self.config.TRAIN_RATIO
        )
        
        (model
         .load_and_prepare_data(str(self.cleaned_csv), price_col=self.config.PRICE_COLUMN)
         .fit()
         .predict()
         .label_regimes()
         .evaluate()
         .get_current_regime()
         .save_results(prefix=str(self.output_dir / "sensor1_trend"))
         .plot(price_col=self.config.PRICE_COLUMN, save_file=str(self.mrs_plot))
        )
        
        # Rename files to match our naming convention
        os.rename(str(self.output_dir / "sensor1_trend_predictions.csv"), 
                  str(self.mrs_predictions))
        os.rename(str(self.output_dir / "sensor1_trend_summary.txt"), 
                  str(self.mrs_summary))
        
        print(f"\n✅ Step 2 complete! MRS outputs saved.")
        return self
        
    def step3_run_garch(self):
        """Step 3: Run GARCH Volatility Sensor"""
        print("\n" + "="*80)
        print("STEP 3/4: GARCH VOLATILITY SENSOR (HIGH/LOW DETECTION)")
        print("="*80)
        
        # Import here to avoid circular dependencies
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from arch import arch_model
        import pickle
        
        # Load cleaned data
        data = pd.read_csv(str(self.cleaned_csv), index_col=0, parse_dates=True)
        data['Returns'] = data['Log Returns'] * 100
        
        # Split data
        train_data = data.loc[:self.config.TRAIN_END_DATE].copy()
        test_data = data.loc[self.config.TEST_START_DATE:].copy()
        
        print(f"\nTrain: {len(train_data)} days")
        print(f"Test: {len(test_data)} days")
        
        # Fit GARCH
        print("\nFitting GARCH(1,1) model...")
        train_returns = train_data['Returns'].dropna()
        garch_model = arch_model(train_returns, vol='Garch', p=1, q=1, 
                                 mean='Constant', dist='normal')
        garch_fit = garch_model.fit(disp='off', show_warning=False)
        print("✓ GARCH model fitted")
        
        # Extract volatility
        train_data['Conditional_Volatility'] = garch_fit.conditional_volatility
        
        # Forecast for test data (simplified)
        print("Forecasting volatility...")
        test_volatilities = []
        for i in range(len(test_data)):
            if i % 100 == 0 and i > 0:
                current_data = pd.concat([train_returns, test_data['Returns'].iloc[:i]])
                current_model = arch_model(current_data, vol='Garch', p=1, q=1, 
                                          mean='Constant', dist='normal')
                current_fit = current_model.fit(disp='off', show_warning=False)
                test_volatilities.append(current_fit.conditional_volatility.iloc[-1])
            elif i == 0:
                forecast = garch_fit.forecast(horizon=1)
                test_volatilities.append(np.sqrt(forecast.variance.values[-1, 0]))
            else:
                test_volatilities.append(test_volatilities[-1])
        
        test_data['Conditional_Volatility'] = test_volatilities
        print("✓ Volatility forecasting complete")
        
        # Define regimes
        vol_threshold = np.percentile(train_data['Conditional_Volatility'], 
                                      self.config.VOLATILITY_PERCENTILE)
        print(f"\nVolatility threshold: {vol_threshold:.4f}%")
        
        train_data['Volatility_Regime'] = np.where(
            train_data['Conditional_Volatility'] > vol_threshold, 'High', 'Low'
        )
        test_data['Volatility_Regime'] = np.where(
            test_data['Conditional_Volatility'] > vol_threshold, 'High', 'Low'
        )
        
        # Save results
        all_data = pd.concat([train_data, test_data])
        output_data = all_data[[self.config.PRICE_COLUMN, 'Returns', 
                               'Conditional_Volatility', 'Volatility_Regime']]
        output_data.to_csv(str(self.garch_regime))
        
        # Save model
        model_data = {
            'garch_fit': garch_fit,
            'vol_threshold': vol_threshold,
            'percentile': self.config.VOLATILITY_PERCENTILE
        }
        with open(str(self.garch_model), 'wb') as f:
            pickle.dump(model_data, f)
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot volatility
        axes[0].plot(all_data.index, all_data['Conditional_Volatility'], 
                    linewidth=0.8, color='orange', label='GARCH Vol')
        axes[0].axhline(vol_threshold, color='red', linestyle='--', linewidth=1.5,
                       label=f'Threshold ({vol_threshold:.2f}%)')
        axes[0].axvline(pd.Timestamp(self.config.TRAIN_END_DATE), color='blue', 
                       linestyle='--', linewidth=2, alpha=0.5, label='Train/Test')
        axes[0].set_title('GARCH Conditional Volatility', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Volatility (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot regime timeline
        regime_numeric = all_data['Volatility_Regime'].map({'Low': 0, 'High': 1})
        colors = all_data['Volatility_Regime'].map({'Low': 'green', 'High': 'red'})
        axes[1].scatter(all_data.index, regime_numeric, c=colors, alpha=0.6, s=10)
        axes[1].axvline(pd.Timestamp(self.config.TRAIN_END_DATE), color='blue',
                       linestyle='--', linewidth=2, label='Train/Test')
        axes[1].set_title('Volatility Regime Timeline', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Regime')
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['Low', 'High'])
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(str(self.garch_plot), dpi=self.config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Step 3 complete! GARCH outputs saved.")
        return self
        
    def step4_aggregate_states(self):
        """Step 4: Run State Aggregator"""
        print("\n" + "="*80)
        print("STEP 4/4: STATE AGGREGATOR (COMBINING SENSORS)")
        print("="*80)
        
        # Import state aggregator
        from state_aggregator import StateAggregator
        
        # Run aggregator
        aggregator = StateAggregator()
        
        (aggregator
         .load_sensor_outputs(
             trend_file=str(self.mrs_predictions),
             volatility_file=str(self.garch_regime)
         )
         .create_state_vectors()
         .analyze_regimes()
         .calculate_transitions()
         .get_current_state()
         .save_results(output_file=str(self.combined_csv))
         .plot(save_file=str(self.combined_plot))
        )
        
        print(f"\n✅ Step 4 complete! Combined outputs saved.")
        return self
        
    def generate_summary_report(self):
        """Generate final summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        import pandas as pd
        
        # Load results
        combined_data = pd.read_csv(str(self.combined_csv), index_col=0, parse_dates=True)
        current_regime = combined_data.iloc[-1]
        
        report = f"""# REGIME DETECTION ANALYSIS REPORT

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Input File:** {self.input_csv.name}  
**Analysis Period:** {combined_data.index.min().date()} to {combined_data.index.max().date()}  
**Total Days:** {len(combined_data)}

---

## CURRENT MARKET STATE

**Date:** {combined_data.index[-1].date()}  
**Price:** ${current_regime[self.config.PRICE_COLUMN]:.2f}  
**State Vector:** {current_regime['State_Vector']}  
**Market Regime:** **{current_regime['Market_Regime']}**

**Components:**
- Trend (Sensor 1): {current_regime['Trend_Regime']}
- Volatility (Sensor 2): {current_regime['Volatility_Regime']} ({current_regime['Conditional_Volatility']:.2f}%)

---

## REGIME DISTRIBUTION

"""
        # Add regime distribution
        regime_counts = combined_data['Market_Regime'].value_counts()
        report += "| Regime | Days | Percentage |\n"
        report += "|--------|------|------------|\n"
        for regime, count in regime_counts.items():
            pct = count / len(combined_data) * 100
            report += f"| {regime} | {count} | {pct:.1f}% |\n"
        
        report += f"""

---

## OUTPUT FILES

All results saved in: `{self.output_dir.relative_to(self.base_dir)}/`

### Cleaned Data
- `{self.cleaned_csv.name}` - Cleaned input data

### Sensor 1: Trend (MRS)
- `{self.mrs_predictions.name}` - BULL/BEAR predictions
- `{self.mrs_summary.name}` - Model statistics
- `{self.mrs_plot.name}` - Visualization

### Sensor 2: Volatility (GARCH)
- `{self.garch_regime.name}` - HIGH/LOW volatility regimes
- `{self.garch_model.name}` - Saved GARCH model
- `{self.garch_plot.name}` - Visualization

### Combined Analysis
- `{self.combined_csv.name}` - **Main output: 4-regime classification**
- `{self.combined_transitions.name}` - Transition probabilities
- `{self.combined_summary.name}` - Summary statistics
- `{self.combined_plot.name}` - Complete visualization

---

## CONFIGURATION USED

- Training Ratio: {self.config.TRAIN_RATIO * 100:.0f}% / {(1-self.config.TRAIN_RATIO) * 100:.0f}%
- Number of Regimes: {self.config.N_REGIMES}
- Volatility Percentile: {self.config.VOLATILITY_PERCENTILE}th
- Date Column: {self.config.DATE_COLUMN}
- Price Column: {self.config.PRICE_COLUMN}

---

## NEXT STEPS

1. Review the visualizations in the output folder
2. Check `{self.combined_csv.name}` for complete regime history
3. Use current regime ({current_regime['Market_Regime']}) for trading decisions
4. Monitor for regime transitions using transition matrix

---

**Analysis Complete!** ✅
"""
        
        # Save report
        with open(str(self.combined_report), 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to: {self.combined_report.name}")
        return self
        
    def run(self):
        """Run the complete pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING MASTER PIPELINE")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # Run all steps
            (self
             .step1_clean_data()
             .step2_run_mrs()
             .step3_run_garch()
             .step4_aggregate_states()
             .generate_summary_report()
            )
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Final summary
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETE!")
            print("="*80)
            print(f"\n⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"\n📁 All results saved in:")
            print(f"   {self.output_dir}")
            print(f"\n📊 Main output file:")
            print(f"   {self.combined_csv.name}")
            print(f"\n📄 Read the report:")
            print(f"   {self.combined_report.name}")
            print("\n" + "="*80)
            
            return True
            
        except Exception as e:
            print("\n" + "="*80)
            print("❌ PIPELINE FAILED!")
            print("="*80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║           MASTER REGIME DETECTION PIPELINE                              ║
║                                                                          ║
║  Cleans → MRS Trend → GARCH Volatility → State Aggregator              ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run pipeline
    pipeline = MasterPipeline(config)
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
