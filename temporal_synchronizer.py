"""Temporal Synchronizer - Align features across multiple time frequencies

This module synchronizes features from different time resolutions (hourly, daily, weekly)
by forward-filling lower-frequency features to higher-frequency timestamps.

Architecture:
    Input: Hourly, Daily, Weekly features (separate DataFrames)
    Process: Forward-fill and align all features to common timestamps
    Output: Synchronized DataFrames at each frequency level

Usage:
    python temporal_synchronizer.py
    
    Or import:
    from temporal_synchronizer import TemporalSynchronizer
    
    ts = TemporalSynchronizer()
    synced_hourly, synced_daily, synced_weekly = ts.synchronize(
        hourly_df, daily_df, weekly_df
    )
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TemporalSynchronizer:
    """
    Synchronize features across multiple time frequencies.
    
    This class aligns features from daily and weekly timeframes to hourly data,
    and weekly features to daily data, enabling multi-timeframe analysis.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the Temporal Synchronizer.
        
        Parameters:
        -----------
        verbose : bool
            If True, print detailed logs during synchronization
        """
        self.verbose = verbose
        self.hourly_synced = None
        self.daily_synced = None
        self.weekly_synced = None
    
    def _log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def _prefix_columns(self, df, prefix):
        """
        Add prefix to all columns except datetime index.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        prefix : str
            Prefix to add (e.g., 'daily_', 'weekly_')
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with prefixed columns
        """
        df_copy = df.copy()
        df_copy.columns = [f"{prefix}{col}" for col in df_copy.columns]
        return df_copy
    
    def _forward_fill_to_frequency(self, source_df, target_index, prefix):
        """
        Forward-fill lower frequency features to higher frequency timestamps.
        
        Parameters:
        -----------
        source_df : pd.DataFrame
            Source dataframe with lower frequency (e.g., daily data)
        target_index : pd.DatetimeIndex
            Target timestamps at higher frequency (e.g., hourly)
        prefix : str
            Prefix for column names
        
        Returns:
        --------
        pd.DataFrame
            Forward-filled dataframe aligned to target_index
        """
        # Add prefix to source columns
        source_prefixed = self._prefix_columns(source_df, prefix)
        
        # Reindex to target frequency with forward fill
        aligned = source_prefixed.reindex(
            target_index.union(source_prefixed.index)
        ).sort_index()
        
        # Forward fill to propagate values
        aligned = aligned.fillna(method='ffill')
        
        # Keep only target timestamps
        aligned = aligned.reindex(target_index)
        
        return aligned
    
    def synchronize_hourly(self, hourly_df, daily_df=None, weekly_df=None):
        """
        Synchronize all features to hourly frequency.
        
        Parameters:
        -----------
        hourly_df : pd.DataFrame
            Hourly features with datetime index
        daily_df : pd.DataFrame, optional
            Daily features with datetime index
        weekly_df : pd.DataFrame, optional
            Weekly features with datetime index
        
        Returns:
        --------
        pd.DataFrame
            Synchronized hourly dataframe with all features
        """
        self._log("\n" + "="*80)
        self._log("SYNCHRONIZING TO HOURLY FREQUENCY")
        self._log("="*80)
        
        # Start with hourly data (prefix it too for consistency)
        result = self._prefix_columns(hourly_df, 'hourly_')
        self._log(f"  Base hourly features: {len(hourly_df)} rows, {len(hourly_df.columns)} columns")
        
        # Align daily features to hourly timestamps
        if daily_df is not None and len(daily_df) > 0:
            self._log(f"  Aligning daily features: {len(daily_df)} rows, {len(daily_df.columns)} columns")
            daily_aligned = self._forward_fill_to_frequency(
                daily_df, result.index, 'daily_'
            )
            result = pd.concat([result, daily_aligned], axis=1)
            self._log(f"    ✓ Added {len(daily_aligned.columns)} daily features")
        
        # Align weekly features to hourly timestamps
        if weekly_df is not None and len(weekly_df) > 0:
            self._log(f"  Aligning weekly features: {len(weekly_df)} rows, {len(weekly_df.columns)} columns")
            weekly_aligned = self._forward_fill_to_frequency(
                weekly_df, result.index, 'weekly_'
            )
            result = pd.concat([result, weekly_aligned], axis=1)
            self._log(f"    ✓ Added {len(weekly_aligned.columns)} weekly features")
        
        self._log(f"\n✓ Hourly synchronization complete: {len(result)} rows, {len(result.columns)} columns")
        
        return result
    
    def synchronize_daily(self, daily_df, weekly_df=None):
        """
        Synchronize features to daily frequency.
        
        Parameters:
        -----------
        daily_df : pd.DataFrame
            Daily features with datetime index
        weekly_df : pd.DataFrame, optional
            Weekly features with datetime index
        
        Returns:
        --------
        pd.DataFrame
            Synchronized daily dataframe with weekly features aligned
        """
        self._log("\n" + "="*80)
        self._log("SYNCHRONIZING TO DAILY FREQUENCY")
        self._log("="*80)
        
        # Start with daily data
        result = self._prefix_columns(daily_df, 'daily_')
        self._log(f"  Base daily features: {len(daily_df)} rows, {len(daily_df.columns)} columns")
        
        # Align weekly features to daily timestamps
        if weekly_df is not None and len(weekly_df) > 0:
            self._log(f"  Aligning weekly features: {len(weekly_df)} rows, {len(weekly_df.columns)} columns")
            weekly_aligned = self._forward_fill_to_frequency(
                weekly_df, result.index, 'weekly_'
            )
            result = pd.concat([result, weekly_aligned], axis=1)
            self._log(f"    ✓ Added {len(weekly_aligned.columns)} weekly features")
        
        self._log(f"\n✓ Daily synchronization complete: {len(result)} rows, {len(result.columns)} columns")
        
        return result
    
    def synchronize_weekly(self, weekly_df):
        """
        Prepare weekly features (no alignment needed, just prefix).
        
        Parameters:
        -----------
        weekly_df : pd.DataFrame
            Weekly features with datetime index
        
        Returns:
        --------
        pd.DataFrame
            Weekly dataframe with prefixed columns
        """
        self._log("\n" + "="*80)
        self._log("PREPARING WEEKLY FEATURES")
        self._log("="*80)
        
        result = self._prefix_columns(weekly_df, 'weekly_')
        self._log(f"  Weekly features: {len(weekly_df)} rows, {len(weekly_df.columns)} columns")
        self._log(f"\n✓ Weekly preparation complete")
        
        return result
    
    def synchronize(self, hourly_df, daily_df=None, weekly_df=None):
        """
        Synchronize all features across all time frequencies.
        
        Parameters:
        -----------
        hourly_df : pd.DataFrame
            Hourly features with datetime index
        daily_df : pd.DataFrame, optional
            Daily features with datetime index
        weekly_df : pd.DataFrame, optional
            Weekly features with datetime index
        
        Returns:
        --------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            (synced_hourly, synced_daily, synced_weekly)
        """
        self._log("\n" + "="*80)
        self._log("TEMPORAL SYNCHRONIZER - MULTI-FREQUENCY ALIGNMENT")
        self._log("="*80)
        
        # Synchronize to each frequency
        self.hourly_synced = self.synchronize_hourly(hourly_df, daily_df, weekly_df)
        
        if daily_df is not None:
            self.daily_synced = self.synchronize_daily(daily_df, weekly_df)
        else:
            self.daily_synced = None
        
        if weekly_df is not None:
            self.weekly_synced = self.synchronize_weekly(weekly_df)
        else:
            self.weekly_synced = None
        
        # Summary
        self._log("\n" + "="*80)
        self._log("SYNCHRONIZATION SUMMARY")
        self._log("="*80)
        self._log(f"✓ Hourly: {len(self.hourly_synced)} rows × {len(self.hourly_synced.columns)} columns")
        if self.daily_synced is not None:
            self._log(f"✓ Daily: {len(self.daily_synced)} rows × {len(self.daily_synced.columns)} columns")
        if self.weekly_synced is not None:
            self._log(f"✓ Weekly: {len(self.weekly_synced)} rows × {len(self.weekly_synced.columns)} columns")
        
        return self.hourly_synced, self.daily_synced, self.weekly_synced
    
    def save_synchronized_data(self, output_dir='synchronized_data'):
        """
        Save synchronized dataframes to CSV files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save synchronized data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.hourly_synced is not None:
            path = os.path.join(output_dir, 'synced_hourly.csv')
            self.hourly_synced.to_csv(path)
            self._log(f"✓ Saved: {path}")
        
        if self.daily_synced is not None:
            path = os.path.join(output_dir, 'synced_daily.csv')
            self.daily_synced.to_csv(path)
            self._log(f"✓ Saved: {path}")
        
        if self.weekly_synced is not None:
            path = os.path.join(output_dir, 'synced_weekly.csv')
            self.weekly_synced.to_csv(path)
            self._log(f"✓ Saved: {path}")


def create_sample_features():
    """
    Create sample hourly, daily, and weekly features for testing.
    
    Returns:
    --------
    tuple of (hourly_df, daily_df, weekly_df)
    """
    # Generate sample hourly data (10 days, 6 hours/day)
    hourly_dates = pd.date_range('2024-01-01 09:00', periods=60, freq='h')
    hourly_df = pd.DataFrame({
        'Close': np.random.uniform(20000, 22000, len(hourly_dates)),
        'Volume': np.random.randint(1000000, 5000000, len(hourly_dates)),
        'Returns': np.random.normal(0.001, 0.02, len(hourly_dates)),
        'Volatility': np.random.uniform(0.01, 0.05, len(hourly_dates))
    }, index=hourly_dates)
    
    # Generate sample daily data
    daily_dates = pd.date_range('2024-01-01', periods=10, freq='D')
    daily_df = pd.DataFrame({
        'Close': np.random.uniform(20000, 22000, len(daily_dates)),
        'SMA_20': np.random.uniform(19000, 21000, len(daily_dates)),
        'RSI': np.random.uniform(30, 70, len(daily_dates)),
        'MACD': np.random.normal(0, 100, len(daily_dates))
    }, index=daily_dates)
    
    # Generate sample weekly data
    weekly_dates = pd.date_range('2024-01-01', periods=2, freq='W-MON')
    weekly_df = pd.DataFrame({
        'Close': np.random.uniform(20000, 22000, len(weekly_dates)),
        'SMA_50': np.random.uniform(19000, 21000, len(weekly_dates)),
        'Weekly_Return': np.random.normal(0.01, 0.05, len(weekly_dates)),
        'Trend': np.random.choice(['UP', 'DOWN'], len(weekly_dates))
    }, index=weekly_dates)
    
    return hourly_df, daily_df, weekly_df


def main():
    """Demo and test the Temporal Synchronizer"""
    
    print("="*80)
    print("TEMPORAL SYNCHRONIZER - DEMO")
    print("="*80)
    
    # Create sample features at different frequencies
    print("\nGenerating sample multi-frequency features...")
    hourly_df, daily_df, weekly_df = create_sample_features()
    
    print(f"  Hourly: {len(hourly_df)} rows")
    print(f"  Daily: {len(daily_df)} rows")
    print(f"  Weekly: {len(weekly_df)} rows")
    
    # Initialize synchronizer
    ts = TemporalSynchronizer(verbose=True)
    
    # Synchronize all frequencies
    synced_hourly, synced_daily, synced_weekly = ts.synchronize(
        hourly_df, daily_df, weekly_df
    )
    
    # Display sample synchronized data
    print("\n" + "="*80)
    print("SAMPLE SYNCHRONIZED HOURLY DATA")
    print("="*80)
    print("\nFirst 10 rows:")
    print(synced_hourly.head(10))
    
    print("\n" + "="*80)
    print("SAMPLE SYNCHRONIZED DAILY DATA")
    print("="*80)
    print("\nFirst 5 rows:")
    print(synced_daily.head() if synced_daily is not None else "N/A")
    
    # Save synchronized data
    print("\n" + "="*80)
    print("SAVING SYNCHRONIZED DATA")
    print("="*80)
    ts.save_synchronized_data('synchronized_data')
    
    print("\n" + "="*80)
    print("✓ TEMPORAL SYNCHRONIZATION COMPLETE!")
    print("="*80)
    print("\nUsage Example:")
    print("""
from temporal_synchronizer import TemporalSynchronizer

# Load your features
hourly_df = pd.read_csv('hourly_features.csv', index_col=0, parse_dates=True)
daily_df = pd.read_csv('daily_features.csv', index_col=0, parse_dates=True)
weekly_df = pd.read_csv('weekly_features.csv', index_col=0, parse_dates=True)

# Synchronize
ts = TemporalSynchronizer()
synced_hourly, synced_daily, synced_weekly = ts.synchronize(
    hourly_df, daily_df, weekly_df
)

# Use synced_hourly for modeling with all timeframe features aligned
""")


if __name__ == '__main__':
    main()
