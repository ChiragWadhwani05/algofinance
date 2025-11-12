"""
Data Cleaning Module - Flexible CSV Cleaner for Any Market Data
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class DataCleaner:
    """Clean and prepare raw market data for regime detection"""
    
    def __init__(self, date_col='Date', price_col='Adj Close', 
                 skip_rows=None, date_format=None, fill_method='ffill'):
        """
        Initialize data cleaner
        
        Parameters:
        -----------
        date_col : str
            Name of date column in CSV
        price_col : str
            Name of price column to analyze
        skip_rows : list or None
            Row numbers to skip when reading CSV
        date_format : str or None
            Date format string (None for auto-detection)
        fill_method : str
            Method to fill missing values ('ffill' or 'bfill')
        """
        self.date_col = date_col
        self.price_col = price_col
        self.skip_rows = skip_rows
        self.date_format = date_format
        self.fill_method = fill_method
        
    def load_raw_data(self, filepath):
        """Load raw CSV data"""
        print("="*80)
        print("DATA CLEANING: LOADING RAW DATA")
        print("="*80)
        
        try:
            # Load with optional row skipping
            if self.skip_rows:
                df = pd.read_csv(filepath, header=0, skiprows=self.skip_rows)
                print(f"✓ Loaded CSV (skipped rows: {self.skip_rows})")
            else:
                df = pd.read_csv(filepath, header=0)
                print(f"✓ Loaded CSV")
                
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            self.raw_data = df
            return self
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {e}")
    
    def prepare_datetime_index(self):
        """Convert date column to datetime index"""
        print("\n" + "="*80)
        print("PREPARING DATETIME INDEX")
        print("="*80)
        
        if self.date_col not in self.raw_data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in CSV")
        
        # Convert to datetime
        if self.date_format:
            self.raw_data[self.date_col] = pd.to_datetime(
                self.raw_data[self.date_col], 
                format=self.date_format
            )
        else:
            self.raw_data[self.date_col] = pd.to_datetime(self.raw_data[self.date_col])
        
        # Set as index and sort
        self.raw_data = self.raw_data.set_index(self.date_col)
        self.raw_data = self.raw_data.sort_index()
        
        print(f"✓ Date range: {self.raw_data.index.min()} to {self.raw_data.index.max()}")
        print(f"✓ Total days: {len(self.raw_data)}")
        
        return self
    
    def clean_price_data(self):
        """Clean and validate price column"""
        print("\n" + "="*80)
        print("CLEANING PRICE DATA")
        print("="*80)
        
        if self.price_col not in self.raw_data.columns:
            raise ValueError(f"Price column '{self.price_col}' not found in CSV")
        
        # Keep only price column
        self.data = self.raw_data[[self.price_col]].copy()
        
        # Convert to numeric (coerce errors to NaN)
        self.data[self.price_col] = pd.to_numeric(
            self.data[self.price_col], 
            errors='coerce'
        )
        
        # Count NaNs
        nan_count = self.data[self.price_col].isna().sum()
        print(f"✓ Converted to numeric")
        print(f"  NaN values found: {nan_count}")
        
        return self
    
    def handle_missing_data(self):
        """Fill missing values and create continuous business day index"""
        print("\n" + "="*80)
        print("HANDLING MISSING DATA")
        print("="*80)
        
        # Create continuous business day range
        all_bdays = pd.bdate_range(
            start=self.data.index.min(), 
            end=self.data.index.max()
        )
        
        # Reindex to business days
        self.data = self.data.reindex(all_bdays)
        print(f"✓ Reindexed to business days: {len(self.data)} days")
        
        # Fill missing values
        if self.fill_method == 'ffill':
            self.data = self.data.fillna(method='ffill')
            print(f"✓ Forward-filled missing values")
        elif self.fill_method == 'bfill':
            self.data = self.data.fillna(method='bfill')
            print(f"✓ Backward-filled missing values")
        
        # Drop any remaining NaNs
        before = len(self.data)
        self.data = self.data.dropna(subset=[self.price_col])
        after = len(self.data)
        
        if before != after:
            print(f"⚠ Dropped {before - after} rows with remaining NaNs")
        
        return self
    
    def calculate_returns(self):
        """Calculate log returns for modeling"""
        print("\n" + "="*80)
        print("CALCULATING RETURNS")
        print("="*80)
        
        # Calculate log returns
        self.data['Log Returns'] = np.log(
            self.data[self.price_col] / self.data[self.price_col].shift(1)
        )
        
        # Drop first row (NaN return)
        self.data = self.data.dropna(subset=['Log Returns'])
        
        print(f"✓ Calculated log returns")
        print(f"✓ Final data shape: {self.data.shape}")
        print(f"  Start date: {self.data.index.min()}")
        print(f"  End date: {self.data.index.max()}")
        print(f"  Total days: {len(self.data)}")
        
        return self
    
    def save_cleaned_data(self, output_path):
        """Save cleaned data to CSV"""
        print("\n" + "="*80)
        print("SAVING CLEANED DATA")
        print("="*80)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save
        self.data.to_csv(output_path)
        print(f"✓ Cleaned data saved to: {output_path}")
        
        # Show sample
        print("\nCleaned Data Sample (last 5 rows):")
        print(self.data.tail())
        
        return self
    
    def get_cleaned_data(self):
        """Return cleaned DataFrame"""
        return self.data
    
    def clean(self, input_file, output_file):
        """
        Main cleaning pipeline
        
        Parameters:
        -----------
        input_file : str
            Path to raw CSV file
        output_file : str
            Path to save cleaned CSV
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        print("\n" + "="*80)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*80)
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        
        (self
         .load_raw_data(input_file)
         .prepare_datetime_index()
         .clean_price_data()
         .handle_missing_data()
         .calculate_returns()
         .save_cleaned_data(output_file)
        )
        
        print("\n" + "="*80)
        print("✅ DATA CLEANING COMPLETE!")
        print("="*80)
        
        return self.data


def clean_data(input_csv, output_csv, date_col='Date', price_col='Adj Close',
               skip_rows=None, date_format=None, fill_method='ffill'):
    """
    Convenience function to clean data in one call
    
    Parameters:
    -----------
    input_csv : str
        Path to raw CSV file
    output_csv : str
        Path to save cleaned CSV
    date_col : str
        Name of date column
    price_col : str
        Name of price column
    skip_rows : list or None
        Rows to skip when reading
    date_format : str or None
        Date format string
    fill_method : str
        Method to fill missing values
        
    Returns:
    --------
    pd.DataFrame
        Cleaned data
    """
    cleaner = DataCleaner(
        date_col=date_col,
        price_col=price_col,
        skip_rows=skip_rows,
        date_format=date_format,
        fill_method=fill_method
    )
    
    return cleaner.clean(input_csv, output_csv)


# Example usage
if __name__ == "__main__":
    # Test the cleaner
    cleaned_data = clean_data(
        input_csv="../Nifty_50.csv",
        output_csv="test_cleaned.csv",
        date_col="Date",
        price_col="Adj Close",
        skip_rows=[1]
    )
    print("\n✅ Test successful!")
