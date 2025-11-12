"""
Configuration file for the Regime Detection Pipeline
Edit this file to specify which CSV file to process
"""

# ============================================================================
# INPUT CONFIGURATION
# ============================================================================

# Path to your raw CSV file (can be absolute or relative)
# Examples:
#   - "Nifty_50.csv"
#   - "../Nifty_IT.csv"
#   - "/full/path/to/your/data.csv"

INPUT_CSV_FILE = "Nifty_50.csv"

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Name for the output folder (will be created if doesn't exist)
# Results will be saved in: CombineApproach/results/{OUTPUT_FOLDER_NAME}/
OUTPUT_FOLDER_NAME = "nifty50_analysis"

# ============================================================================
# DATA CLEANING CONFIGURATION
# ============================================================================

# Column names in your CSV
DATE_COLUMN = "Date"           # Column containing dates
PRICE_COLUMN = "Adj Close"     # Column containing prices to analyze

# Date format in your CSV (if needed)
# Examples: "%Y-%m-%d", "%d-%m-%Y", etc.
# Set to None for auto-detection
DATE_FORMAT = None

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# MRS (Trend Sensor) Configuration
TRAIN_RATIO = 0.70             # 70% training, 30% testing
N_REGIMES = 2                  # 2 regimes: BULL/BEAR

# GARCH (Volatility Sensor) Configuration
TRAIN_END_DATE = '2020-12-31'  # End of training period
TEST_START_DATE = '2021-01-01' # Start of test period
VOLATILITY_PERCENTILE = 75     # 75th percentile for High/Low threshold

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# Skip problematic rows during CSV loading (row numbers to skip, 0-indexed)
# Example: [1] will skip the second row
SKIP_ROWS = [1]  # Set to None or [] if not needed

# Fill missing data method
FILL_METHOD = 'ffill'  # 'ffill' (forward fill) or 'bfill' (backward fill)

# ============================================================================
# VISUALIZATION OPTIONS
# ============================================================================

# DPI for saved plots (higher = better quality but larger files)
PLOT_DPI = 300

# Figure sizes
MRS_FIGSIZE = (14, 8)
GARCH_FIGSIZE = (14, 12)
AGGREGATOR_FIGSIZE = (16, 12)
