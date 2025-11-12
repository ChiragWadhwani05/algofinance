"""
TEST KALMAN FILTER: Standalone test to see Kalman sensor in action

This script runs ONLY the Kalman filter sensor on your data.
Use this to understand how it works before integrating into the full pipeline.

Usage:
    1. Make sure you have pykalman installed: pip install pykalman
    2. Edit TEST_CSV below to point to your cleaned data file
    3. Run: python3 test_kalman.py
"""

from models.kalman_trend import KalmanTrendSensor
import os

# =============================================================================
# CONFIGURATION - Edit these 2 lines
# =============================================================================
TEST_CSV = "Nifty_50_Cleaned.csv"  # Your cleaned CSV file
PRICE_COL = "Adj Close"                # Price column name

# Kalman tuning parameters
OBSERVATION_COV = 1.0    # Measurement noise (higher = smoother)
TRANSITION_COV = 0.01    # Process noise (higher = more responsive)
# =============================================================================


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║                    KALMAN FILTER SENSOR TEST                            ║
║                                                                          ║
║  Tests Sensor 3 (Kalman Trend Detection) in isolation                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configuration:")
    print(f"  Input CSV: {TEST_CSV}")
    print(f"  Price column: {PRICE_COL}")
    print(f"  Observation covariance: {OBSERVATION_COV}")
    print(f"  Transition covariance: {TRANSITION_COV}")
    print("\n" + "="*80)
    
    # Check if file exists
    if not os.path.exists(TEST_CSV):
        print(f"\n❌ ERROR: File not found: {TEST_CSV}")
        print("\nOptions:")
        print("  1. Run master_pipeline.py first to generate cleaned CSV")
        print("  2. Update TEST_CSV variable to point to your cleaned data")
        return
    
    # Initialize sensor
    sensor = KalmanTrendSensor(
        observation_covariance=OBSERVATION_COV,
        transition_covariance=TRANSITION_COV
    )
    
    # Run the sensor
    try:
        (sensor
         .load_data(TEST_CSV, price_col=PRICE_COL)
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
        print("✅ KALMAN SENSOR TEST COMPLETE!")
        print("="*80)
        print("\nOutput Files Created:")
        print("  📄 sensor3_kalman_trend.csv       - UP/DOWN trend predictions")
        print("  📄 sensor3_kalman_summary.txt     - Summary statistics")
        print("  📊 sensor3_kalman_trend.png       - Visualization")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("\n1. View the visualization:")
        print("   Open sensor3_kalman_trend.png to see:")
        print("   - Kalman smoothed price vs actual price")
        print("   - Velocity (trend strength) over time")
        print("   - UP/DOWN regime timeline")
        
        print("\n2. Check the summary:")
        print("   cat sensor3_kalman_summary.txt")
        
        print("\n3. Compare with MRS sensor:")
        print("   - Kalman detects trends based on velocity")
        print("   - MRS detects regime switches (BULL/BEAR)")
        print("   - They capture different aspects of the market")
        
        print("\n4. Integration options:")
        print("   Option A: Replace MRS with Kalman (keep 4 regimes)")
        print("   Option B: Add Kalman as 3rd sensor (expand to 8 regimes)")
        print("   Read KALMAN_INTEGRATION_GUIDE.md for details")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED!")
        print("="*80)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("  1. Install pykalman: pip install pykalman")
        print("  2. Check that TEST_CSV points to a valid cleaned CSV file")
        print("  3. Verify PRICE_COL matches your CSV column name")


if __name__ == "__main__":
    main()
