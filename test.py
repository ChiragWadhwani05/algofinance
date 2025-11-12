"""
SIMPLE TEST SCRIPT
Quick way to test the pipeline on any CSV file
"""

# Just change these two lines:
TEST_CSV = "Nifty_50.csv"              # ← Your CSV file
OUTPUT_NAME = "test_run"                # ← Results folder name

# That's it! Now run: python3 test.py

if __name__ == "__main__":
    import pipeline_config as config
    from master_pipeline import MasterPipeline
    
    # Override config
    config.INPUT_CSV_FILE = TEST_CSV
    config.OUTPUT_FOLDER_NAME = OUTPUT_NAME
    
    print(f"\n🧪 Testing pipeline with:")
    print(f"   Input: {TEST_CSV}")
    print(f"   Output: results/{OUTPUT_NAME}/\n")
    
    # Run
    pipeline = MasterPipeline(config)
    success = pipeline.run()
    
    if success:
        print(f"\n✅ Test successful!")
        print(f"📁 Check: results/{OUTPUT_NAME}/")
    else:
        print(f"\n❌ Test failed! Check error messages above.")
