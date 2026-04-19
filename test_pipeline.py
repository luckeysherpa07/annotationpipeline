#!/usr/bin/env python
"""
Quick test script for the annotation pipeline.
Run this to test different scenarios without modifying main.py
"""

from pathlib import Path
import sys

# Add parent directory to path so we can import annotation_feature
sys.path.insert(0, str(Path(__file__).parent))

from annotation_feature.pipeline import run


def main():
    print("\n" + "=" * 60)
    print("BATCH + PARALLEL ANNOTATION PIPELINE TEST RUNNER")
    print("=" * 60)
    print("NEW: Single mega-prompt per pair, concurrent execution (3 pairs max)")
    print("=" * 60)
    
    while True:
        print("\nChoose a test to run:\n")
        print("1. Test preprocessing + batch pipeline (no API calls, using DEMO data)")
        print("2. Test batch pipeline on 1 pair (with real Gemini API calls)")
        print("3. Run batch pipeline on all videos (production)")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n" + "-" * 60)
            print("Running: Batch pipeline test (using DEMO data)")
            print("-" * 60)
            print("✓ 1 Gemini call per pair (all 12 types in single request)")
            print("✓ skip_api=True → results from DEMO_RESULT\n")
            run(test_mode=True, skip_api=True)
            
        elif choice == "2":
            print("\n" + "-" * 60)
            print("Running: Batch pipeline on 1 pair (REAL GEMINI API CALLS)")
            print("⚠️  WARNING: This will use Gemini API quota!")
            print("ℹ️  Uses gemini-1.5-flash with max 6 frames per call")
            print("-" * 60)
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm == "yes":
                run(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")
            
        elif choice == "3":
            print("\n" + "-" * 60)
            print("Running: Batch pipeline on ALL videos (PRODUCTION)")
            print("⚠️  WARNING: This will use Gemini API quota for each video!")
            print("ℹ️  Parallel execution: up to 3 pairs concurrently, 4-sec spacing")
            print("-" * 60)
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm == "yes":
                run(test_mode=False)
            else:
                print("Cancelled.")
            
        elif choice == "4":
            print("\nExiting.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
