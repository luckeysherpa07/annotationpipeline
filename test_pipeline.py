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
    print("ANNOTATION PIPELINE TEST RUNNER")
    print("=" * 60)
    
    while True:
        print("\nChoose a test to run:\n")
        print("1. Test preprocessing only (no API calls, no cost)")
        print("2. Test full pipeline on 1 pair (with real API calls)")
        print("3. Run full pipeline on all videos (production)")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n" + "-" * 60)
            print("Running: Preprocessing only (using DEMO data)")
            print("-" * 60)
            run(test_mode=True, skip_api=True)
            
        elif choice == "2":
            print("\n" + "-" * 60)
            print("Running: Full pipeline on 1 pair (REAL API CALLS)")
            print("⚠️  WARNING: This will use OpenAI API credits!")
            print("-" * 60)
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm == "yes":
                run(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")
            
        elif choice == "3":
            print("\n" + "-" * 60)
            print("Running: Full pipeline on ALL videos (PRODUCTION)")
            print("⚠️  WARNING: This will use OpenAI API credits for each video!")
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
