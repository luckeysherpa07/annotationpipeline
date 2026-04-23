#!/usr/bin/env python
"""
Quick test script for the annotation pipeline.
Run this to test different scenarios without modifying main.py.
"""

from pathlib import Path
import sys

# Add parent directory to path so we can import annotation_feature
sys.path.insert(0, str(Path(__file__).parent))

from annotation_feature.pipeline import (
    run,
    run_audio,
    run_depth,
    run_event,
    run_ir,
    run_late_fusion,
)


def _confirm(prompt: str = "Continue? (yes/no): ") -> bool:
    return input(prompt).strip().lower() == "yes"


def _run_all_pipelines(test_mode: bool, skip_api: bool) -> None:
    print("[1/6] RGB pipeline...")
    run(test_mode=test_mode, skip_api=skip_api)

    print("[2/6] EVENT pipeline...")
    run_event(test_mode=test_mode, skip_api=skip_api)

    print("[3/6] DEPTH pipeline...")
    run_depth(test_mode=test_mode, skip_api=skip_api)

    print("[4/6] IR pipeline...")
    run_ir(test_mode=test_mode, skip_api=skip_api)

    print("[5/6] AUDIO pipeline...")
    run_audio(test_mode=test_mode, skip_api=skip_api)

    print("[6/6] Late fusion...")
    fused_results = run_late_fusion()
    print(f"Fused {len(fused_results)} samples into fused_qa_results.json")


def main():
    print("\n" + "=" * 60)
    print("BATCH + PARALLEL ANNOTATION PIPELINE TEST RUNNER")
    print("=" * 60)
    print("RGB: Single mega-prompt per pair with QA generation")
    print("EVENT: Single mega-prompt per pair with caption, question, and answer generation")
    print("DEPTH: Single mega-prompt per pair with caption, question, and answer generation")
    print("IR: Single mega-prompt per pair with caption, question, and answer generation")
    print("AUDIO: Single mega-prompt per audio file with QA generation (night only)")
    print("LATE FUSION: Post-process modality captions into fused scene summaries")
    print("=" * 60)

    while True:
        print("\nChoose a test to run:\n")
        print("--- ALL PIPELINES ---")
        print("1. Test ALL pipelines (no API calls, using DEMO data) + late fusion")
        print("2. Test ALL pipelines on 1 pair/file (with real Gemini API calls) + late fusion")
        print("3. Run ALL pipelines on all videos/files (production) + late fusion")
        print("\n--- RGB PIPELINE ---")
        print("4. Test RGB preprocessing + batch pipeline (no API calls, using DEMO data)")
        print("5. Test RGB batch pipeline on 1 pair (with real Gemini API calls)")
        print("6. Run RGB batch pipeline on all videos (production)")
        print("\n--- EVENT PIPELINE ---")
        print("7. Test EVENT preprocessing + batch pipeline (no API calls, demo Q&A)")
        print("8. Test EVENT batch pipeline on 1 pair with Q&A (with real Gemini API calls)")
        print("9. Run EVENT batch pipeline on all videos with Q&A (production)")
        print("\n--- DEPTH PIPELINE ---")
        print("10. Test DEPTH preprocessing + batch pipeline (no API calls, demo Q&A)")
        print("11. Test DEPTH batch pipeline on 1 pair with Q&A (with real Gemini API calls)")
        print("12. Run DEPTH batch pipeline on all videos with Q&A (production)")
        print("\n--- IR PIPELINE ---")
        print("13. Test IR preprocessing + batch pipeline (no API calls, demo Q&A)")
        print("14. Test IR batch pipeline on 1 pair with Q&A (with real Gemini API calls)")
        print("15. Run IR batch pipeline on all videos with Q&A (production)")
        print("\n--- AUDIO PIPELINE ---")
        print("16. Test AUDIO pipeline on 1 file (no API calls, demo Q&A)")
        print("17. Test AUDIO pipeline on 1 file with real Gemini API calls")
        print("18. Run AUDIO pipeline on all files (production)")
        print("\n--- LATE FUSION ---")
        print("19. Run late fusion on existing modality JSON results")
        print("\n20. Exit")

        choice = input("\nEnter choice (1-20): ").strip()

        if choice == "1":
            print("\n" + "-" * 60)
            print("Running: ALL pipelines test (using DEMO data) + late fusion")
            print("-" * 60)
            print("skip_api=True -> demo outputs for all modality pipelines\n")
            _run_all_pipelines(test_mode=True, skip_api=True)

        elif choice == "2":
            print("\n" + "-" * 60)
            print("Running: ALL pipelines on 1 pair/file (real Gemini API calls) + late fusion")
            print("WARNING: This will use Gemini API quota across all modalities!")
            print("-" * 60)
            if _confirm():
                _run_all_pipelines(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "3":
            print("\n" + "-" * 60)
            print("Running: ALL pipelines on all videos/files (production) + late fusion")
            print("WARNING: This will use Gemini API quota across all modalities!")
            print("-" * 60)
            if _confirm():
                _run_all_pipelines(test_mode=False, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "4":
            print("\n" + "-" * 60)
            print("Running: RGB batch pipeline test (using DEMO data)")
            print("-" * 60)
            print("1 Gemini call per pair (all 12 types in single request)")
            print("skip_api=True -> results from DEMO_RESULT\n")
            run(test_mode=True, skip_api=True)

        elif choice == "5":
            print("\n" + "-" * 60)
            print("Running: RGB batch pipeline on 1 pair (real Gemini API calls)")
            print("WARNING: This will use Gemini API quota!")
            print("Uses gemini-3-flash-preview with max 6 frames per call")
            print("-" * 60)
            if _confirm():
                run(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "6":
            print("\n" + "-" * 60)
            print("Running: RGB batch pipeline on all videos (production)")
            print("WARNING: This will use Gemini API quota for each video!")
            print("Parallel execution: up to 3 pairs concurrently, 4-second spacing")
            print("-" * 60)
            if _confirm():
                run(test_mode=False)
            else:
                print("Cancelled.")

        elif choice == "7":
            print("\n" + "-" * 60)
            print("Running: EVENT batch pipeline test (demo Q&A)")
            print("-" * 60)
            print("1 Gemini call per event pair (caption, question, and answer generation)")
            print("skip_api=True -> demo Q&A\n")
            run_event(test_mode=True, skip_api=True)

        elif choice == "8":
            print("\n" + "-" * 60)
            print("Running: EVENT batch pipeline on 1 pair (real Gemini API calls)")
            print("WARNING: This will use Gemini API quota!")
            print("Uses gemini-3-flash-preview with max 6 frames per call")
            print("-" * 60)
            if _confirm():
                run_event(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "9":
            print("\n" + "-" * 60)
            print("Running: EVENT batch pipeline on all videos (production)")
            print("WARNING: This will use Gemini API quota for each video!")
            print("Parallel execution: up to 3 pairs concurrently, 4-second spacing")
            print("-" * 60)
            if _confirm():
                run_event(test_mode=False)
            else:
                print("Cancelled.")

        elif choice == "10":
            print("\n" + "-" * 60)
            print("Running: DEPTH batch pipeline test (demo Q&A)")
            print("-" * 60)
            print("1 Gemini call per depth pair (caption, question, and answer generation)")
            print("skip_api=True -> demo Q&A\n")
            run_depth(test_mode=True, skip_api=True)

        elif choice == "11":
            print("\n" + "-" * 60)
            print("Running: DEPTH batch pipeline on 1 pair (real Gemini API calls)")
            print("WARNING: This will use Gemini API quota!")
            print("Uses gemini-3-flash-preview with max 6 frames per call")
            print("-" * 60)
            if _confirm():
                run_depth(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "12":
            print("\n" + "-" * 60)
            print("Running: DEPTH batch pipeline on all videos (production)")
            print("WARNING: This will use Gemini API quota for each video!")
            print("Parallel execution: up to 3 pairs concurrently, 4-second spacing")
            print("-" * 60)
            if _confirm():
                run_depth(test_mode=False)
            else:
                print("Cancelled.")

        elif choice == "13":
            print("\n" + "-" * 60)
            print("Running: IR batch pipeline test (demo Q&A)")
            print("-" * 60)
            print("1 Gemini call per IR pair (caption, question, and answer generation)")
            print("skip_api=True -> demo Q&A\n")
            run_ir(test_mode=True, skip_api=True)

        elif choice == "14":
            print("\n" + "-" * 60)
            print("Running: IR batch pipeline on 1 pair (real Gemini API calls)")
            print("WARNING: This will use Gemini API quota!")
            print("Uses gemini-3-flash-preview with max 6 frames per call")
            print("-" * 60)
            if _confirm():
                run_ir(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "15":
            print("\n" + "-" * 60)
            print("Running: IR batch pipeline on all videos (production)")
            print("WARNING: This will use Gemini API quota for each video!")
            print("Parallel execution: up to 3 pairs concurrently, 4-second spacing")
            print("-" * 60)
            if _confirm():
                run_ir(test_mode=False)
            else:
                print("Cancelled.")

        elif choice == "16":
            print("\n" + "-" * 60)
            print("Running: AUDIO pipeline test on 1 file (using demo data)")
            print("-" * 60)
            print("Processes night audio only (day/night do not affect audio)")
            print("Single mega-prompt with 10 audio annotation types")
            print("skip_api=True -> results from AUDIO_DEMO_RESULT\n")
            run_audio(test_mode=True, skip_api=True)

        elif choice == "17":
            print("\n" + "-" * 60)
            print("Running: AUDIO pipeline on 1 file (real Gemini API calls)")
            print("WARNING: This will use Gemini API quota!")
            print("Uses gemini-2-flash with audio file input")
            print("-" * 60)
            if _confirm():
                run_audio(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "18":
            print("\n" + "-" * 60)
            print("Running: AUDIO pipeline on all files (production)")
            print("WARNING: This will use Gemini API quota for each audio file!")
            print("Parallel execution: up to 3 files concurrently, 4-second spacing")
            print("Audio types: 10 (sound recognition, speech, music, etc.)")
            print("-" * 60)
            if _confirm():
                run_audio(test_mode=False)
            else:
                print("Cancelled.")

        elif choice == "19":
            print("\n" + "-" * 60)
            print("Running: late fusion on existing modality JSON results")
            print("-" * 60)
            print("This step reads the current RGB, IR, event, audio, and depth result files.")
            print("It writes fused scene summaries to fused_qa_results.json.\n")
            fused_results = run_late_fusion()
            print(f"Fused {len(fused_results)} samples into fused_qa_results.json")

        elif choice == "20":
            print("\nExiting.")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
