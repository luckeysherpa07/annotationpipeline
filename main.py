#!/usr/bin/env python
"""
Quick test script for the annotation pipeline.
Run this to test different scenarios without modifying main.py.
"""

from pathlib import Path
import sys

# Add parent directory to path so we can import annotation_feature
sys.path.insert(0, str(Path(__file__).parent))

from annotation_feature.cli.actions.aligned_rgb import build_aligned_rgb_actions
from annotation_feature.cli.actions.aligned_event import build_aligned_event_actions
from annotation_feature.cli.actions.aligned_ir import build_aligned_ir_actions
from annotation_feature.cli.actions.aligned_audio import build_aligned_audio_actions
from annotation_feature.cli.actions.aligned_marigold import build_aligned_marigold_actions
from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions
from annotation_feature.cli.actions.aligned_choices import (
    REGISTERED_MENU_CHOICE_ORDER,
    REGISTERED_MENU_SECTION_ORDER,
)
from annotation_feature.pipeline import (
    run,
    run_audio,
    run_audio_repair,
    run_depth,
    run_event,
    run_ir,
    run_aligned_marigold_depth_estimation,
    run_marigold_depth_estimation,
    run_marigold_ir_depth_estimation,
    run_marigold_depth_qa,
    run_late_fusion,
    run_multimodal_qa_pipeline,
    run_multimodal_qa_verifier,
    run_task_slicing,
    run_segmented_pipeline,
)
from annotation_feature.reasoning import (
    normalize_all_modalities,
    run_export_grouped_qa,
    run_export_segmented_grouped_qa,
    run_export_segmented_normalized_evidence_csv,
    run_group_evidence,
)
from annotation_feature.segmented_pipeline import estimate_segmented_work
from annotation_feature.pipeline.modalities.marigold import (
    list_cached_ir_night_folders,
    list_cached_rgb_folders,
)
from annotation_feature.temporal_alignment import (
    export_check_mailbox_day_rgb_event_optical_flow_alignment,
    export_day_night_rgb_event_depth_ir_alignment_grids,
    run_and_export_aligned_rgb_with_audio_segments,
    run_and_export_all_aligned_dataset_segments,
    run_and_export_all_rgb_event_dtw_alignments,
    run_and_export_all_rgb_event_dtw_with_audio_alignments,
    run_and_export_check_mailbox_day_rgb_audio_cross_correlation_alignment,
    run_and_export_check_mailbox_day_rgb_event_dtw_alignment,
    run_and_export_check_mailbox_day_rgb_event_dtw_with_audio_alignment,
    run_and_export_check_mailbox_day_rgb_event_feature_alignment,
    run_and_export_cut_carrot_aligned_dataset_segments,
    run_and_export_source_rgb_with_audio_segments_for_aligned_dataset,
    run_check_mailbox_day_rgb_event_optical_flow_alignment,
    run_day_night_temporal_alignment,
)


ALIGNED_QA_OUTPUT_DIR = Path("qa_pairs/aligned")
ALIGNED_QA_OUTPUT_FILES = {
    "rgb": ALIGNED_QA_OUTPUT_DIR / "rgb_qa_results_aligned.json",
    "event": ALIGNED_QA_OUTPUT_DIR / "event_qa_results_aligned.json",
    "ir": ALIGNED_QA_OUTPUT_DIR / "ir_qa_results_aligned.json",
    "audio": ALIGNED_QA_OUTPUT_DIR / "audio_qa_results_aligned.json",
    "marigold_depth": ALIGNED_QA_OUTPUT_DIR / "marigold_depth_qa_results_aligned.json",
}
LEGACY_ALIGNED_QA_OUTPUT_FILES = {
    modality: Path("aligned_dataset") / output_path.name
    for modality, output_path in ALIGNED_QA_OUTPUT_FILES.items()
}


def _migrate_legacy_aligned_qa_results() -> None:
    """Move old aligned QA JSON files into qa_pairs/aligned when the new file is absent."""
    ALIGNED_QA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for modality, legacy_path in LEGACY_ALIGNED_QA_OUTPUT_FILES.items():
        output_path = ALIGNED_QA_OUTPUT_FILES[modality]
        if legacy_path.exists() and not output_path.exists():
            legacy_path.replace(output_path)
            print(f"Migrated {legacy_path} -> {output_path}")


MULTIMODAL_QA_MODEL_NAME = "gemini-3.1-flash-lite"
MULTIMODAL_QA_CANDIDATES_PATH = Path("outputs/implicit_multimodal_qa_candidates_gemini_v2.json")
MULTIMODAL_QA_VERIFIED_PATH = Path("outputs/implicit_multimodal_qa_verified_gemini_v2.json")


def _confirm(prompt: str = "Continue? (yes/no): ") -> bool:
    return input(prompt).strip().lower() == "yes"


REGISTERED_MENU_ACTIONS = {
    **build_aligned_rgb_actions(
        output_file=ALIGNED_QA_OUTPUT_FILES["rgb"],
        confirm=_confirm,
    ),
    **build_aligned_event_actions(
        output_file=ALIGNED_QA_OUTPUT_FILES["event"],
        confirm=_confirm,
    ),
    **build_aligned_ir_actions(
        output_file=ALIGNED_QA_OUTPUT_FILES["ir"],
        confirm=_confirm,
    ),
    **build_aligned_audio_actions(
        output_file=ALIGNED_QA_OUTPUT_FILES["audio"],
        confirm=_confirm,
    ),
    **build_aligned_marigold_actions(
        output_file=ALIGNED_QA_OUTPUT_FILES["marigold_depth"],
        confirm=_confirm,
    ),
    **build_aligned_qa_quality_actions(
        confirm=_confirm,
        output_dir="outputs",
    ),
}


def _print_registered_menu_sections(section_names: tuple[str, ...] = REGISTERED_MENU_SECTION_ORDER) -> None:
    """Print registry-backed menu actions in a stable order."""
    wanted_sections = set(section_names)
    printed_sections = set()
    for choice in REGISTERED_MENU_CHOICE_ORDER:
        action = REGISTERED_MENU_ACTIONS.get(choice)
        if action is None or action.section not in wanted_sections:
            continue
        if action.section not in printed_sections:
            print(f"\n--- {action.section} ---")
            printed_sections.add(action.section)
        print(f"{choice}. {action.title}")


def _select_cache_folder(dataset_folder: Path | str = "dataset") -> str | None:
    folders = list_cached_rgb_folders(Path(dataset_folder))
    if not folders:
        print("No cache folders found in dataset/.frames_cache")
        return None

    print("\nAvailable RGB cache folders:\n")
    for index, folder in enumerate(folders, start=1):
        print(f"{index}. {folder.name}")

    raw_choice = input(f"\nChoose cache folder (1-{len(folders)}): ").strip()
    try:
        selected_index = int(raw_choice)
    except ValueError:
        print("Invalid selection.")
        return None

    if selected_index < 1 or selected_index > len(folders):
        print("Invalid selection.")
        return None

    return folders[selected_index - 1].name


def _select_ir_night_cache_folder(dataset_folder: Path | str = "dataset") -> str | None:
    folders = list_cached_ir_night_folders(Path(dataset_folder))
    if not folders:
        print("No night IR cache folders found in dataset/.frames_cache_ir")
        return None

    print("\nAvailable night IR cache folders:\n")
    for index, folder in enumerate(folders, start=1):
        print(f"{index}. {folder.name}")

    raw_choice = input(f"\nChoose night IR cache folder (1-{len(folders)}): ").strip()
    try:
        selected_index = int(raw_choice)
    except ValueError:
        print("Invalid selection.")
        return None

    if selected_index < 1 or selected_index > len(folders):
        print("Invalid selection.")
        return None

    return folders[selected_index - 1].name


def _list_cache_frames(
    selected_folder: str,
    dataset_folder: Path | str = "dataset",
) -> list[Path]:
    folder_path = Path(dataset_folder) / ".frames_cache" / selected_folder
    nested_frames = sorted(folder_path.glob("day/frame_*.png")) + sorted(folder_path.glob("night/frame_*.png"))
    if nested_frames:
        return nested_frames
    return sorted(folder_path.glob("frame_*.png"))


def _select_cache_frame(
    selected_folder: str,
    dataset_folder: Path | str = "dataset",
) -> str | None:
    frames = _list_cache_frames(selected_folder, dataset_folder=dataset_folder)
    if not frames:
        print(f"No frame files found in dataset/.frames_cache/{selected_folder}")
        return None

    print(f"\nAvailable RGB frames in {selected_folder}:\n")
    for index, frame in enumerate(frames, start=1):
        try:
            frame_label = str(frame.relative_to(frame.parent.parent if frame.parent.name in {'day', 'night'} else frame.parent))
        except ValueError:
            frame_label = frame.name
        print(f"{index}. {frame_label}")

    raw_choice = input(f"\nChoose frame (1-{len(frames)}): ").strip()
    try:
        selected_index = int(raw_choice)
    except ValueError:
        print("Invalid selection.")
        return None

    if selected_index < 1 or selected_index > len(frames):
        print("Invalid selection.")
        return None

    return str(frames[selected_index - 1])


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
    fused_results = run_late_fusion(collect_diagnostics=True)
    print(f"Fused {len(fused_results)} samples into fused_qa_results.json")
    print("Wrote fusion_diagnostics.json, fusion_qa_stats.json, and fusion_qa_rows.csv")


def _run_segmented_qa_menu_option(modalities: list[str], label: str) -> None:
    estimate = estimate_segmented_work(
        dataset_folder="dataset",
        output_folder="segmented_outputs",
        modalities=modalities,
        resume=True,
        batch_segments=True,
        batch_audio=True,
    )
    print("\n" + "-" * 60)
    print(f"Running: {label}")
    print("-" * 60)
    print("This step reads segmented_outputs/dataset/**/*_task_segments.json.")
    print("It writes selected segmented modality results to segmented_outputs/.")
    print(f"Selected modality/modalities: {', '.join(modalities)}")
    print(f"Found {estimate['total_segments']} task segment(s). Resume + batched mode is enabled.")
    _print_segmented_work_estimate(estimate)
    print("Quota controls: source-batched Gemini calls, max_concurrent=1, delay_between_batches=12s.")
    print("-" * 60)
    if _confirm():
        output_paths = run_segmented_pipeline(
            dataset_folder="dataset",
            output_folder="segmented_outputs",
            skip_api=False,
            modalities=modalities,
            resume=True,
            max_concurrent=1,
            delay_between_segments=12,
            batch_segments=True,
            batch_audio=True,
        )
        print("Segmented outputs:")
        for modality, path in sorted(output_paths.items()):
            print(f"  {modality}: {path}")
    else:
        print("Cancelled.")


def _print_segmented_work_estimate(estimate: dict) -> None:
    modality_counts = estimate.get("modality_counts", {})
    for modality, counts in modality_counts.items():
        print(
            f"  {modality}: "
            f"{counts.get('pending_segments', 0)} pending, "
            f"{counts.get('skipped_segments', 0)} skipped, "
            f"{counts.get('source_batches', 0)} source batch(es), "
            f"~{counts.get('estimated_calls', 0)} Gemini call(s)"
        )
    print(f"Estimated remaining Gemini calls: ~{estimate.get('estimated_gemini_calls', 0)}")


def _run_all_segmented_qa_menu_option() -> None:
    visual_modalities = ["rgb", "event", "depth", "ir"]
    audio_modalities = ["audio"]

    print("\n" + "-" * 60)
    print("Running: ALL QA pipelines on task segments")
    print("-" * 60)
    print("Default quota-friendly flow: run visual modalities first, then optionally run AUDIO.")
    print("Resume mode is enabled, so completed segment results are skipped.")
    print("Batched mode is enabled: Gemini calls are grouped by source segment manifest.")
    print("Depth segmented QA uses Marigold maps from dataset/.frames_cache_marigold.")
    print("Quota controls: max_concurrent=1, delay_between_batches=12s.")
    print("\nVisual pass estimate:")
    visual_estimate = estimate_segmented_work(
        dataset_folder="dataset",
        output_folder="segmented_outputs",
        modalities=visual_modalities,
        resume=True,
        batch_segments=True,
        batch_audio=True,
    )
    _print_segmented_work_estimate(visual_estimate)
    print("-" * 60)

    if not _confirm("Run visual segmented QA now? (yes/no): "):
        print("Cancelled.")
        return

    output_paths = run_segmented_pipeline(
        dataset_folder="dataset",
        output_folder="segmented_outputs",
        skip_api=False,
        modalities=visual_modalities,
        resume=True,
        max_concurrent=1,
        delay_between_segments=12,
        batch_segments=True,
        batch_audio=True,
    )
    print("Visual segmented outputs:")
    for modality, path in sorted(output_paths.items()):
        print(f"  {modality}: {path}")

    print("\nAudio pass estimate:")
    audio_estimate = estimate_segmented_work(
        dataset_folder="dataset",
        output_folder="segmented_outputs",
        modalities=audio_modalities,
        resume=True,
        batch_segments=True,
        batch_audio=True,
    )
    _print_segmented_work_estimate(audio_estimate)
    print("AUDIO uses one Gemini call per pending source batch in batched mode.")
    print("-" * 60)

    if not _confirm("Run AUDIO segmented QA too? (yes/no): "):
        print("Skipped AUDIO segmented QA.")
        return

    audio_paths = run_segmented_pipeline(
        dataset_folder="dataset",
        output_folder="segmented_outputs",
        skip_api=False,
        modalities=audio_modalities,
        resume=True,
        max_concurrent=1,
        delay_between_segments=12,
        batch_segments=True,
        batch_audio=True,
    )
    print("Audio segmented outputs:")
    for modality, path in sorted(audio_paths.items()):
        print(f"  {modality}: {path}")


def _run_multimodal_qa_generation_menu_option() -> None:
    input_path = Path("segmented_normalized_evidence_units.json")
    output_path = MULTIMODAL_QA_CANDIDATES_PATH

    print("\n" + "-" * 60)
    print("Running: generate v2 implicit multimodal QA candidates")
    print("-" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {MULTIMODAL_QA_MODEL_NAME}")
    print("Mode: Gemini, segment-batched, resume/checkpoint enabled.")
    print("Batch controls: max_bundles_per_call=5, max_concurrent=1, delay_between_calls=5s.")
    print("This will use Gemini API quota.")
    print("-" * 60)

    if not input_path.exists():
        print(f"Missing input file: {input_path}")
        return

    if not _confirm():
        print("Cancelled.")
        return

    result_path = run_multimodal_qa_pipeline(
        input_path=input_path,
        output_path=output_path,
        generation_mode="gemini",
        test_mode=False,
        delay_between_calls=5,
        max_concurrent_calls=1,
        max_retries=3,
        resume=True,
        checkpoint_every=1,
        gemini_batch_scope="segment",
        max_bundles_per_gemini_call=5,
        model_name=MULTIMODAL_QA_MODEL_NAME,
    )
    print(f"Wrote v2 implicit multimodal QA candidates to {result_path}")


def _run_multimodal_qa_verifier_menu_option() -> None:
    input_path = MULTIMODAL_QA_CANDIDATES_PATH
    output_path = MULTIMODAL_QA_VERIFIED_PATH

    print("\n" + "-" * 60)
    print("Running: verify/filter v2 implicit multimodal QA candidates")
    print("-" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {MULTIMODAL_QA_MODEL_NAME}")
    print("Mode: Gemini verifier, batch_size=5, resume/checkpoint enabled.")
    print("This verifies validation-passed candidates and writes keep/benchmark_keep decisions.")
    print("This will use Gemini API quota.")
    print("-" * 60)

    if not input_path.exists():
        print(f"Missing input file: {input_path}")
        print("Run option 61 first, or generate the candidates file from the command script.")
        return

    if not _confirm():
        print("Cancelled.")
        return

    result_path = run_multimodal_qa_verifier(
        input_path=input_path,
        output_path=output_path,
        limit=None,
        resume=True,
        delay_between_calls=5,
        model_name=MULTIMODAL_QA_MODEL_NAME,
        batch_size=5,
        max_concurrent_calls=1,
        checkpoint_every=1,
    )
    print(f"Wrote verified v2 implicit multimodal QA to {result_path}")


def main():
    _migrate_legacy_aligned_qa_results()

    print("\n" + "=" * 60)
    print("BATCH + PARALLEL ANNOTATION PIPELINE TEST RUNNER")
    print("=" * 60)
    print("RGB: Single mega-prompt per pair with QA generation")
    print("EVENT: Single mega-prompt per pair with caption, question, and answer generation")
    print("DEPTH: Single mega-prompt per pair with caption, question, and answer generation")
    print("IR: Single mega-prompt per pair with caption, question, and answer generation")
    print("AUDIO: Cascaded HIA -> timestamped audio-visual caption -> QA generation")
    print("MARIGOLD: Estimate depth from cached RGB frames, then reuse depth QA prompts on Marigold maps")
    print("LATE FUSION: Post-process modality captions into fused scene summaries")
    print("MULTIMODAL QA: Generate and verify v2 implicit cross-modal QA benchmark items")
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
        print("\n--- MARIGOLD DEPTH ---")
        print("19. Estimate Marigold depth maps from cached RGB frames")
        print("20. Estimate Marigold depth maps from one selected night IR frame folder")
        print("21. Test Marigold depth estimation on one selected frame from one .frames_cache folder")
        print("22. Test Marigold depth estimation on one .frames_cache folder")
        print("23. Test Marigold depth QA on 1 cached pair (no API calls, demo Q&A)")
        print("24. Test Marigold depth QA on 1 cached pair with real Gemini API calls")
        print("25. Run Marigold depth QA on all cached pairs (production)")
        print("\n--- TEMPORAL ALIGNMENT (ONLY TESTS) ---")
        print("26. Run temporal alignment for day/night RGB/EVENT/IR/DEPTH videos")
        print("27. Export all day/night RGB/EVENT/DEPTH/IR aligned grid videos")
        print("28. Test optical-flow temporal alignment for check_mailbox day RGB/EVENT")
        print("29. Export optical-flow RGB/EVENT aligned video for check_mailbox day")
        print("30. Run DTW temporal alignment + export drift-corrected video for check_mailbox day RGB/EVENT")
        print("31. Run DTW temporal alignment + export all RGB/EVENT day/night videos")
        print("32. Run feature-based temporal alignment + export aligned video for check_mailbox day RGB/EVENT")
        print("33. Run cross-correlation temporal alignment + export video for check_mailbox day RGB/AUDIO")
        print("\n--- TEMPORAL ALIGNMENT ---")
        print("34. Run combined RGB/EVENT/IR/DEPTH DTW + RGB/AUDIO alignment for all dataset day/night files")
        print("35. Export cut_carrot aligned dataset as 30s separated segments")
        print("36. Export all dataset aligned samples as 30s separated segments")
        _print_registered_menu_sections(
            (
                "ALIGNED RGB PIPELINE",
                "ALIGNED EVENT PIPELINE",
                "ALIGNED IR PIPELINE",
            )
        )
        _print_registered_menu_sections(("ALIGNED AUDIO PIPELINE",))
        _print_registered_menu_sections(("ALIGNED MARIGOLD DEPTH PIPELINE",))
        print("\n--- LATE FUSION ---")
        print("48. Run late fusion on existing modality JSON results")
        print("\n--- TASK SLICING ---")
        print("49. Generate semantic task segment suggestions")
        print("50. Run RGB QA after task segment")
        print("51. Run EVENT QA after task segment")
        print("52. Run MARIGOLD DEPTH QA after task segment")
        print("53. Run IR QA after task segment")
        print("54. Run AUDIO QA after task segment")
        print("55. Run ALL QA pipelines on task segments")
        print("56. Export grouped Q/A pairs from segmented modality results")
        print("\n--- HOLISTIC QA ---")
        print("57. Normalize evidence units from existing modality JSON results")
        print("58. Group normalized evidence units by reasoning category")
        print("59. Export Q/A pairs from grouped QA into JSON")
        print("\n--- CSV EXPORT ---")
        print("60. Export segmented normalized evidence units to CSV")
        print("\n--- MULTIMODAL QA BENCHMARK ---")
        print("61. Generate v2 implicit multimodal QA candidates")
        print("62. Verify/filter v2 implicit multimodal QA candidates")
        _print_registered_menu_sections((
            "ALIGNED QA QUALITY",
            "BENCHMARK EVALUATION",
            "FRAME INPUT ANSWER BENCHMARK",
        ))
        print("\n63. Exit")

        choice = input("\nEnter choice (1-73 or action id): ").strip()

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
            print("Running: AUDIO cascade test on 1 pair (using demo data)")
            print("-" * 60)
            print("Uses demo HIA, timestamped caption, and Q&A")
            print("skip_api=True -> no Gemini API calls\n")
            run_audio(test_mode=True, skip_api=True)

        elif choice == "17":
            print("\n" + "-" * 60)
            print("Running: AUDIO cascade on 1 pair (real Gemini API calls)")
            print("WARNING: This will use Gemini API quota!")
            print("Uploads source RGB video for HIA, then matching with_audio media")
            print("-" * 60)
            if _confirm():
                run_audio(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "18":
            print("\n" + "-" * 60)
            print("Running: AUDIO cascade on all pairs (production)")
            print("WARNING: This will use Gemini API quota for each pair!")
            print("Parallel execution: up to 3 pairs concurrently, 4-second spacing")
            print("Outputs HIA, timestamped caption, and sound-centric Q&A")
            print("-" * 60)
            if _confirm():
                run_audio(test_mode=False)
            else:
                print("Cancelled.")

        elif choice == "19":
            print("\n" + "-" * 60)
            print("Running: Marigold depth estimation from cached RGB frames")
            print("Uses cached RGB frames from dataset/.frames_cache")
            print("Writes depth maps to dataset/.frames_cache_marigold")
            print("-" * 60)
            run_marigold_depth_estimation(test_mode=False)

        elif choice == "20":
            print("\n" + "-" * 60)
            print("Running: Marigold depth estimation on one selected night IR frame folder")
            print("Uses cached IR frames from dataset/.frames_cache_ir")
            print("Writes depth maps to dataset/.frames_cache_marigold_ir")
            print("-" * 60)
            selected_folder = _select_ir_night_cache_folder()
            if selected_folder:
                run_marigold_ir_depth_estimation(
                    selected_cache_folder=selected_folder,
                )
            else:
                print("Cancelled.")

        elif choice == "21":
            print("\n" + "-" * 60)
            print("Running: Marigold depth estimation on one selected frame from one .frames_cache folder")
            print("The selected frame will be processed on its inferred day/night side only")
            print("-" * 60)
            selected_folder = _select_cache_folder()
            if selected_folder:
                selected_frame = _select_cache_frame(selected_folder)
                if selected_frame:
                    run_marigold_depth_estimation(
                        test_mode=True,
                        selected_cache_folder=selected_folder,
                        selected_frame=selected_frame,
                    )
                else:
                    print("Cancelled.")
            else:
                print("Cancelled.")

        elif choice == "22":
            print("\n" + "-" * 60)
            print("Running: Marigold depth estimation on one selected .frames_cache folder")
            print("The selected folder will be resolved to its full day/night pair when possible")
            print("-" * 60)
            selected_folder = _select_cache_folder()
            if selected_folder:
                run_marigold_depth_estimation(
                    test_mode=True,
                    selected_cache_folder=selected_folder,
                )
            else:
                print("Cancelled.")

        elif choice == "23":
            print("\n" + "-" * 60)
            print("Running: Marigold depth QA test on 1 cached pair (demo Q&A)")
            print("Requires existing Marigold depth maps in dataset/.frames_cache_marigold")
            print("-" * 60)
            print("skip_api=True -> demo Q&A\n")
            run_marigold_depth_qa(test_mode=True, skip_api=True)

        elif choice == "24":
            print("\n" + "-" * 60)
            print("Running: Marigold depth QA on 1 cached pair (real Gemini API calls)")
            print("WARNING: This will use Gemini API quota!")
            print("Reuses the existing depth QA prompts on Marigold-estimated depth maps")
            print("-" * 60)
            if _confirm():
                run_marigold_depth_qa(test_mode=True, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "25":
            print("\n" + "-" * 60)
            print("Running: Marigold depth QA on all cached pairs (production)")
            print("WARNING: This will use Gemini API quota for each cached Marigold pair!")
            print("Reuses the existing depth QA pipeline and writes marigold_depth_qa_results.json")
            print("-" * 60)
            if _confirm():
                run_marigold_depth_qa(test_mode=False, skip_api=False)
            else:
                print("Cancelled.")

        elif choice == "26":
            print("\n" + "-" * 60)
            print("Running: temporal alignment for day/night RGB/EVENT/IR/DEPTH videos")
            print("-" * 60)
            print("Uses local OpenCV processing only; no Gemini/API calls.")
            print("Uses RGB as the reference and aligns event, IR, and depth to it.")
            print("Writes temporal_alignment_json/temporal_alignment_day_results.json and temporal_alignment_json/temporal_alignment_night_results.json.")
            print("Writes side-labeled combined and per-modality activity-signal PNG plots under temporal_alignment_plots/.\n")
            alignment_results = run_day_night_temporal_alignment(
                dataset_folder="dataset",
                day_output_path="temporal_alignment_json/temporal_alignment_day_results.json",
                night_output_path="temporal_alignment_json/temporal_alignment_night_results.json",
                plot_output_folder="temporal_alignment_plots",
            )
            print(f"Aligned {len(alignment_results['day'])} day multimodal sample(s).")
            print(f"Aligned {len(alignment_results['night'])} night multimodal sample(s).")
            print("Results saved to temporal_alignment_json/temporal_alignment_day_results.json and temporal_alignment_json/temporal_alignment_night_results.json")
            print("Plots saved under temporal_alignment_plots/")

        elif choice == "27":
            print("\n" + "-" * 60)
            print("Running: export all day/night RGB/EVENT/DEPTH/IR aligned grid videos")
            print("-" * 60)
            print("Reads temporal_alignment_json/temporal_alignment_day_results.json and temporal_alignment_json/temporal_alignment_night_results.json.")
            print("Uses stored EVENT, DEPTH, and IR offset_seconds for each sample.")
            print("Writes low-resolution preview grids under temporal_alignment_exports/.")
            print("Tries GPU h264_nvenc first, then falls back to CPU libx264 if needed.\n")
            export_summary = export_day_night_rgb_event_depth_ir_alignment_grids(
                day_alignment_input_path="temporal_alignment_json/temporal_alignment_day_results.json",
                night_alignment_input_path="temporal_alignment_json/temporal_alignment_night_results.json",
                output_folder="temporal_alignment_exports",
                prefer_gpu=True,
            )
            print(f"Exported {export_summary['exported_count']} aligned preview video(s).")
            print(f"Skipped {export_summary['skipped_count']} sample(s).")
            print(f"Summary saved to {export_summary['summary_file']}")
            for item in export_summary["exported"]:
                print(
                    f"- {item['sample']} {item['side']}: {item['output_file']} "
                    f"({item['duration_seconds']}s, {item['encoder']})"
                )
            for item in export_summary["skipped"]:
                print(f"- skipped {item['sample']} {item['side']}: {item['reason']}")

        elif choice == "28":
            print("\n" + "-" * 60)
            print("Running: optical-flow temporal alignment for check_mailbox day RGB/EVENT")
            print("-" * 60)
            print("This is diagnostic only and does not overwrite temporal_alignment_json/temporal_alignment_day_results.json.")
            print("Writes temporal_alignment_json/temporal_alignment_optical_flow_check_mailbox_day_event.json.")
            print("Writes temporal_alignment_plots/check_mailbox_day_rgb_event_optical_flow_activity_signal.png.\n")
            optical_flow_result = run_check_mailbox_day_rgb_event_optical_flow_alignment(
                dataset_folder="dataset",
                output_path="temporal_alignment_json/temporal_alignment_optical_flow_check_mailbox_day_event.json",
                plot_output_folder="temporal_alignment_plots",
            )
            alignment = optical_flow_result.get("alignment") or {}
            comparison = optical_flow_result.get("comparison") or {}
            print(f"Optical-flow selected offset: {alignment.get('offset_seconds')}s")
            print(f"Optical-flow peak correlation: {alignment.get('peak_correlation')}")
            print(f"Optical-flow confidence: {alignment.get('confidence_label')}")
            print(
                "Motion-energy comparison: "
                f"selected={comparison.get('motion_energy_selected_offset_seconds')}s, "
                f"raw_best={comparison.get('motion_energy_raw_best_offset_seconds')}s"
            )
            print("Top optical-flow candidates:")
            for candidate in alignment.get("candidate_offsets", [])[:5]:
                print(
                    f"- offset={candidate.get('offset_seconds')}s, "
                    f"corr={candidate.get('correlation')}, score={candidate.get('score')}"
                )
            if optical_flow_result.get("warnings"):
                print("Warnings:")
                for warning in optical_flow_result["warnings"]:
                    print(f"- {warning}")

        elif choice == "29":
            print("\n" + "-" * 60)
            print("Running: export optical-flow RGB/EVENT aligned video for check_mailbox day")
            print("-" * 60)
            print("Reads temporal_alignment_json/temporal_alignment_optical_flow_check_mailbox_day_event.json.")
            print("Uses the stored optical-flow EVENT offset_seconds.")
            print("Writes a low-resolution RGB/EVENT preview under temporal_alignment_exports/.")
            print("Tries GPU h264_nvenc first, then falls back to CPU libx264 if needed.\n")
            export_summary = export_check_mailbox_day_rgb_event_optical_flow_alignment(
                alignment_input_path="temporal_alignment_json/temporal_alignment_optical_flow_check_mailbox_day_event.json",
                output_folder="temporal_alignment_exports",
                prefer_gpu=True,
            )
            print(f"Exported {export_summary['exported_count']} aligned preview video(s).")
            print(f"Skipped {export_summary['skipped_count']} sample(s).")
            print(f"Summary saved to {export_summary['summary_file']}")
            for item in export_summary["exported"]:
                print(
                    f"- {item['sample']} {item['side']}: {item['output_file']} "
                    f"({item['duration_seconds']}s, {item['encoder']})"
                )
            for item in export_summary["skipped"]:
                print(f"- skipped {item['sample']} {item['side']}: {item['reason']}")

        elif choice == "30":
            print("\n" + "-" * 60)
            print("Running: DTW temporal alignment + drift-corrected export for check_mailbox day RGB/EVENT")
            print("-" * 60)
            print("This is diagnostic only and does not overwrite temporal_alignment_json/temporal_alignment_day_results.json.")
            print("Uses optical-flow activity traces and Dynamic Time Warping to estimate a time-varying EVENT offset.")
            print("Writes temporal_alignment_json/temporal_alignment_dtw_check_mailbox_day_event.json.")
            print("Writes temporal_alignment_plots/check_mailbox_day_rgb_event_dtw_activity_signal.png.")
            print("Writes temporal_alignment_exports/check_mailbox_day_rgb_event_dtw_sliced_aligned.mp4.\n")
            dtw_result = run_and_export_check_mailbox_day_rgb_event_dtw_alignment(
                dataset_folder="dataset",
                output_path="temporal_alignment_json/temporal_alignment_dtw_check_mailbox_day_event.json",
                plot_output_folder="temporal_alignment_plots",
                output_folder="temporal_alignment_exports",
                window_seconds=10.0,
            )
            alignment = dtw_result.get("alignment") or {}
            export_summary = dtw_result.get("export") or {}
            print(f"DTW median offset: {alignment.get('offset_seconds')}s")
            print(f"DTW start offset: {alignment.get('start_offset_seconds')}s")
            print(f"DTW end offset: {alignment.get('end_offset_seconds')}s")
            print(f"DTW drift: {alignment.get('offset_drift_seconds')}s")
            print(f"DTW path length: {alignment.get('dtw_path_length')}")
            print(f"Exported {export_summary.get('exported_count', 0)} drift-corrected preview video(s).")
            print(f"Skipped {export_summary.get('skipped_count', 0)} sample(s).")
            if export_summary.get("summary_file"):
                print(f"Summary saved to {export_summary['summary_file']}")
            for item in export_summary.get("exported", []):
                print(
                    f"- {item['sample']} {item['side']}: {item['output_file']} "
                    f"({item['duration_seconds']}s, {item['encoder']})"
                )
            for item in export_summary.get("skipped", []):
                print(f"- skipped {item['sample']} {item['side']}: {item['reason']}")
            if dtw_result.get("warnings"):
                print("Warnings:")
                for warning in dtw_result["warnings"]:
                    print(f"- {warning}")

        elif choice == "31":
            print("\n" + "-" * 60)
            print("Running: DTW temporal alignment + export all RGB/EVENT day/night videos")
            print("-" * 60)
            print("Uses optical-flow activity traces and Dynamic Time Warping for every complete RGB/EVENT pair.")
            print("Includes both day and night videos discovered under dataset/.")
            print("Writes per-pair temporal_alignment_json/temporal_alignment_dtw_<sample>_<side>_event.json files.")
            print("Writes temporal_alignment_exports/*_rgb_event_dtw_sliced_aligned.mp4.")
            print("Writes temporal_alignment_exports/rgb_event_dtw_all_export_summary.json.\n")
            export_summary = run_and_export_all_rgb_event_dtw_alignments(
                dataset_folder="dataset",
                alignment_output_folder="temporal_alignment_json",
                plot_output_folder="temporal_alignment_plots",
                output_folder="temporal_alignment_exports",
                window_seconds=10.0,
            )
            print(f"Discovered {export_summary.get('discovered_count', 0)} RGB/EVENT pair(s).")
            print(f"Exported {export_summary['exported_count']} drift-corrected preview video(s).")
            print(f"Skipped {export_summary['skipped_count']} pair(s).")
            print(f"Summary saved to {export_summary['summary_file']}")
            for item in export_summary["exported"]:
                print(
                    f"- {item['sample']} {item['side']}: {item['output_file']} "
                    f"({item['duration_seconds']}s, {item['frames_written']} frames)"
                )
            for item in export_summary["skipped"]:
                print(f"- skipped {item['sample']} {item['side']}: {item['reason']}")

        elif choice == "32":
            print("\n" + "-" * 60)
            print("Running: feature-based temporal alignment + aligned export for check_mailbox day RGB/EVENT")
            print("-" * 60)
            print("This is diagnostic only and does not overwrite temporal_alignment_json/temporal_alignment_day_results.json.")
            print("Uses ORB feature matches on edge-like RGB/EVENT frames to estimate local EVENT offsets.")
            print("Writes temporal_alignment_json/temporal_alignment_feature_check_mailbox_day_event.json.")
            print("Writes temporal_alignment_plots/check_mailbox_day_rgb_event_feature_offsets.png.")
            print("Writes temporal_alignment_exports/check_mailbox_day_rgb_event_feature_aligned.mp4.\n")
            feature_result = run_and_export_check_mailbox_day_rgb_event_feature_alignment(
                dataset_folder="dataset",
                output_path="temporal_alignment_json/temporal_alignment_feature_check_mailbox_day_event.json",
                plot_output_folder="temporal_alignment_plots",
                output_folder="temporal_alignment_exports",
            )
            alignment = feature_result.get("alignment") or {}
            export_summary = feature_result.get("export") or {}
            local_windows = alignment.get("local_windows") or []
            high_or_medium = [
                window for window in local_windows if window.get("confidence_label") in {"high", "medium"}
            ]
            print(f"Feature median offset: {alignment.get('offset_seconds')}s")
            print(f"Feature start offset: {alignment.get('start_offset_seconds')}s")
            print(f"Feature end offset: {alignment.get('end_offset_seconds')}s")
            print(f"Feature drift: {alignment.get('offset_drift_seconds')}s")
            print(f"Feature windows: {len(local_windows)} total, {len(high_or_medium)} medium/high confidence")
            print(f"Exported {export_summary.get('exported_count', 0)} feature-aligned preview video(s).")
            print(f"Skipped {export_summary.get('skipped_count', 0)} sample(s).")
            if export_summary.get("summary_file"):
                print(f"Summary saved to {export_summary['summary_file']}")
            for item in export_summary.get("exported", []):
                print(
                    f"- {item['sample']} {item['side']}: {item['output_file']} "
                    f"({item['duration_seconds']}s, {item['encoder']})"
                )
            for item in export_summary.get("skipped", []):
                print(f"- skipped {item['sample']} {item['side']}: {item['reason']}")
            if feature_result.get("warnings"):
                print("Warnings:")
                for warning in feature_result["warnings"]:
                    print(f"- {warning}")

        elif choice == "33":
            print("\n" + "-" * 60)
            print("Running: cross-correlation temporal alignment + export for check_mailbox day RGB/AUDIO")
            print("-" * 60)
            print("Uses RGB optical-flow activity and separate .m4a audio RMS activity.")
            print("Ignores check_mailbox_day_rgb_with_audio.mp4 and embedded RGB audio.")
            print("Uses one fixed offset, so audio playback speed is not warped.")
            print("Writes temporal_alignment_json/temporal_alignment_cross_correlation_check_mailbox_day_audio.json.")
            print("Writes temporal_alignment_plots/check_mailbox_day_rgb_audio_cross_correlation_activity_signal.png.")
            print("Writes temporal_alignment_exports/check_mailbox_day_rgb_audio_cross_correlation_aligned.mp4.\n")
            audio_result = run_and_export_check_mailbox_day_rgb_audio_cross_correlation_alignment(
                dataset_folder="dataset",
                output_path="temporal_alignment_json/temporal_alignment_cross_correlation_check_mailbox_day_audio.json",
                plot_output_folder="temporal_alignment_plots",
                output_folder="temporal_alignment_exports",
            )
            alignment = audio_result.get("alignment") or {}
            export_summary = audio_result.get("export") or {}
            print(f"RGB/AUDIO offset: {alignment.get('offset_seconds')}s")
            print(f"RGB/AUDIO correlation: {alignment.get('peak_correlation')}")
            print(f"RGB/AUDIO confidence: {alignment.get('confidence_label')}")
            if audio_result.get("plot_file"):
                print(f"Activity plot saved to {audio_result['plot_file']}")
            print(f"Exported {export_summary.get('exported_count', 0)} RGB/AUDIO preview video(s).")
            print(f"Skipped {export_summary.get('skipped_count', 0)} sample(s).")
            if export_summary.get("summary_file"):
                print(f"Summary saved to {export_summary['summary_file']}")
            for item in export_summary.get("exported", []):
                print(
                    f"- {item['sample']} {item['side']}: {item['output_file']} "
                    f"({item['duration_seconds']}s, {item['encoder']})"
                )
            for item in export_summary.get("skipped", []):
                print(f"- skipped {item['sample']} {item['side']}: {item['reason']}")
            if audio_result.get("warnings"):
                print("Warnings:")
                for warning in audio_result["warnings"]:
                    print(f"- {warning}")
            if alignment.get("warnings"):
                print("Alignment warnings:")
                for warning in alignment["warnings"]:
                    print(f"- {warning}")

        elif choice == "34":
            print("\n" + "-" * 60)
            print("Running: combined RGB/EVENT/IR/DEPTH DTW + RGB/AUDIO alignment for all dataset day/night files")
            print("-" * 60)
            print("Discovers every complete RGB/EVENT/IR/DEPTH/.m4a set under dataset/.")
            print("Includes day and night files, and ignores *_rgb_with_audio.mp4 plus embedded RGB audio.")
            print("Writes per-pair EVENT DTW and RGB/AUDIO JSON files under temporal_alignment_json/.")
            print("Aligns IR and DEPTH to RGB with fixed-offset cross-correlation.")
            print("Writes temporal_alignment_exports/*_rgb_event_ir_depth_dtw_with_aligned_audio.mp4.")
            print("Does not keep intermediate no-audio preview videos.")
            print("Writes temporal_alignment_exports/rgb_event_ir_depth_dtw_with_audio_all_export_summary.json.\n")
            combined_summary = run_and_export_all_rgb_event_dtw_with_audio_alignments(
                dataset_folder="dataset",
                alignment_output_folder="temporal_alignment_json",
                plot_output_folder="temporal_alignment_plots",
                output_folder="temporal_alignment_exports",
                window_seconds=10.0,
            )
            print(f"Discovered {combined_summary.get('discovered_count', 0)} complete candidate set(s).")
            print(f"Exported {combined_summary.get('exported_count', 0)} combined 2x2 preview video(s).")
            print(f"Skipped {combined_summary.get('skipped_count', 0)} set(s).")
            if combined_summary.get("summary_file"):
                print(f"Summary saved to {combined_summary['summary_file']}")
            for item in combined_summary.get("exported", []):
                print(
                    f"- {item['sample']} {item['side']}: {item['output_file']} "
                    f"({item['duration_seconds']}s, audio offset {item['audio_offset_seconds']}s)"
                )
            for item in combined_summary.get("skipped", []):
                print(f"- skipped {item['sample']} {item['side']}: {item['reason']}")

        elif choice == "35":
            print("\n" + "-" * 60)
            print("Running: export cut_carrot aligned dataset as 30s separated segments")
            print("-" * 60)
            print("Uses dataset/cut_carrot_split for both day and night.")
            print("Writes separated aligned RGB/EVENT/IR/DEPTH/AUDIO files under aligned_dataset/cut_carrot_split/SegN/.")
            print("Uses EVENT DTW plus fixed-offset IR/DEPTH/AUDIO alignment to RGB.")
            print("Exports exact 30-second full segments and records any dropped remainder.")
            print("No grid layout, labels, overlays, or embedded RGB audio are used.\n")
            summary = run_and_export_cut_carrot_aligned_dataset_segments(
                dataset_folder="dataset",
                output_folder="aligned_dataset",
                alignment_output_folder="temporal_alignment_json",
                plot_output_folder="temporal_alignment_plots",
                segment_seconds=30.0,
                window_seconds=10.0,
            )
            print(f"Exported {summary.get('exported_segment_count', 0)} segment folder(s).")
            print(f"Skipped {summary.get('skipped_count', 0)} item(s).")
            if summary.get("summary_file"):
                print(f"Summary saved to {summary['summary_file']}")
            for side, item in summary.get("sides", {}).items():
                print(
                    f"- {side}: {item.get('segment_count', 0)} segment(s), "
                    f"dropped remainder {item.get('dropped_remainder_seconds', 0)}s"
                )
            for item in summary.get("skipped", []):
                print(f"- skipped {item.get('side', 'unknown')} {item.get('segment', '')}: {item.get('reason')}")

        elif choice == "36":
            print("\n" + "-" * 60)
            print("Running: export all dataset aligned samples as 30s separated segments")
            print("-" * 60)
            print("Discovers complete RGB/EVENT/IR/DEPTH/.m4a sets under dataset/.")
            print("Writes separated aligned modality files under aligned_dataset/<split>/SegN/.")
            print("Uses EVENT DTW plus fixed-offset IR/DEPTH/AUDIO alignment to RGB.")
            print("Exports exact 30-second full segments and records any dropped remainder.")
            print("No grid layout, labels, overlays, or embedded RGB audio are used.\n")
            summary = run_and_export_all_aligned_dataset_segments(
                dataset_folder="dataset",
                output_folder="aligned_dataset",
                alignment_output_folder="temporal_alignment_json",
                plot_output_folder="temporal_alignment_plots",
                segment_seconds=30.0,
                window_seconds=10.0,
            )
            print(f"Discovered {summary.get('discovered_count', 0)} complete candidate side(s).")
            print(f"Exported {summary.get('exported_segment_count', 0)} segment folder record(s).")
            print(f"Skipped {summary.get('skipped_count', 0)} item(s).")
            if summary.get("summary_file"):
                print(f"Summary saved to {summary['summary_file']}")
            for item in summary.get("splits", []):
                print(
                    f"- {item.get('split_folder_name')}: {item.get('exported_segment_count', 0)} segment folder(s), "
                    f"summary {item.get('summary_file')}"
                )
            for item in summary.get("skipped", []):
                print(
                    f"- skipped {item.get('sample', 'unknown')} {item.get('side', '')} "
                    f"{item.get('segment', '')}: {item.get('reason')}"
                )

        elif choice in REGISTERED_MENU_ACTIONS:
            REGISTERED_MENU_ACTIONS[choice].run()

        elif choice == "48":
            print("\n" + "-" * 60)
            print("Running: late fusion on existing modality JSON results")
            print("-" * 60)
            print("This step reads the current RGB, IR, event, audio, and depth result files.")
            print("It writes fused QA results, diagnostics, and analysis files.\n")
            fused_results = run_late_fusion(collect_diagnostics=True)
            print(f"Fused {len(fused_results)} samples into fused_qa_results.json")
            print("Wrote fusion_diagnostics.json, fusion_qa_stats.json, and fusion_qa_rows.csv")

        elif choice == "49":
            print("\n" + "-" * 60)
            print("Running: generate semantic task segment suggestions")
            print("-" * 60)
            print("This step reads dataset videos/audio across RGB, event, depth, IR, and audio.")
            print("It writes editable metadata-only *_task_segments.json manifests under segmented_outputs/dataset/.")
            print("WARNING: This will use Gemini API quota for each day/night sample.")
            print("-" * 60)
            if _confirm():
                output_paths = run_task_slicing(dataset_folder="dataset", test_mode=False)
                print(f"Generated {len(output_paths)} task segment manifest(s).")
            else:
                print("Cancelled.")

        elif choice == "50":
            _run_segmented_qa_menu_option(["rgb"], "RGB QA after task segment")

        elif choice == "51":
            _run_segmented_qa_menu_option(["event"], "EVENT QA after task segment")

        elif choice == "52":
            _run_segmented_qa_menu_option(["depth"], "MARIGOLD DEPTH QA after task segment")

        elif choice == "53":
            _run_segmented_qa_menu_option(["ir"], "IR QA after task segment")

        elif choice == "54":
            _run_segmented_qa_menu_option(["audio"], "AUDIO QA after task segment")

        elif choice == "55":
            _run_all_segmented_qa_menu_option()

        elif choice == "56":
            print("\n" + "-" * 60)
            print("Running: export grouped Q/A pairs from segmented modality results")
            print("-" * 60)
            print("This step reads segmented_outputs/segmented_*_qa_results.json.")
            print("It writes split grouped Q/A pairs to segmented_grouped_qa_pairs.json.\n")
            segmented_grouped_qa_results = run_export_segmented_grouped_qa()
            print(
                f"Exported {len(segmented_grouped_qa_results)} segmented samples "
                "into segmented_grouped_qa_pairs.json"
            )

        elif choice == "57":
            print("\n" + "-" * 60)
            print("Running: normalize evidence units from existing modality JSON results")
            print("-" * 60)
            print("This step reads RGB, event, depth, IR, and audio result files.")
            print("It writes normalized section-level evidence to normalized_evidence_units.json.\n")
            normalized_results = normalize_all_modalities()
            print(f"Normalized {len(normalized_results)} samples into normalized_evidence_units.json")

        elif choice == "58":
            print("\n" + "-" * 60)
            print("Running: group normalized evidence units by reasoning category")
            print("-" * 60)
            print("This step reads normalized_evidence_units.json.")
            print("It writes grouped evidence to grouped_evidence.json.\n")
            grouped_results = run_group_evidence()
            print(f"Grouped {len(grouped_results)} samples into grouped_evidence.json")

        elif choice == "59":
            print("\n" + "-" * 60)
            print("Running: export grouped Q/A pairs to separate JSON")
            print("-" * 60)
            print("This step reads grouped_evidence.json.")
            print("It writes split Q/A pairs to grouped_qa_pairs.json.\n")
            grouped_qa_results = run_export_grouped_qa()
            print(f"Exported {len(grouped_qa_results)} samples into grouped_qa_pairs.json")

        elif choice == "60":
            print("\n" + "-" * 60)
            print("Running: export segmented normalized evidence units to CSV")
            print("-" * 60)
            print("This step reads segmented_normalized_evidence_units.json.")
            print("It writes segmented_normalized_evidence_units.csv.\n")
            row_count = run_export_segmented_normalized_evidence_csv()
            print(f"Exported {row_count} row(s) into segmented_normalized_evidence_units.csv")

        elif choice == "61":
            _run_multimodal_qa_generation_menu_option()

        elif choice == "62":
            _run_multimodal_qa_verifier_menu_option()

        elif choice == "63":
            print("\nExiting.")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
