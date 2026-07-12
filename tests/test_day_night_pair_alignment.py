import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from annotation_feature import day_night_pair_alignment as alignment


class DayNightPairAlignmentTests(unittest.TestCase):
    def test_constrained_dtw_is_monotonic_and_respects_step_ratios(self):
        reference = np.arange(8, dtype=np.float32)[:, None]
        target = np.linspace(0, 7, 11, dtype=np.float32)[:, None]
        cost = np.abs(reference - target.T)
        path = alignment._constrained_dtw(cost)

        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (7, 10))
        for (old_i, old_j), (new_i, new_j) in zip(path, path[1:]):
            self.assertIn((new_i - old_i, new_j - old_j), {(1, 1), (1, 2), (2, 1)})

    def test_constrained_dtw_rejects_extreme_duration_ratio(self):
        with self.assertRaisesRegex(ValueError, "outside 0.5-2.0"):
            alignment._constrained_dtw(np.zeros((3, 10), dtype=np.float32))

    def test_active_interval_retains_at_least_seventy_percent(self):
        motion = np.zeros(100, dtype=np.float32)
        motion[48:52] = 8.0
        start, end = alignment._active_interval(motion)
        self.assertGreaterEqual(end - start + 1, 70)
        self.assertLessEqual(start, 48)
        self.assertGreaterEqual(end, 51)

    def test_frame_mapping_is_monotonic_bounded_and_marks_unmatched_edges(self):
        source = {"fps": 10.0, "frame_count": 50}
        target = {"fps": 20.0, "frame_count": 80}
        rows = alignment._interpolate_mapping(
            source,
            target,
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0.5, 1.5, 2.5]),
            np.asarray([0.9, 0.8, 0.7]),
        )
        self.assertEqual(rows[0]["status"], "unmatched")
        self.assertEqual(rows[-1]["status"], "unmatched")
        matched = [row for row in rows if row["target_frame"] != ""]
        target_frames = [int(row["target_frame"]) for row in matched]
        self.assertEqual(target_frames, sorted(target_frames))
        self.assertTrue(all(0 <= frame < 80 for frame in target_frames))

    def test_inverse_knots_uses_median_for_many_to_one_matches(self):
        knots = [
            {"day_sample_index": 2, "day_time_seconds": 0.4, "night_time_seconds": 0.2, "confidence": 0.8},
            {"day_sample_index": 2, "day_time_seconds": 0.4, "night_time_seconds": 0.4, "confidence": 0.6},
            {"day_sample_index": 3, "day_time_seconds": 0.6, "night_time_seconds": 0.6, "confidence": 0.9},
        ]
        day, night, confidence = alignment._inverse_knots(knots)
        np.testing.assert_allclose(day, [0.4, 0.6])
        np.testing.assert_allclose(night, [0.3, 0.6])
        np.testing.assert_allclose(confidence, [0.7, 0.9])

    def test_missing_wash_cup_input_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "wash_cup_day_rgb"):
                alignment.run_wash_cup_day_night_rgb_alignment(dataset_folder=directory, write_preview=False)

    def test_cut_carrot_wrapper_uses_cut_carrot_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "cut_carrot_day_rgb"):
                alignment.run_cut_carrot_day_night_rgb_alignment(
                    dataset_folder=directory, write_preview=False
                )

    def test_check_mailbox_wrapper_uses_check_mailbox_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "check_mailbox_day_rgb"):
                alignment.run_check_mailbox_day_night_rgb_alignment(
                    dataset_folder=directory, write_preview=False
                )

    def test_discovery_supports_sample_name_different_from_split_name(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "wash_hand_split"
            split.mkdir()
            (split / "wash_hands_day_rgb.mp4").touch()
            (split / "wash_hands_night_rgb.mp4").touch()
            pairs = alignment._discover_day_night_rgb_pairs(Path(directory))
        self.assertEqual(pairs[0]["sample"], "wash_hands")
        self.assertEqual(pairs[0]["split_folder_name"], "wash_hand_split")

    def test_batch_alignment_continues_after_pair_failure_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for sample in ("alpha", "beta"):
                split = dataset / f"{sample}_split"
                split.mkdir(parents=True)
                (split / f"{sample}_day_rgb.mp4").touch()
                (split / f"{sample}_night_rgb.mp4").touch()

            def run_pair(sample_name, **_kwargs):
                if sample_name == "beta":
                    raise RuntimeError("synthetic failure")
                return {
                    "coverage": {"day": 1.0, "night": 1.0},
                    "review_intervals": [],
                    "outputs": {"alignment_json": "alpha.json"},
                }

            with patch.object(alignment, "run_day_night_rgb_pair_alignment", side_effect=run_pair):
                summary = alignment.run_all_day_night_rgb_pair_alignments(
                    dataset_folder=dataset,
                    output_folder=root / "output",
                    write_preview=False,
                )
            self.assertEqual(summary["discovered_count"], 2)
            self.assertEqual(summary["aligned_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertTrue(Path(summary["summary_file"]).is_file())

    def test_end_to_end_writes_json_and_bidirectional_csv_without_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "dataset" / "wash_cup_split"
            split.mkdir(parents=True)
            day_path = split / "wash_cup_day_rgb.mp4"
            night_path = split / "wash_cup_night_rgb.mp4"
            day_path.touch()
            night_path.touch()
            output = root / "output"
            times = np.arange(20, dtype=np.float32) / alignment.FEATURE_FPS
            semantic = np.eye(20, dtype=np.float32)
            features = {"times": times, "motion": np.linspace(0, 1, 20), "semantic": semantic}

            def metadata(path):
                return {
                    "path": str(path),
                    "fps": 10.0,
                    "frame_count": 40,
                    "duration_seconds": 4.0,
                    "width": 16,
                    "height": 16,
                }

            with patch.object(alignment, "_video_metadata", side_effect=metadata), patch.object(
                alignment, "_load_or_extract_features", return_value=features
            ):
                summary = alignment.run_wash_cup_day_night_rgb_alignment(
                    dataset_folder=root / "dataset", output_folder=output, write_preview=False
                )

            json_path = Path(summary["outputs"]["alignment_json"])
            self.assertTrue(json_path.is_file())
            self.assertIsNone(summary["outputs"]["preview"])
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["reference_side"], "night")
            for key in ("night_to_day_csv", "day_to_night_csv"):
                with open(summary["outputs"][key], newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 40)

    def test_check_mailbox_robustness_export_writes_aligned_1fps_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "dataset" / "check_mailbox_split"
            split.mkdir(parents=True)
            day_path = split / "check_mailbox_day_rgb.mp4"
            night_path = split / "check_mailbox_night_rgb.mp4"

            writer = cv2.VideoWriter(str(day_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (16, 16))
            for index in range(6):
                writer.write(np.full((16, 16, 3), index * 20, dtype=np.uint8))
            writer.release()
            writer = cv2.VideoWriter(str(night_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (16, 16))
            for index in range(6):
                writer.write(np.full((16, 16, 3), 255 - index * 20, dtype=np.uint8))
            writer.release()

            alignment_folder = root / "day_night_alignment" / "check_mailbox_split"
            alignment_folder.mkdir(parents=True)
            with open(alignment_folder / "check_mailbox_night_to_day_frames.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "night_frame",
                        "night_time_seconds",
                        "day_frame",
                        "day_time_seconds",
                        "confidence",
                        "status",
                    ],
                )
                writer.writeheader()
                for index in range(6):
                    writer.writerow(
                        {
                            "night_frame": index,
                            "night_time_seconds": index / 2.0,
                            "day_frame": 5 - index,
                            "day_time_seconds": (5 - index) / 2.0,
                            "confidence": 0.9,
                            "status": "matched",
                        }
                    )

            summary = alignment.export_check_mailbox_day_night_robustness_qa_1fps_frames(
                dataset_folder=root / "dataset",
                alignment_folder=alignment_folder,
                sample_fps=1.0,
            )

            self.assertEqual(summary["exported_pair_count"], 3)
            self.assertTrue((alignment_folder / "day_night_robustness_qa_1fps_frames" / "night" / "frame_000000.png").is_file())
            self.assertTrue((alignment_folder / "day_night_robustness_qa_1fps_frames" / "day" / "frame_000000.png").is_file())
            self.assertTrue(
                (alignment_folder / "day_night_robustness_qa_1fps_frames" / "side_by_side" / "frame_000000.png").is_file()
            )
            self.assertEqual(
                summary["frames"][0]["side_by_side_frame_path"],
                str(alignment_folder / "day_night_robustness_qa_1fps_frames" / "side_by_side" / "frame_000000.png"),
            )
            self.assertEqual(
                summary["side_by_side_frame_folder"],
                str(alignment_folder / "day_night_robustness_qa_1fps_frames" / "side_by_side"),
            )
            self.assertTrue(Path(summary["manifest_json"]).is_file())
            self.assertTrue(Path(summary["manifest_csv"]).is_file())

    def test_native_rgb_cut_plan_writes_svg_and_exact_json_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "dataset" / "check_mailbox_split"
            split.mkdir(parents=True)
            day_path = split / "check_mailbox_day_rgb.mp4"
            night_path = split / "check_mailbox_night_rgb.mp4"
            day_path.touch()
            night_path.touch()
            alignment_folder = root / "alignment"
            alignment_folder.mkdir()
            mapping = alignment_folder / "check_mailbox_night_to_day_frames.csv"
            with open(mapping, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["night_frame", "night_time_seconds", "day_frame", "day_time_seconds", "confidence", "status"])
                writer.writeheader()
                for second in range(0, 71):
                    writer.writerow({"night_frame": second, "night_time_seconds": second, "day_frame": second + 10, "day_time_seconds": second + 10, "confidence": 0.9, "status": "matched"})

            def metadata(path):
                duration = 90.0 if path == day_path else 71.0
                return {"path": str(path), "fps": 1.0, "frame_count": int(duration), "duration_seconds": duration, "width": 16, "height": 16}

            with patch.object(alignment, "_video_metadata", side_effect=metadata):
                plan = alignment.create_check_mailbox_native_rgb_cut_plan(root / "dataset", alignment_folder, segment_seconds=30.0)

            self.assertEqual(plan["manifest_type"], "native_day_night_rgb_cut_plan_v2")
            self.assertEqual(plan["cut_count"], 4)
            self.assertEqual(plan["required_segment_count"], 4)
            self.assertEqual(plan["effective_minimum_seconds"], 8.75)
            self.assertEqual([item["night_start_seconds"] for item in plan["cuts"]], [0.0, 17.0, 35.0, 52.0])
            self.assertEqual([item["night_end_seconds"] for item in plan["cuts"]], [17.0, 35.0, 52.0, 70.0])
            self.assertTrue(all(item["night_duration_seconds"] <= 45 for item in plan["cuts"]))
            self.assertTrue(all(item["day_duration_seconds"] <= 45 for item in plan["cuts"]))
            self.assertTrue(all(item["start_confidence"] >= 0.65 for item in plan["cuts"]))
            self.assertTrue(all(item["end_confidence"] >= 0.65 for item in plan["cuts"]))
            svg = Path(plan["timeline_svg"]).read_text(encoding="utf-8")
            self.assertIn("Cut 1", svg)
            self.assertIn("Split 0: 10.00s", svg)
            self.assertIn("END Seg 1 / START Seg 2", svg)
            self.assertIn("4 planned segment pair(s), 5 shared splits", svg)
            self.assertIn("effective min 8.75s", svg)
            self.assertIn("max 45.00s", svg)
            self.assertIn("Segment lengths", svg)
            self.assertIn("17.000 seconds", svg)
            self.assertIn("18.000 seconds", svg)
            self.assertIn("unmatched / low confidence", svg)
            self.assertTrue(Path(plan["plan_json"]).is_file())

    def test_native_boundary_selection_allows_long_segment_across_confidence_gap(self):
        rows = []
        for second in range(101):
            confidence = 0.9 if second <= 10 or second >= 55 else 0.4
            rows.append({
                "source_frame": second,
                "source_time_seconds": float(second),
                "target_frame": second,
                "target_time_seconds": float(second),
                "confidence": confidence,
                "status": "matched",
            })
        anchors, _, selection = alignment._select_native_boundary_anchors(rows, 30.0, 20.0, 0.65)
        self.assertEqual(anchors[0]["source_time_seconds"], 0.0)
        self.assertEqual(anchors[-1]["source_time_seconds"], 100.0)
        for start, end in zip(anchors, anchors[1:]):
            self.assertGreaterEqual(end["source_time_seconds"] - start["source_time_seconds"], selection["effective_minimum_seconds"])
            self.assertGreaterEqual(end["target_time_seconds"] - start["target_time_seconds"], selection["effective_minimum_seconds"])
        self.assertGreaterEqual(len(anchors) - 1, 4)
        self.assertTrue(selection["weak_boundary_fallback_used"])
        self.assertTrue(all(
            end["source_time_seconds"] - start["source_time_seconds"] <= 45.0
            for start, end in zip(anchors, anchors[1:])
        ))

    def test_native_rgb_export_consumes_plan_and_rejects_stale_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = root / "day.mp4"
            night = root / "night.mp4"
            mapping = root / "mapping.csv"
            for path in (day, night, mapping):
                path.write_bytes(b"source")
            plan_folder = root / "native_rgb_cut_plan"
            plan_folder.mkdir()
            plan_path = plan_folder / "cut_plan.json"
            plan = {
                "manifest_type": "native_day_night_rgb_cut_plan_v2",
                "sample": "check_mailbox",
                "inputs": {label: alignment._file_fingerprint(path) for label, path in (("mapping", mapping), ("day", day), ("night", night))},
                "cuts": [
                    {"cut_index": index + 1, "start_confidence": 0.8, "end_confidence": 0.9, "boundary_confidence": 0.8, "review": False,
                     "day_start_seconds": 3.0 + index * 25.0, "day_end_seconds": 28.0 + index * 25.0, "day_duration_seconds": 25.0,
                     "night_start_seconds": 7.0 + index * 22.0, "night_end_seconds": 29.0 + index * 22.0, "night_duration_seconds": 22.0}
                    for index in range(4)
                ],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            def run_ffmpeg(command, **kwargs):
                Path(command[-1]).write_bytes(b"clip")
                return type("Completed", (), {"returncode": 0, "stderr": ""})()

            metadata = {"fps": 24.0, "frame_count": 48, "duration_seconds": 2.0, "width": 16, "height": 16}
            with patch.object(alignment.subprocess, "run", side_effect=run_ffmpeg), patch.object(
                alignment, "_video_metadata", return_value=metadata
            ), patch("builtins.print") as print_mock:
                summary = alignment.export_check_mailbox_native_rgb_segments(plan_path)
            self.assertEqual(summary["exported_segment_count"], 4)
            self.assertEqual(summary["segments"][0]["day_start_seconds"], 3.0)
            self.assertEqual(summary["segments"][0]["night_start_seconds"], 7.0)
            self.assertEqual(summary["segments"][0]["day_requested_duration_seconds"], 25.0)
            self.assertEqual(summary["segments"][0]["night_requested_duration_seconds"], 22.0)
            self.assertEqual(summary["segments"][0]["day_fps"], 24.0)
            terminal = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("[segment 1/4] DAY", terminal)
            self.assertIn(str(day), terminal)
            self.assertIn("start=3.000s duration=25.000s", terminal)
            mapping.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "stale"):
                alignment.export_check_mailbox_native_rgb_segments(plan_path)

    def test_native_rgb_export_rejects_version_one_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "cut_plan.json"
            plan_path.write_text(json.dumps({"manifest_type": "native_day_night_rgb_cut_plan_v1"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run option 84 again"):
                alignment.export_check_mailbox_native_rgb_segments(plan_path)

    def test_all_native_rgb_workflow_continues_after_failed_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for sample in ("alpha", "beta"):
                split = dataset / f"{sample}_split"
                split.mkdir(parents=True)
                (split / f"{sample}_day_rgb.mp4").touch()
                (split / f"{sample}_night_rgb.mp4").touch()

            def create_plan(*, sample_name, alignment_folder, **kwargs):
                if sample_name == "beta":
                    raise ValueError("missing reliable anchors")
                return {
                    "cut_count": 4,
                    "plan_json": str(Path(alignment_folder) / "native_rgb_cut_plan" / "cut_plan.json"),
                    "timeline_svg": str(Path(alignment_folder) / "native_rgb_cut_plan" / "cut_plan.svg"),
                }

            def export_plan(plan_path, **kwargs):
                folder = Path(plan_path).parent / "native_rgb_segments_v2"
                return {"output_folder": str(folder), "manifest_json": str(folder / "manifest.json"), "exported_segment_count": 4}

            with patch.object(alignment, "create_native_rgb_cut_plan", side_effect=create_plan), patch.object(
                alignment, "export_native_rgb_segments", side_effect=export_plan
            ), patch("builtins.print") as print_mock:
                summary = alignment.run_all_native_day_night_rgb_cut_plans_and_exports(
                    dataset_folder=dataset,
                    alignment_folder=root / "alignment",
                )
            self.assertEqual(summary["discovered_count"], 2)
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["completed"][0]["sample"], "alpha")
            self.assertEqual(summary["failed"][0]["sample"], "beta")
            self.assertEqual(summary["total_exported_segment_count"], 4)
            self.assertTrue(Path(summary["summary_file"]).is_file())
            terminal = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("[split 1/2] alpha", terminal)
            self.assertIn("Planning from:", terminal)
            self.assertIn("Completed alpha: 4 segment pair(s)", terminal)
            self.assertIn("FAILED beta", terminal)
            self.assertIn("Exported segment pairs: 4", terminal)

    def test_main_menu_contains_option_79_and_heading(self):
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertIn("--- DAY NIGHT PAIR ALIGNMENT ---", source)
        self.assertIn("79. Align wash_cup day/night RGB pair", source)
        self.assertIn("80. Align cut_carrot day/night RGB pair", source)
        self.assertIn("81. Align check_mailbox day/night RGB pair", source)
        self.assertIn("82. Align all day/night RGB pairs", source)
        self.assertIn("--- DAY NIGHT ROBUSTNESS QA ---", source)
        self.assertIn("83. Export check_mailbox aligned day/night RGB frames at 1 FPS", source)
        self.assertIn('elif choice == "79":', source)
        self.assertIn('elif choice == "80":', source)
        self.assertIn('elif choice == "81":', source)
        self.assertIn('elif choice == "82":', source)
        self.assertIn('elif choice == "83":', source)
        self.assertIn("84. Visualize check_mailbox native day/night RGB cut plan", source)
        self.assertIn("85. Export check_mailbox native day/night RGB segments", source)
        self.assertIn("86. Plan and export native day/night RGB segments for all dataset splits", source)
        self.assertIn("Enter choice (1-86 or action id)", source)


if __name__ == "__main__":
    unittest.main()
