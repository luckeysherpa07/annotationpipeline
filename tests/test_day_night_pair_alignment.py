import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_main_menu_contains_option_79_and_heading(self):
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertIn("--- DAY NIGHT PAIR ALIGNMENT ---", source)
        self.assertIn("79. Align wash_cup day/night RGB pair", source)
        self.assertIn("80. Align cut_carrot day/night RGB pair", source)
        self.assertIn("81. Align check_mailbox day/night RGB pair", source)
        self.assertIn('elif choice == "79":', source)
        self.assertIn('elif choice == "80":', source)
        self.assertIn('elif choice == "81":', source)
        self.assertIn("Enter choice (1-81 or action id)", source)


if __name__ == "__main__":
    unittest.main()
