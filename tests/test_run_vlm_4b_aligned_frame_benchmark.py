import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_vlm_4b_aligned_frame_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("run_vlm_4b_benchmark", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FixedFrameBenchmarkTests(unittest.TestCase):
    def _manifest(self):
        paths = [
            *(f"aligned_dataset/example/day/frame_{index:06d}.jpg" for index in range(4)),
            *(f"aligned_dataset/example/night/frame_{index:06d}.jpg" for index in range(4)),
        ]
        items = [{"qa_id": "qa-1", "frame_paths": paths}]
        return {
            "metadata": {
                "manifest_sha256": MODULE._json_hash(items),
                "total_frames": 8,
                "frames_per_side": 4,
                "frame_order": "day_then_night",
                "sampling_algorithm": MODULE.DEFAULT_SAMPLING_ALGORITHM,
            },
            "items": items,
        }

    def test_validate_manifest_accepts_valid_content(self):
        manifest = self._manifest()
        MODULE._validate_manifest(
            manifest,
            Path("frames_8.json"),
            manifest["metadata"]["manifest_sha256"],
        )

    def test_validate_manifest_rejects_changed_content_with_stale_hash(self):
        manifest = self._manifest()
        expected_hash = manifest["metadata"]["manifest_sha256"]
        manifest["items"][0]["frame_paths"][0] = "aligned_dataset/example/night/frame.jpg"
        with self.assertRaisesRegex(RuntimeError, "manifest_sha256"):
            MODULE._validate_manifest(manifest, Path("frames_8.json"), expected_hash)

    def test_validate_manifest_rejects_wrong_frame_order(self):
        manifest = self._manifest()
        manifest["items"][0]["frame_paths"].reverse()
        manifest["metadata"]["manifest_sha256"] = MODULE._json_hash(manifest["items"])
        with self.assertRaisesRegex(ValueError, "day_then_night"):
            MODULE._validate_manifest(
                manifest,
                Path("frames_8.json"),
                manifest["metadata"]["manifest_sha256"],
            )

    def test_load_results_rejects_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "model_name": "old-model",
                            "frame_manifest_sha256": "hash",
                        },
                        "results": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "model_name"):
                MODULE._load_results(
                    path,
                    model_name="new-model",
                    manifest_sha256="hash",
                )

    def test_completed_requires_nonempty_answer(self):
        self.assertTrue(MODULE._completed({"status": "answered", "model_answer": "yes"}))
        self.assertFalse(MODULE._completed({"status": "answered", "model_answer": ""}))
        self.assertFalse(
            MODULE._completed(
                {
                    "status": "answered",
                    "model_answer": "yes",
                    "reason": "Frame answer call failed: stale result",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
