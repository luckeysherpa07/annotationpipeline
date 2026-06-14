import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_vlm_4b_aligned_frame_benchmark_cu130.py"
)
SPEC = importlib.util.spec_from_file_location("run_vlm_4b_benchmark_cu130", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Cu130FixedFrameBenchmarkTests(unittest.TestCase):
    def test_uses_independent_experiment_directory(self):
        self.assertEqual(
            MODULE.DEFAULT_EXPERIMENT_DIR,
            Path("outputs/benchmarks/vlm_8frame_aligned_4b"),
        )

    def test_molmo2_uses_local_adapter(self):
        with patch.object(
            MODULE.LocalMolmo2FrameAnswerAdapter,
            "__init__",
            return_value=None,
        ):
            adapter = MODULE._adapter_for("molmo2", "models/molmo2/Molmo2-4B")
        self.assertIsInstance(adapter, MODULE.LocalMolmo2FrameAnswerAdapter)

    def test_saved_metadata_identifies_runtime_and_model_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}',
                encoding="utf-8",
            )
            output_json = root / "result.json"
            output_csv = root / "result.csv"
            MODULE._save_frame_answer_outputs(
                output_json,
                output_csv,
                {},
                {"model_name": model_dir.as_posix()},
            )
            metadata = json.loads(output_json.read_text(encoding="utf-8"))["metadata"]

        self.assertEqual(
            metadata["benchmark_type"],
            "fixed_8frame_aligned_4b_frame_input_cu130_v1",
        )
        self.assertEqual(
            metadata["runtime_environment"]["runner"],
            SCRIPT_PATH.name,
        )
        self.assertTrue(metadata["model_index_sha256"])

    def test_runner_checkpoints_in_batches_and_saves_final_remainder(self):
        rows = [
            (
                {
                    "qa_id": f"qa-{index}",
                    "modality": "rgb",
                    "section": "section",
                    "pair_key": "pair",
                    "question": "question",
                    "answer": "answer",
                },
                [Path("day/frame.jpg"), Path("night/frame.jpg")],
            )
            for index in range(3)
        ]
        manifest = {
            "metadata": {
                "manifest_sha256": "hash",
                "total_frames": 2,
                "frames_per_side": 1,
                "frame_order": "day_then_night",
                "sampling_algorithm": "test",
            }
        }
        adapter = type(
            "Adapter",
            (),
            {
                "quantization": "test",
                "frame_cache_hits": 0,
                "frame_cache_misses": 0,
                "last_input_stats": {},
                "answer": lambda self, _item, _paths: "model answer",
            },
        )()

        with (
            patch.object(MODULE.shared, "_load_results", return_value={}),
            patch.object(MODULE.shared, "manifest_items", return_value=rows),
            patch.object(MODULE.shared, "_group_rows_by_frame_set", return_value=rows),
            patch.object(
                MODULE.shared,
                "_result_output_paths",
                return_value=(Path("result.json"), Path("result.csv")),
            ),
            patch.object(MODULE, "_save_frame_answer_outputs") as save,
            patch.dict("os.environ", {"VLM_CHECKPOINT_EVERY_ITEMS": "2"}),
        ):
            MODULE._run_fixed_frame_model(
                label="test",
                model_name="model",
                adapter=adapter,
                input_path=Path("input.json"),
                frame_manifest=manifest,
                frame_manifest_path=Path("frames.json"),
                output_dir=Path("output"),
                resume=True,
            )

        self.assertEqual(save.call_count, 2)
        self.assertEqual(len(save.call_args_list[0].args[2]), 3)
        self.assertEqual(len(save.call_args_list[1].args[2]), 3)
        first_metadata = save.call_args_list[0].args[3]
        self.assertEqual(first_metadata["attempted_items"], 2)
        self.assertEqual(first_metadata["checkpoint_every_items"], 2)

    def test_runner_saves_pending_results_on_keyboard_interrupt(self):
        rows = [
            (
                {
                    "qa_id": f"qa-{index}",
                    "modality": "rgb",
                    "section": "section",
                    "pair_key": "pair",
                    "question": "question",
                    "answer": "answer",
                },
                [Path("day/frame.jpg"), Path("night/frame.jpg")],
            )
            for index in range(2)
        ]
        manifest = {
            "metadata": {
                "manifest_sha256": "hash",
                "total_frames": 2,
                "frames_per_side": 1,
                "frame_order": "day_then_night",
                "sampling_algorithm": "test",
            }
        }
        adapter = type(
            "Adapter",
            (),
            {
                "quantization": "test",
                "frame_cache_hits": 0,
                "frame_cache_misses": 0,
                "last_input_stats": {},
            },
        )()
        adapter.answer = unittest.mock.Mock(
            side_effect=["model answer", KeyboardInterrupt()]
        )

        with (
            patch.object(MODULE.shared, "_load_results", return_value={}),
            patch.object(MODULE.shared, "manifest_items", return_value=rows),
            patch.object(MODULE.shared, "_group_rows_by_frame_set", return_value=rows),
            patch.object(
                MODULE.shared,
                "_result_output_paths",
                return_value=(Path("result.json"), Path("result.csv")),
            ),
            patch.object(MODULE, "_save_frame_answer_outputs") as save,
            patch.dict("os.environ", {"VLM_CHECKPOINT_EVERY_ITEMS": "25"}),
            self.assertRaises(KeyboardInterrupt),
        ):
            MODULE._run_fixed_frame_model(
                label="test",
                model_name="model",
                adapter=adapter,
                input_path=Path("input.json"),
                frame_manifest=manifest,
                frame_manifest_path=Path("frames.json"),
                output_dir=Path("output"),
                resume=True,
            )

        self.assertEqual(save.call_count, 1)
        self.assertEqual(set(save.call_args.args[2]), {"qa-0"})


if __name__ == "__main__":
    unittest.main()
