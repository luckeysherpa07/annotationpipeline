import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


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

    def test_group_rows_by_frame_set_makes_repeated_frames_contiguous(self):
        paths_a = [Path("day/a.jpg"), Path("night/a.jpg")]
        paths_b = [Path("day/b.jpg"), Path("night/b.jpg")]
        rows = [
            ({"qa_id": "a1"}, paths_a),
            ({"qa_id": "b1"}, paths_b),
            ({"qa_id": "a2"}, paths_a),
        ]
        grouped = MODULE._group_rows_by_frame_set(rows)
        self.assertEqual([item["qa_id"] for item, _ in grouped], ["a1", "a2", "b1"])

    def test_rgb_cache_reuses_one_set_and_closes_it_on_replacement(self):
        cache = MODULE._SingleRGBFrameSetCache()
        cache._init_frame_cache()
        opened = []

        class FakeImage:
            def __init__(self, name):
                self.name = name
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def convert(self, _mode):
                converted = FakeImage(self.name + "-rgb")
                opened.append(converted)
                return converted

            def close(self):
                self.closed = True

        with patch.object(MODULE.Image, "open", side_effect=lambda path: FakeImage(str(path))):
            first = cache._rgb_frames([Path("a.jpg")])
            repeated = cache._rgb_frames([Path("a.jpg")])
            second = cache._rgb_frames([Path("b.jpg")])

        self.assertIs(first, repeated)
        self.assertTrue(first[0].closed)
        self.assertFalse(second[0].closed)
        self.assertEqual(cache.frame_cache_hits, 1)
        self.assertEqual(cache.frame_cache_misses, 2)
        cache.clear_frame_cache()
        self.assertTrue(second[0].closed)

    def test_internvl_cache_reuses_preprocessed_tensor(self):
        adapter = MODULE.CachedInternVLFrameAnswerAdapter.__new__(
            MODULE.CachedInternVLFrameAnswerAdapter
        )
        adapter._cached_frame_key = None
        adapter._cached_pixel_values = None
        adapter._cached_num_patches = []
        adapter.frame_cache_hits = 0
        adapter.frame_cache_misses = 0
        adapter.image_size = 448
        adapter.max_num_tiles = 1
        adapter.model = type("FakeModel", (), {"device": "cpu"})()
        first_tensor = type(
            "FakeTensor",
            (),
            {"to": lambda self, _device: self},
        )()
        second_tensor = type(
            "FakeTensor",
            (),
            {"to": lambda self, _device: self},
        )()

        with patch.object(
            MODULE,
            "load_internvl_pixel_values",
            side_effect=[(first_tensor, [1]), (second_tensor, [1])],
        ) as loader:
            first = adapter._frame_inputs([Path("a.jpg")])
            repeated = adapter._frame_inputs([Path("a.jpg")])
            second = adapter._frame_inputs([Path("b.jpg")])

        self.assertIs(first[0], repeated[0])
        self.assertIs(second[0], second_tensor)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(adapter.frame_cache_hits, 1)
        self.assertEqual(adapter.frame_cache_misses, 2)

    def test_molmo2_rope_compatibility_registers_default(self):
        original = MODULE.ROPE_INIT_FUNCTIONS
        MODULE.ROPE_INIT_FUNCTIONS = {}
        try:
            MODULE._ensure_molmo2_rope_compatibility()
            config = type(
                "Config",
                (),
                {
                    "hidden_size": 16,
                    "num_attention_heads": 2,
                    "rope_theta": 10000.0,
                },
            )()
            inv_freq, scaling = MODULE.ROPE_INIT_FUNCTIONS["default"](config)
        finally:
            MODULE.ROPE_INIT_FUNCTIONS = original

        self.assertEqual(tuple(inv_freq.shape), (4,))
        self.assertEqual(scaling, 1.0)

    def test_molmo2_generation_compatibility_creates_cache_position(self):
        calls = []

        class FakeModel:
            def prepare_inputs_for_generation(
                self,
                input_ids,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                calls.append(cache_position)
                return {"cache_position": cache_position}

        model = FakeModel()
        MODULE._ensure_molmo2_generation_compatibility(model)
        input_ids = MODULE.torch.zeros((1, 5), dtype=MODULE.torch.long)
        result = model.prepare_inputs_for_generation(input_ids)
        self.assertEqual(result["cache_position"].tolist(), [0, 1, 2, 3, 4])

        cache = type("FakeCache", (), {"get_seq_length": lambda self: 5})()
        result = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=cache,
            next_sequence_length=1,
        )
        self.assertEqual(result["cache_position"].tolist(), [5])

    def test_molmo2_mask_compatibility_renames_legacy_arguments(self):
        module_name = "fake_molmo2_modeling"
        module = ModuleType(module_name)
        calls = []

        def mask_function(*args, **kwargs):
            calls.append((args, kwargs))
            return kwargs

        module.create_causal_mask = mask_function
        module.create_masks_for_generate = mask_function
        original_module = MODULE.sys.modules.get(module_name)
        MODULE.sys.modules[module_name] = module
        fake_class = type("FakeMolmo2", (), {})
        fake_class.__module__ = module_name
        try:
            MODULE._ensure_molmo2_mask_compatibility(fake_class())
            result = module.create_causal_mask(
                input_embeds="embeds",
                cache_position="legacy",
            )
        finally:
            if original_module is None:
                MODULE.sys.modules.pop(module_name, None)
            else:
                MODULE.sys.modules[module_name] = original_module

        self.assertEqual(result, {"inputs_embeds": "embeds"})
        self.assertEqual(calls[0][1], {"inputs_embeds": "embeds"})


if __name__ == "__main__":
    unittest.main()
