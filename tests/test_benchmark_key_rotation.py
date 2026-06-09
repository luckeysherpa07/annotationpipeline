import json
import tempfile
import unittest
from pathlib import Path

from annotation_feature.qa_quality.benchmark import (
    BenchmarkJudge,
    BenchmarkModelAdapter,
    load_api_keys,
    run_aligned_qa_benchmark,
)


def write_valid_items(path: Path, count: int = 1) -> None:
    items = []
    for index in range(count):
        items.append(
            {
                "qa_id": f"qa-{index}",
                "modality": "rgb",
                "section": "test",
                "pair_key": f"pair-{index}",
                "question": "What is shown?",
                "answer": "a test action",
                "caption": "A person performs a test action.",
            }
        )
    path.write_text(json.dumps({"valid_qa": items}), encoding="utf-8")


class StaticJudge(BenchmarkJudge):
    def judge(self, item, model_answer):
        return {"score": "correct", "numeric_score": 1.0, "reason": "matches"}


class KeyedAdapter(BenchmarkModelAdapter):
    provider = "gemini"
    model_name = "test-model"

    def __init__(self, key: str | None):
        self.key = key

    def answer(self, item):
        if self.key == "key-one":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return f"answer from {self.key}"


class QuotaAdapter(BenchmarkModelAdapter):
    provider = "gemini"
    model_name = "test-model"

    def answer(self, item):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


class InterruptingAdapter(BenchmarkModelAdapter):
    provider = "gemini"
    model_name = "test-model"

    def __init__(self):
        self.calls = 0

    def answer(self, item):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt()
        return "answer before interrupt"


class BenchmarkKeyRotationTests(unittest.TestCase):
    def test_load_api_keys_supports_plain_env_lines_comments_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_key_list"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "plain-key",
                        "GEMINI_API_KEY='env-key'",
                        "personal api key number 1 = AIza-metadata-key",
                        "OTHER=value",
                        "plain-key",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_api_keys(path), ["plain-key", "env-key", "AIza-metadata-key"])

    def test_quota_error_rotates_key_and_retries_same_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            key_path = root / "api_key_list"
            output_dir = root / "benchmarks"
            write_valid_items(input_path)
            key_path.write_text("key-one\nkey-two\n", encoding="utf-8")
            used_keys = []

            def adapter_factory(key):
                used_keys.append(key)
                return KeyedAdapter(key)

            run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                model_name="test-model",
                max_items=1,
                batch_size=1,
                delay_between_batches=0,
                api_key_list_path=key_path,
                adapter_factory=adapter_factory,
                judge_factory=lambda key: StaticJudge(),
            )

            payload = json.loads((output_dir / "aligned_qa_benchmark_test-model.json").read_text(encoding="utf-8"))
            self.assertEqual(used_keys, ["key-one", "key-two"])
            self.assertEqual(payload["results"]["qa-0"]["model_answer"], "answer from key-two")
            self.assertEqual(payload["metadata"]["keys_available"], 2)
            self.assertEqual(payload["metadata"]["exhausted_key_count"], 1)

    def test_invalid_key_error_rotates_key_and_retries_same_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            key_path = root / "api_key_list"
            output_dir = root / "benchmarks"
            write_valid_items(input_path)
            key_path.write_text("key-one\nkey-two\n", encoding="utf-8")

            class InvalidKeyAdapter(KeyedAdapter):
                def answer(self, item):
                    if self.key == "key-one":
                        raise RuntimeError("400 INVALID_ARGUMENT API_KEY_INVALID API key not valid")
                    return f"answer from {self.key}"

            run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                model_name="test-model",
                max_items=1,
                batch_size=1,
                delay_between_batches=0,
                api_key_list_path=key_path,
                adapter_factory=lambda key: InvalidKeyAdapter(key),
                judge_factory=lambda key: StaticJudge(),
            )

            payload = json.loads((output_dir / "aligned_qa_benchmark_test-model.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["results"]["qa-0"]["model_answer"], "answer from key-two")
            self.assertEqual(payload["metadata"]["exhausted_key_count"], 1)

    def test_all_keys_exhausted_saves_progress_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            key_path = root / "api_key_list"
            output_dir = root / "benchmarks"
            write_valid_items(input_path)
            key_path.write_text("key-one\n", encoding="utf-8")

            run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                model_name="test-model",
                max_items=1,
                batch_size=1,
                delay_between_batches=0,
                api_key_list_path=key_path,
                adapter_factory=lambda key: QuotaAdapter(),
                judge_factory=lambda key: StaticJudge(),
            )

            payload = json.loads((output_dir / "aligned_qa_benchmark_test-model.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["results"], {})
            self.assertEqual(payload["metadata"]["stopped_reason"], "quota_or_rate_limit")
            self.assertEqual(payload["metadata"]["exhausted_key_count"], 1)

    def test_keyboard_interrupt_preserves_completed_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            adapter = InterruptingAdapter()
            write_valid_items(input_path, count=2)

            run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                model_name="test-model",
                max_items=2,
                batch_size=2,
                delay_between_batches=0,
                enable_key_rotation=False,
                adapter=adapter,
                judge=StaticJudge(),
            )

            payload = json.loads((output_dir / "aligned_qa_benchmark_test-model.json").read_text(encoding="utf-8"))
            self.assertEqual(list(payload["results"]), ["qa-0"])
            self.assertEqual(payload["metadata"]["stopped_reason"], "user_cancelled")


if __name__ == "__main__":
    unittest.main()
