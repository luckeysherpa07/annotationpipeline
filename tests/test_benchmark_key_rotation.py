import json
import tempfile
import unittest
from pathlib import Path

from annotation_feature.qa_quality.benchmark import (
    BenchmarkJudge,
    BenchmarkModelAdapter,
    OpenAICaptionAdapter,
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


class CountingAdapter(BenchmarkModelAdapter):
    provider = "openai"
    model_name = "test-model"

    def __init__(self):
        self.calls = 0

    def answer(self, item):
        self.calls += 1
        return "counted answer"


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

    def test_load_openai_api_keys_supports_labeled_lines_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openai_api_key_list"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "sk-plain-key",
                        "OPENAI_API_KEY='sk-env-key'",
                        "personal key number 1 = sk-labeled-key",
                        "OTHER=value",
                        "sk-plain-key",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_api_keys(path, env_var_name="OPENAI_API_KEY", key_prefixes=("sk-",)),
                ["sk-plain-key", "sk-env-key", "sk-labeled-key"],
            )

    def test_openai_adapter_extracts_text_from_fake_responses_client(self):
        class FakeResponses:
            def __init__(self):
                self.calls = []

            def create(self, model, input):
                self.calls.append({"model": model, "input": input})
                return type("Response", (), {"output_text": "fake openai answer"})()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        client = FakeClient()
        adapter = OpenAICaptionAdapter(model_name="gpt-test", client=client)
        answer = adapter.answer(
            {
                "modality": "rgb",
                "section": "test",
                "caption": "A caption.",
                "question": "A question?",
            }
        )

        self.assertEqual(answer, "fake openai answer")
        self.assertEqual(client.responses.calls[0]["model"], "gpt-test")
        self.assertIn("A caption.", client.responses.calls[0]["input"])

    def test_openai_quota_error_rotates_key_and_retries_same_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            openai_key_path = root / "openai_api_key_list"
            output_dir = root / "benchmarks"
            write_valid_items(input_path)
            openai_key_path.write_text("sk-key-one\nsk-key-two\n", encoding="utf-8")
            used_keys = []

            class OpenAIKeyedAdapter(BenchmarkModelAdapter):
                provider = "openai"
                model_name = "test-model"

                def __init__(self, key):
                    self.key = key

                def answer(self, item):
                    if self.key == "sk-key-one":
                        raise RuntimeError("rate_limit_exceeded")
                    return f"answer from {self.key}"

            def adapter_factory(key):
                used_keys.append(key)
                return OpenAIKeyedAdapter(key)

            run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                provider="openai",
                model_name="gpt-5.4-mini",
                max_items=1,
                batch_size=1,
                delay_between_batches=0,
                openai_api_key_list_path=openai_key_path,
                judge_factory=lambda key: StaticJudge(),
                adapter_factory=adapter_factory,
            )

            payload = json.loads((output_dir / "aligned_qa_benchmark_gpt-5.4-mini.json").read_text(encoding="utf-8"))
            self.assertEqual(used_keys, ["sk-key-one", "sk-key-two"])
            self.assertEqual(payload["results"]["qa-0"]["model_answer"], "answer from sk-key-two")
            self.assertEqual(payload["metadata"]["provider"], "openai")
            self.assertEqual(payload["metadata"]["keys_available"], 2)

    def test_gemini_judge_rotation_does_not_regenerate_openai_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            openai_key_path = root / "openai_api_key_list"
            judge_key_path = root / "gemini_api_key_list"
            output_dir = root / "benchmarks"
            write_valid_items(input_path)
            openai_key_path.write_text("sk-openai-one\n", encoding="utf-8")
            judge_key_path.write_text("AIza-judge-one\nAIza-judge-two\n", encoding="utf-8")
            adapter = CountingAdapter()
            judge_keys = []

            class RotatingJudge(BenchmarkJudge):
                def __init__(self, key):
                    self.key = key

                def judge(self, item, model_answer):
                    if self.key == "AIza-judge-one":
                        raise RuntimeError("429 RESOURCE_EXHAUSTED")
                    return {"score": "correct", "numeric_score": 1.0, "reason": "matches"}

            def judge_factory(key):
                judge_keys.append(key)
                return RotatingJudge(key)

            run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                provider="openai",
                model_name="gpt-5.4-mini",
                max_items=1,
                batch_size=1,
                delay_between_batches=0,
                openai_api_key_list_path=openai_key_path,
                judge_api_key_list_path=judge_key_path,
                adapter=adapter,
                judge_factory=judge_factory,
            )

            payload = json.loads((output_dir / "aligned_qa_benchmark_gpt-5.4-mini.json").read_text(encoding="utf-8"))
            self.assertEqual(adapter.calls, 1)
            self.assertEqual(judge_keys, ["AIza-judge-one", "AIza-judge-two"])
            self.assertEqual(payload["results"]["qa-0"]["score"], "correct")
            self.assertEqual(payload["metadata"]["exhausted_judge_key_count"], 1)

    def test_option_68_is_registered(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_QUALITY_GPT_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_QUALITY_GPT_BENCHMARK, "68")
        self.assertIn("68", actions)
        self.assertEqual(actions["68"].action_id, "aligned.qa_quality.gpt_benchmark")


if __name__ == "__main__":
    unittest.main()
