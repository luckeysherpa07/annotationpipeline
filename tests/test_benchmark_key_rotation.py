import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import annotation_feature.qa_quality.benchmark as benchmark_module
from annotation_feature.qa_quality.benchmark import (
    BenchmarkJudge,
    BenchmarkModelAdapter,
    OpenAICaptionAdapter,
    QwenLocalCaptionAdapter,
    QwenVLFrameAnswerAdapter,
    build_frame_answer_prompt,
    build_qwen_vl_frame_messages,
    build_model_prompt,
    cleanup_stale_qwen_workers,
    load_api_keys,
    resolve_frame_inputs_for_item,
    run_aligned_qa_benchmark,
    run_gemini_frame_answer_benchmark,
    run_qwen_vl_frame_answer_benchmark,
    select_frames_for_question,
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


class StaticFrameAdapter:
    def answer(self, item, frame_paths):
        return f"frame answer using {len(frame_paths)} frame(s)"


class KeyedFrameAdapter:
    def __init__(self, key):
        self.key = key

    def answer(self, item, frame_paths):
        if self.key == "key-one":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return f"frame answer from {self.key}"


class StaticQwenVLFrameAdapter:
    def answer(self, item, frame_paths):
        return f"qwen vl answer using {len(frame_paths)} frame(s)"


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

    def test_qwen_vllm_adapter_extracts_text_from_fake_llm(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompts, sampling_params):
                self.calls.append({"prompts": prompts, "sampling_params": sampling_params})
                completion = type("Completion", (), {"text": " qwen local answer "})()
                return [type("Generation", (), {"outputs": [completion]})()]

        fake_sampling = object()
        llm = FakeLLM()
        adapter = QwenLocalCaptionAdapter(
            model_name="Qwen/Qwen3-8B",
            engine="vllm",
            llm=llm,
            sampling_params=fake_sampling,
        )

        answer = adapter.answer(
            {
                "modality": "rgb",
                "section": "test",
                "caption": "A caption.",
                "question": "A question?",
            }
        )

        self.assertEqual(answer, "qwen local answer")
        self.assertIn("A caption.", llm.calls[0]["prompts"][0])
        self.assertIs(llm.calls[0]["sampling_params"], fake_sampling)

    def test_qwen_default_adapter_uses_transformers_4bit(self):
        class FakeTokenizer:
            def __call__(self, prompt, **kwargs):
                return {"input_ids": [[1, 2, 3]]}

            def decode(self, tokens, skip_special_tokens=True):
                return " decoded qwen answer "

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return [[1, 2, 3, 4, 5]]

        adapter = QwenLocalCaptionAdapter(
            model_name="Qwen/Qwen3-8B",
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
        )

        self.assertEqual(adapter.engine, "transformers_4bit")
        self.assertEqual(
            adapter.answer(
                {
                    "modality": "rgb",
                    "section": "test",
                    "caption": "A caption.",
                    "question": "A question?",
                }
            ),
            "decoded qwen answer",
        )

    def test_qwen_transformers_4bit_loader_uses_nf4_config(self):
        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch.cuda.is_available", return_value=True):
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.cleanup_stale_qwen_workers"):
                with unittest.mock.patch("annotation_feature.qa_quality.benchmark.BitsAndBytesConfig") as mock_bnb:
                    with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoTokenizer") as mock_tokenizer:
                        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoModelForCausalLM") as mock_model:
                            loaded_model = mock_model.from_pretrained.return_value

                            QwenLocalCaptionAdapter(model_name="Qwen/Qwen3-8B")

                            mock_bnb.assert_called_once_with(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=benchmark_module.torch.float16,
                                bnb_4bit_quant_type="nf4",
                            )
                            mock_tokenizer.from_pretrained.assert_called_once_with(
                                "Qwen/Qwen3-8B",
                                trust_remote_code=True,
                            )
                            mock_model.from_pretrained.assert_called_once_with(
                                "Qwen/Qwen3-8B",
                                trust_remote_code=True,
                                quantization_config=mock_bnb.return_value,
                                device_map="auto",
                            )
                            loaded_model.eval.assert_called_once()

    def test_qwen_vllm_load_uses_memory_saving_defaults(self):
        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.SamplingParams") as mock_sampling:
                with unittest.mock.patch("annotation_feature.qa_quality.benchmark.LLM") as mock_llm:
                    mock_torch.cuda.is_available.return_value = True

                    QwenLocalCaptionAdapter(
                        model_name="Qwen/Qwen3-8B",
                        engine="vllm",
                        cleanup_stale_workers=False,
                    )

                    mock_sampling.assert_called_once_with(temperature=0.0, max_tokens=128)
                    mock_llm.assert_called_once_with(
                        model="Qwen/Qwen3-8B",
                        trust_remote_code=True,
                        max_model_len=1024,
                        gpu_memory_utilization=0.7,
                        enforce_eager=True,
                        dtype="half",
                        max_num_seqs=1,
                    )

    def test_qwen_prompt_is_trimmed_for_local_context(self):
        prompt = build_model_prompt(
            {
                "modality": "rgb",
                "section": "test",
                "caption": "c" * 5000,
                "question": "q" * 1000,
            },
            for_qwen=True,
        )

        self.assertIn("c" * 3000, prompt)
        self.assertNotIn("c" * 3001, prompt)
        self.assertIn("q" * 600, prompt)
        self.assertNotIn("q" * 601, prompt)

    def test_qwen_cleanup_kills_only_vllm_worker_patterns(self):
        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.subprocess.run") as mock_run:
            cleanup_stale_qwen_workers()

        commands = [call.args[0] for call in mock_run.call_args_list]
        self.assertEqual(commands, [["pkill", "-9", "-f", "EngineCore"], ["pkill", "-9", "-f", "vllm"]])

    def test_qwen_cuda_unavailable_guard_raises_before_loading_model(self):
        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                QwenLocalCaptionAdapter._validate_runtime(engine="transformers_4bit", require_cuda=True)

    def test_qwen_provider_writes_separate_output_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            write_valid_items(input_path)
            adapter = CountingAdapter()

            run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                provider="qwen",
                model_name="Qwen/Qwen3-8B",
                max_items=1,
                batch_size=1,
                delay_between_batches=0,
                adapter=adapter,
                judge=StaticJudge(),
            )

            output_json = output_dir / "aligned_qa_benchmark_Qwen_Qwen3-8B.json"
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(output_json.exists())
            self.assertEqual(payload["metadata"]["provider"], "qwen")
            self.assertEqual(payload["metadata"]["model_name"], "Qwen/Qwen3-8B")
            self.assertEqual(payload["results"]["qa-0"]["model_answer"], "counted answer")

    def test_option_69_is_registered(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_QUALITY_QWEN_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_QUALITY_QWEN_BENCHMARK, "69")
        self.assertIn("69", actions)
        self.assertEqual(actions["69"].action_id, "aligned.qa_quality.qwen_benchmark")

    def test_frame_cache_resolution_supports_all_modalities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "aligned_dataset"
            specs = {
                "rgb": (".frames_cache", "scene_day_rgb", "frame_000000.png"),
                "ir": (".frames_cache_ir", "scene_day_ir", "frame_000000.png"),
                "event": (".frames_cache_event", "scene_event", "frame_000000.png"),
            }
            for modality, (cache_subdir, folder_name, frame_name) in specs.items():
                frame_path = root / cache_subdir / "scene_split" / "Seg1" / folder_name / frame_name
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(b"fake")

                frames = resolve_frame_inputs_for_item(
                    {
                        "modality": modality,
                        "pair_key": f"aligned_dataset/scene_split/Seg1/{modality}",
                        "question": "What is shown?",
                    },
                    frame_cache_root=root,
                    max_frames_per_item=6,
                )

                self.assertEqual(frames, [frame_path])

            depth_frame = root / ".frames_cache_marigold" / "scene_split" / "Seg1" / "scene" / "day" / "frame_000000_depth.png"
            depth_frame.parent.mkdir(parents=True, exist_ok=True)
            depth_frame.write_bytes(b"fake")

            depth_frames = resolve_frame_inputs_for_item(
                {
                    "modality": "depth",
                    "pair_key": "aligned_dataset/scene_split/Seg1/depth",
                    "question": "What is shown?",
                },
                frame_cache_root=root,
                max_frames_per_item=6,
            )

            self.assertEqual(depth_frames, [depth_frame])

    def test_frame_sampling_uses_exact_references_and_even_fill(self):
        frames = [Path(f"frame_{index:06d}.png") for index in range(0, 100, 10)]

        selected = select_frames_for_question(frames, "What is visible in frame 000060?", max_frames_per_item=4)

        self.assertEqual(selected[0], Path("frame_000060.png"))
        self.assertEqual(len(selected), 4)

    def test_frame_sampling_zero_uses_all_frames(self):
        frames = [Path(f"frame_{index:06d}.png") for index in range(0, 50, 10)]

        selected = select_frames_for_question(frames, "What is shown?", max_frames_per_item=0)

        self.assertEqual(selected, frames)

    def test_frame_answer_prompt_excludes_caption_and_gold_answer(self):
        prompt = build_frame_answer_prompt(
            {
                "modality": "rgb",
                "section": "test",
                "pair_key": "aligned_dataset/scene_split/Seg1/rgb",
                "question": "What is shown?",
                "caption": "SECRET CAPTION",
                "answer": "SECRET ANSWER",
            },
            [Path("frame_000000.png")],
        )

        self.assertIn("What is shown?", prompt)
        self.assertNotIn("SECRET CAPTION", prompt)
        self.assertNotIn("SECRET ANSWER", prompt)

    def test_frame_answer_benchmark_resume_skips_completed_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            frame_root = root / "aligned_dataset"
            write_valid_items(input_path, count=2)
            for index in range(2):
                frame_path = frame_root / ".frames_cache" / f"pair-{index}" / "rgb" / "frame_000000.png"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(b"fake")

            run_gemini_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=1,
                frame_cache_root=frame_root,
                adapter=StaticFrameAdapter(),
                enable_key_rotation=False,
            )
            run_gemini_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=2,
                frame_cache_root=frame_root,
                adapter=StaticFrameAdapter(),
                enable_key_rotation=False,
            )

            output_json = output_dir / "aligned_qa_frame_answers_gemini-3.1-flash-lite.json"
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["metadata"]["answered_items"], 2)
            self.assertFalse(payload["metadata"]["judge_enabled"])

    def test_frame_answer_key_rotation_retries_same_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            frame_root = root / "aligned_dataset"
            key_file = root / "gemini_api_key_list"
            write_valid_items(input_path)
            key_file.write_text("key-one\nkey-two\n", encoding="utf-8")
            frame_path = frame_root / ".frames_cache" / "pair-0" / "rgb" / "frame_000000.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(b"fake")

            run_gemini_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                frame_cache_root=frame_root,
                api_key_list_path=key_file,
                adapter_factory=lambda key: KeyedFrameAdapter(key),
            )

            output_json = output_dir / "aligned_qa_frame_answers_gemini-3.1-flash-lite.json"
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["results"]["qa-0"]["model_answer"], "frame answer from key-two")
            self.assertEqual(payload["metadata"]["exhausted_key_count"], 1)

    def test_option_70_is_registered(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_FRAME_ANSWER_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_FRAME_ANSWER_BENCHMARK, "70")
        self.assertIn("70", actions)
        self.assertEqual(actions["70"].action_id, "aligned.qa_quality.frame_answer_benchmark")
        self.assertEqual(actions["70"].section, "FRAME INPUT ANSWER BENCHMARK")

    def test_qwen_vl_messages_exclude_caption_and_gold_answer(self):
        messages = build_qwen_vl_frame_messages(
            {
                "modality": "rgb",
                "section": "test",
                "pair_key": "aligned_dataset/scene_split/Seg1/rgb",
                "question": "What is shown?",
                "caption": "SECRET CAPTION",
                "answer": "SECRET ANSWER",
            },
            [Path("frame_000000.png")],
        )

        text_content = messages[0]["content"][-1]["text"]
        self.assertIn("What is shown?", text_content)
        self.assertNotIn("SECRET CAPTION", text_content)
        self.assertNotIn("SECRET ANSWER", text_content)
        self.assertEqual(messages[0]["content"][0]["type"], "image")

    def test_qwen_vl_adapter_extracts_generated_answer_from_fake_model(self):
        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "chat prompt"

            def __call__(self, **kwargs):
                return {"input_ids": [[1, 2, 3]]}

            def batch_decode(self, generated, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                return [" qwen vl decoded answer "]

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return [[1, 2, 3, 4, 5]]

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.process_vision_info") as mock_process:
            mock_process.return_value = (["image"], [])
            adapter = QwenVLFrameAnswerAdapter(
                model_name="Qwen/Qwen3-VL-4B-Instruct",
                model=FakeModel(),
                processor=FakeProcessor(),
            )

            answer = adapter.answer(
                {
                    "modality": "rgb",
                    "section": "test",
                    "pair_key": "aligned_dataset/scene_split/Seg1/rgb",
                    "question": "What is shown?",
                },
                [Path("frame_000000.png")],
            )

        self.assertEqual(answer, "qwen vl decoded answer")

    def test_qwen_vl_cuda_unavailable_guard_raises_before_loading_model(self):
        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                QwenVLFrameAnswerAdapter._validate_runtime(
                    model_name="Qwen/Qwen3-VL-4B-Instruct",
                    require_cuda=True,
                )

    def test_qwen_vl_frame_answer_benchmark_output_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            frame_root = root / "aligned_dataset"
            write_valid_items(input_path, count=2)
            for index in range(2):
                frame_path = frame_root / ".frames_cache" / f"pair-{index}" / "rgb" / "frame_000000.png"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(b"fake")

            run_qwen_vl_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=1,
                frame_cache_root=frame_root,
                adapter=StaticQwenVLFrameAdapter(),
            )
            run_qwen_vl_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=2,
                frame_cache_root=frame_root,
                adapter=StaticQwenVLFrameAdapter(),
            )

            output_json = output_dir / "aligned_qa_frame_answers_Qwen_Qwen3-VL-4B-Instruct.json"
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["metadata"]["provider"], "qwen_vl")
            self.assertEqual(payload["metadata"]["answered_items"], 2)
            self.assertFalse(payload["metadata"]["judge_enabled"])

    def test_option_71_is_registered(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_QWEN_VL_FRAME_ANSWER_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_QWEN_VL_FRAME_ANSWER_BENCHMARK, "71")
        self.assertIn("71", actions)
        self.assertEqual(actions["71"].action_id, "aligned.qa_quality.qwen_vl_frame_answer_benchmark")
        self.assertEqual(actions["71"].section, "FRAME INPUT ANSWER BENCHMARK")


if __name__ == "__main__":
    unittest.main()
