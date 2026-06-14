import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import annotation_feature.qa_quality.benchmark as benchmark_module
from annotation_feature.qa_quality.benchmark import (
    BenchmarkJudge,
    BenchmarkModelAdapter,
    InternVLFrameAnswerAdapter,
    Molmo2FrameAnswerAdapter,
    OpenAICaptionAdapter,
    QwenLocalCaptionAdapter,
    QwenVLFrameAnswerAdapter,
    QwenVLVideoAnswerAdapter,
    build_frame_answer_prompt,
    build_internvl_frame_prompt,
    build_qwen_vl_frame_messages,
    build_qwen_vl_video_messages,
    build_video_answer_prompt,
    build_model_prompt,
    cleanup_stale_qwen_workers,
    load_api_keys,
    _auto_map_class_ref,
    _ensure_rope_default_compatibility,
    _molmo2_model_class_from_config_module,
    _patch_tie_weights_accepts_extra_kwargs,
    resolve_frame_inputs_for_item,
    resolve_video_input_for_item,
    run_aligned_qa_benchmark,
    run_gemini_frame_answer_benchmark,
    run_internvl_frame_answer_benchmark,
    run_molmo2_frame_answer_benchmark,
    run_qwen_vl_frame_answer_benchmark,
    run_qwen_vl_video_answer_benchmark,
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


def write_rgb_frame_cache(frame_root: Path, pair_index: int, count: int = 30) -> list[Path]:
    frame_dir = frame_root / ".frames_cache" / f"pair-{pair_index}" / "rgb"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        frame_path = frame_dir / f"frame_{index * 30:06d}.png"
        frame_path.write_bytes(b"fake")
        paths.append(frame_path)
    return paths


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


class StaticInternVLFrameAdapter:
    def answer(self, item, frame_paths):
        return f"internvl answer using {len(frame_paths)} frame(s)"


class StaticMolmo2FrameAdapter:
    def answer(self, item, frame_paths):
        return f"molmo2 answer using {len(frame_paths)} frame(s)"


class FailingSecondMolmo2FrameAdapter:
    def answer(self, item, frame_paths):
        if item["qa_id"] == "qa-1":
            raise RuntimeError("molmo2 exploded")
        return f"molmo2 answer using {len(frame_paths)} frame(s)"


class StaticQwenVLVideoAdapter:
    def answer(self, item, video_path):
        return f"qwen vl video answer from {video_path.name}"


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
                write_rgb_frame_cache(frame_root, pair_index=index, count=30)

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
        processor_calls = []

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "chat prompt"

            def __call__(self, **kwargs):
                processor_calls.append(kwargs)
                return {"input_ids": [[1, 2, 3]]}

            def batch_decode(self, generated, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                return [" qwen vl decoded answer "]

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return [[1, 2, 3, 4, 5]]

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.process_vision_info") as mock_process:
            mock_process.return_value = (["image"], [], {"fps": []})
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
        self.assertNotIn("fps", processor_calls[0])

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
                write_rgb_frame_cache(frame_root, pair_index=index, count=30)

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
            self.assertEqual(payload["metadata"]["max_frames_per_item"], 30)
            self.assertFalse(payload["metadata"]["judge_enabled"])
            for result in payload["results"].values():
                self.assertEqual(result["frame_count"], 30)
                self.assertEqual(len(result["frame_paths"]), 30)

    def test_option_71_is_registered(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_QWEN_VL_FRAME_ANSWER_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_QWEN_VL_FRAME_ANSWER_BENCHMARK, "71")
        self.assertIn("71", actions)
        self.assertEqual(actions["71"].action_id, "aligned.qa_quality.qwen_vl_frame_answer_benchmark")
        self.assertEqual(actions["71"].section, "FRAME INPUT ANSWER BENCHMARK")

    def test_internvl_frame_prompt_excludes_caption_and_gold_answer(self):
        prompt = build_internvl_frame_prompt(
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

        self.assertIn("Frame-1: <image>", prompt)
        self.assertIn("What is shown?", prompt)
        self.assertNotIn("SECRET CAPTION", prompt)
        self.assertNotIn("SECRET ANSWER", prompt)

    def test_internvl_adapter_extracts_generated_answer_from_fake_model(self):
        chat_calls = []

        class FakePixelValues:
            def to(self, device):
                return self

        class FakeModel:
            device = None

            def chat(self, tokenizer, pixel_values, prompt, generation_config, **kwargs):
                chat_calls.append(
                    {
                        "tokenizer": tokenizer,
                        "pixel_values": pixel_values,
                        "prompt": prompt,
                        "generation_config": generation_config,
                        "kwargs": kwargs,
                    }
                )
                return " internvl decoded answer "

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.load_internvl_pixel_values") as mock_pixels:
            mock_pixels.return_value = (FakePixelValues(), [1])
            adapter = InternVLFrameAnswerAdapter(
                model_name="OpenGVLab/InternVL2_5-4B",
                model=FakeModel(),
                tokenizer=object(),
                image_size=224,
                max_num_tiles=1,
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

        self.assertEqual(answer, "internvl decoded answer")
        mock_pixels.assert_called_once_with(
            [Path("frame_000000.png")],
            image_size=224,
            max_num_tiles=1,
        )
        self.assertEqual(chat_calls[0]["generation_config"]["do_sample"], False)
        self.assertEqual(chat_calls[0]["kwargs"]["num_patches_list"], [1])

    def test_internvl_cuda_unavailable_guard_raises_before_loading_model(self):
        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                InternVLFrameAnswerAdapter._validate_runtime(require_cuda=True)

    def test_internvl_load_error_includes_original_exception_and_revision(self):
        class DummyPreTrainedModel:
            pass

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoTokenizer") as mock_tokenizer:
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoModel") as mock_model:
                with unittest.mock.patch("annotation_feature.qa_quality.benchmark.BitsAndBytesConfig") as mock_bnb:
                    with unittest.mock.patch("annotation_feature.qa_quality.benchmark.PreTrainedModel", DummyPreTrainedModel):
                        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
                            mock_torch.cuda.is_available.return_value = True
                            mock_torch.bfloat16 = "bfloat16"
                            mock_torch.float16 = "float16"
                            mock_model.from_pretrained.side_effect = ValueError("bitsandbytes exploded")

                            with self.assertRaisesRegex(RuntimeError, "Original error: ValueError: bitsandbytes exploded"):
                                InternVLFrameAnswerAdapter(
                                    model_name="OpenGVLab/InternVL2_5-4B",
                                    revision="abc123",
                                )

        mock_tokenizer.from_pretrained.assert_called_once()
        mock_bnb.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
        self.assertEqual(DummyPreTrainedModel.all_tied_weights_keys, {})
        self.assertEqual(mock_tokenizer.from_pretrained.call_args.kwargs["revision"], "abc123")
        self.assertEqual(mock_model.from_pretrained.call_args.kwargs["revision"], "abc123")
        self.assertEqual(mock_model.from_pretrained.call_args.kwargs["quantization_config"], mock_bnb.return_value)
        self.assertNotIn("load_in_8bit", mock_model.from_pretrained.call_args.kwargs)
        self.assertNotIn("load_in_4bit", mock_model.from_pretrained.call_args.kwargs)

    def test_internvl_frame_answer_benchmark_output_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            frame_root = root / "aligned_dataset"
            write_valid_items(input_path, count=2)
            for index in range(2):
                write_rgb_frame_cache(frame_root, pair_index=index, count=30)

            run_internvl_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=1,
                frame_cache_root=frame_root,
                revision="abc123",
                image_size=224,
                max_num_tiles=1,
                adapter=StaticInternVLFrameAdapter(),
            )
            run_internvl_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=2,
                frame_cache_root=frame_root,
                revision="abc123",
                image_size=224,
                max_num_tiles=1,
                adapter=StaticInternVLFrameAdapter(),
            )

            output_json = output_dir / "aligned_qa_frame_answers_OpenGVLab_InternVL2_5-4B.json"
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["metadata"]["provider"], "internvl")
            self.assertEqual(payload["metadata"]["revision"], "abc123")
            self.assertEqual(payload["metadata"]["answered_items"], 2)
            self.assertEqual(payload["metadata"]["max_frames_per_item"], 30)
            self.assertEqual(payload["metadata"]["internvl_image_size"], 224)
            self.assertEqual(payload["metadata"]["internvl_max_num_tiles"], 1)
            self.assertFalse(payload["metadata"]["judge_enabled"])
            for result in payload["results"].values():
                self.assertEqual(result["frame_count"], 30)
                self.assertEqual(len(result["frame_paths"]), 30)

    def test_option_73_is_registered_with_internvl_4b_label(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_INTERNVL_FRAME_ANSWER_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_INTERNVL_FRAME_ANSWER_BENCHMARK, "73")
        self.assertIn("73", actions)
        self.assertEqual(actions["73"].action_id, "aligned.qa_quality.internvl_frame_answer_benchmark")
        self.assertEqual(actions["73"].section, "FRAME INPUT ANSWER BENCHMARK")
        self.assertIn("InternVL 4B", actions["73"].title)

    def test_molmo2_frame_answer_benchmark_output_resume_failures_and_no_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            frame_root = root / "aligned_dataset"
            write_valid_items(input_path, count=3)
            write_rgb_frame_cache(frame_root, pair_index=0, count=30)
            write_rgb_frame_cache(frame_root, pair_index=1, count=30)

            run_molmo2_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=1,
                frame_cache_root=frame_root,
                adapter=StaticMolmo2FrameAdapter(),
            )
            run_molmo2_frame_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=3,
                batch_size=3,
                frame_cache_root=frame_root,
                adapter=FailingSecondMolmo2FrameAdapter(),
            )

            output_json = output_dir / "aligned_qa_frame_answers_allenai_Molmo2-4B.json"
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["metadata"]["provider"], "molmo2")
            self.assertEqual(payload["metadata"]["model_name"], "allenai/Molmo2-4B")
            self.assertEqual(payload["metadata"]["answered_items"], 1)
            self.assertEqual(payload["metadata"]["skipped_no_frames"], 1)
            self.assertEqual(payload["metadata"]["batch_size"], 1)
            self.assertEqual(payload["metadata"]["quantization"], "4bit_nf4")
            self.assertEqual(payload["metadata"]["max_frames_per_item"], 30)
            self.assertFalse(payload["metadata"]["judge_enabled"])
            self.assertEqual(payload["results"]["qa-0"]["status"], "answered")
            self.assertEqual(payload["results"]["qa-0"]["frame_count"], 30)
            self.assertEqual(payload["results"]["qa-1"]["status"], "failed")
            self.assertIn("molmo2 exploded", payload["results"]["qa-1"]["reason"])
            self.assertNotIn("qa-2", payload["results"])

    def test_molmo2_frame_answer_benchmark_requires_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            write_valid_items(input_path)

            with self.assertRaisesRegex(ValueError, "Molmo2 model name is required"):
                run_molmo2_frame_answer_benchmark(
                    input_path=input_path,
                    output_dir=root / "benchmarks",
                    model_name="",
                    adapter=StaticMolmo2FrameAdapter(),
                )

            with self.assertRaisesRegex(ValueError, "Molmo2 model name is required"):
                Molmo2FrameAnswerAdapter(model_name="", model=object(), processor=object())

    def test_molmo2_adapter_skips_chat_template_when_processor_has_no_template(self):
        process_calls = []
        chat_template_calls = []

        class FakeProcessor:
            chat_template = "template that the runtime still rejects"

            def apply_chat_template(self, *args, **kwargs):
                chat_template_calls.append((args, kwargs))
                raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template")

            def process(self, **kwargs):
                process_calls.append(kwargs)
                return {"input_ids": [1, 2, 3]}

            def decode(self, tokens, skip_special_tokens=True):
                return " molmo answer "

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return [[1, 2, 3, 4, 5]]

        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.Image") as mock_image:
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
                mock_image.open.return_value.convert.return_value = "image"
                mock_torch.no_grad.return_value = NoGrad()
                adapter = Molmo2FrameAnswerAdapter(
                    model_name="allenai/Molmo2-4B",
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

        self.assertEqual(answer, "molmo answer")
        self.assertEqual(chat_template_calls, [])
        self.assertEqual(len(process_calls), 1)
        self.assertEqual(process_calls[0]["text"].count("What is shown?"), 1)
        self.assertIn("<|image|>", process_calls[0]["text"])

    def test_molmo2_adapter_falls_back_to_plain_prompt_when_chat_template_rejected(self):
        processor_calls = []
        chat_template_calls = []

        class FakeProcessor:
            chat_template = "template that the runtime still rejects"

            def apply_chat_template(self, *args, **kwargs):
                chat_template_calls.append((args, kwargs))
                raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template")

            def __call__(self, **kwargs):
                processor_calls.append(kwargs)
                return {"input_ids": [1, 2, 3]}

            def decode(self, tokens, skip_special_tokens=True):
                return " molmo answer "

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return [[1, 2, 3, 4, 5]]

        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.Image") as mock_image:
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
                mock_image.open.return_value.convert.return_value = "image"
                mock_torch.no_grad.return_value = NoGrad()
                adapter = Molmo2FrameAnswerAdapter(
                    model_name="allenai/Molmo2-4B",
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

        self.assertEqual(answer, "molmo answer")
        self.assertEqual(len(chat_template_calls), 1)
        self.assertEqual(len(processor_calls), 1)
        self.assertEqual(processor_calls[0]["text"][0].count("What is shown?"), 1)
        self.assertIn("<|image|>", processor_calls[0]["text"][0])
        self.assertEqual(processor_calls[0]["images"], ["image"])

    def test_molmo2_adapter_uses_image_placeholder_without_chat_template(self):
        processor_calls = []

        class FakeProcessor:
            image_placeholder_token = "<|image|>"

            def __call__(self, **kwargs):
                processor_calls.append(kwargs)
                return {"input_ids": [1, 2, 3]}

            def decode(self, tokens, skip_special_tokens=True):
                return " molmo answer "

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return [[1, 2, 3, 4, 5]]

        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.Image") as mock_image:
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
                mock_image.open.return_value.convert.return_value = "image"
                mock_torch.no_grad.return_value = NoGrad()
                adapter = Molmo2FrameAnswerAdapter(
                    model_name="allenai/Molmo2-4B",
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

        self.assertEqual(answer, "molmo answer")
        self.assertEqual(len(processor_calls), 1)
        self.assertIn("<|image|>", processor_calls[0]["text"][0])
        self.assertEqual(processor_calls[0]["images"], ["image"])

    def test_tie_weights_patch_ignores_unexpected_missing_keys_kwarg(self):
        calls = []

        class FakeMolmoModelClass:
            def tie_weights(self):
                calls.append("called")
                return "tied"

        patched = _patch_tie_weights_accepts_extra_kwargs(FakeMolmoModelClass)

        self.assertTrue(patched)
        self.assertEqual(FakeMolmoModelClass().tie_weights(missing_keys=["lm_head.weight"]), "tied")
        self.assertEqual(calls, ["called"])

    def test_auto_map_class_ref_supports_default_dict_shape(self):
        self.assertEqual(
            _auto_map_class_ref(
                {"AutoModelForImageTextToText": {"default": "modeling_molmo2.Molmo2ForConditionalGeneration"}},
                "AutoModelForImageTextToText",
            ),
            "modeling_molmo2.Molmo2ForConditionalGeneration",
        )
        self.assertEqual(
            _auto_map_class_ref(
                {
                    "AutoModelForImageTextToText": {
                        "default": {
                            "default": ["modeling_molmo2.Molmo2ForConditionalGeneration"]
                        }
                    }
                },
                "AutoModelForImageTextToText",
            ),
            "modeling_molmo2.Molmo2ForConditionalGeneration",
        )

    def test_molmo2_load_sets_tied_weights_compatibility_attribute(self):
        class DummyPreTrainedModel:
            pass

        class FakeMolmoModel:
            def eval(self):
                return None

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoProcessor") as mock_processor:
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoModelForImageTextToText") as mock_model:
                with unittest.mock.patch("annotation_feature.qa_quality.benchmark.BitsAndBytesConfig") as mock_bnb:
                    with unittest.mock.patch("annotation_feature.qa_quality.benchmark.PreTrainedModel", DummyPreTrainedModel):
                        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
                            mock_torch.cuda.is_available.return_value = True
                            mock_torch.float16 = "float16"
                            fake_model = FakeMolmoModel()
                            mock_model.from_pretrained.return_value = fake_model

                            adapter = Molmo2FrameAnswerAdapter(model_name="allenai/Molmo2-4B")

        self.assertIs(adapter.model, fake_model)
        self.assertEqual(DummyPreTrainedModel.all_tied_weights_keys, {})
        self.assertEqual(fake_model.all_tied_weights_keys, {})
        mock_processor.from_pretrained.assert_called_once_with("allenai/Molmo2-4B", trust_remote_code=True)
        mock_bnb.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
        self.assertEqual(mock_model.from_pretrained.call_args.kwargs["quantization_config"], mock_bnb.return_value)
        self.assertEqual(mock_model.from_pretrained.call_args.kwargs["device_map"], "auto")

    def test_molmo2_loader_uses_dynamic_auto_map_when_auto_models_fail(self):
        class FakeConfig:
            auto_map = {
                "AutoModelForImageTextToText": {
                    "default": "modeling_molmo2.Molmo2ForConditionalGeneration",
                }
            }

        class FakeDynamicModel:
            pass

        class FakeDynamicModelClass:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                FakeDynamicModelClass.load_args = args
                FakeDynamicModelClass.load_kwargs = kwargs
                return FakeDynamicModel()

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoProcessor"):
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoModelForImageTextToText") as mock_image_text:
                with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoModelForCausalLM") as mock_causal:
                    with unittest.mock.patch("annotation_feature.qa_quality.benchmark.AutoConfig") as mock_config:
                        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.get_class_from_dynamic_module") as mock_get_class:
                            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.BitsAndBytesConfig") as mock_bnb:
                                with unittest.mock.patch("annotation_feature.qa_quality.benchmark.torch") as mock_torch:
                                    mock_torch.cuda.is_available.return_value = True
                                    mock_torch.float16 = "float16"
                                    mock_image_text.from_pretrained.side_effect = KeyError("default")
                                    mock_causal.from_pretrained.side_effect = ValueError("Unrecognized configuration class")
                                    mock_config.from_pretrained.return_value = FakeConfig()
                                    mock_get_class.return_value = FakeDynamicModelClass

                                    adapter = Molmo2FrameAnswerAdapter(model_name="allenai/Molmo2-4B")

        self.assertIsInstance(adapter.model, FakeDynamicModel)
        mock_get_class.assert_called_with(
            "modeling_molmo2.Molmo2ForConditionalGeneration",
            "allenai/Molmo2-4B",
            trust_remote_code=True,
        )
        self.assertEqual(FakeDynamicModelClass.load_args[0], "allenai/Molmo2-4B")
        self.assertEqual(FakeDynamicModelClass.load_kwargs["quantization_config"], mock_bnb.return_value)
        self.assertEqual(FakeDynamicModelClass.load_kwargs["device_map"], "auto")
        self.assertIs(FakeDynamicModelClass.load_kwargs["config"], mock_config.from_pretrained.return_value)

    def test_molmo2_model_class_can_be_found_from_config_module(self):
        class FakeModelClass:
            pass

        FakeConfig = type(
            "FakeConfig",
            (),
            {"__module__": "transformers_modules.allenai.Molmo2_hyphen_4B.hash.configuration_molmo2"},
        )

        fake_module = type("FakeModule", (), {"Molmo2ForConditionalGeneration": FakeModelClass})()

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark.importlib.import_module", return_value=fake_module) as mock_import:
            model_class = _molmo2_model_class_from_config_module(FakeConfig())

        self.assertIs(model_class, FakeModelClass)
        mock_import.assert_called_with("transformers_modules.allenai.Molmo2_hyphen_4B.hash.modeling_molmo2")

    def test_rope_default_compatibility_registers_missing_default_key(self):
        fake_registry = {}

        with unittest.mock.patch.dict("sys.modules", {}):
            fake_module = type("FakeRopeModule", (), {"ROPE_INIT_FUNCTIONS": fake_registry})()
            with unittest.mock.patch.dict("sys.modules", {"transformers.modeling_rope_utils": fake_module}):
                _ensure_rope_default_compatibility()

        self.assertIn("default", fake_registry)

    def test_option_74_is_registered_with_molmo2_4b_label(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_MOLMO2_FRAME_ANSWER_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_MOLMO2_FRAME_ANSWER_BENCHMARK, "74")
        self.assertIn("74", actions)
        self.assertEqual(actions["74"].action_id, "aligned.qa_quality.molmo2_frame_answer_benchmark")
        self.assertEqual(actions["74"].section, "FRAME INPUT ANSWER BENCHMARK")
        self.assertIn("Molmo2-4B", actions["74"].title)

    def test_video_resolution_supports_modalities_and_excludes_rgb_with_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "aligned_dataset"
            segment = root / "scene_split" / "Seg1"
            segment.mkdir(parents=True)
            files = [
                "scene_day_rgb_with_audio.mp4",
                "scene_night_rgb.mp4",
                "scene_day_rgb.mp4",
                "scene_day_ir.mp4",
                "scene_day_event.mp4",
                "scene_day_depth.mp4",
            ]
            for name in files:
                (segment / name).write_bytes(b"fake")

            self.assertEqual(
                resolve_video_input_for_item(
                    {"modality": "rgb", "pair_key": "aligned_dataset/scene_split/Seg1/rgb"},
                    dataset_root=root,
                ),
                segment / "scene_day_rgb.mp4",
            )
            self.assertEqual(
                resolve_video_input_for_item(
                    {"modality": "ir", "pair_key": "aligned_dataset/scene_split/Seg1/ir"},
                    dataset_root=root,
                ),
                segment / "scene_day_ir.mp4",
            )
            self.assertEqual(
                resolve_video_input_for_item(
                    {"modality": "event", "pair_key": "aligned_dataset/scene_split/Seg1/event"},
                    dataset_root=root,
                ),
                segment / "scene_day_event.mp4",
            )
            self.assertEqual(
                resolve_video_input_for_item(
                    {"modality": "depth", "pair_key": "aligned_dataset/scene_split/Seg1/depth"},
                    dataset_root=root,
                ),
                segment / "scene_day_depth.mp4",
            )

    def test_qwen_vl_video_message_excludes_caption_and_gold_answer(self):
        message = build_qwen_vl_video_messages(
            {
                "modality": "rgb",
                "section": "test",
                "pair_key": "aligned_dataset/scene_split/Seg1/rgb",
                "question": "What is shown?",
                "caption": "SECRET CAPTION",
                "answer": "SECRET ANSWER",
            },
            Path("scene_day_rgb.mp4"),
            video_fps=1.0,
            video_frames=["frame"],
            raw_fps=30.0,
        )

        text_content = message[0]["content"][-1]["text"]
        self.assertEqual(message[0]["content"][0]["type"], "video")
        self.assertEqual(message[0]["content"][0]["video"], ["frame"])
        self.assertEqual(message[0]["content"][0]["sample_fps"], 1.0)
        self.assertEqual(message[0]["content"][0]["raw_fps"], 30.0)
        self.assertNotIn("nframes", message[0]["content"][0])
        self.assertIn("What is shown?", text_content)
        self.assertNotIn("SECRET CAPTION", text_content)
        self.assertNotIn("SECRET ANSWER", text_content)

    def test_qwen_vl_video_adapter_extracts_generated_answer_from_fake_model(self):
        process_messages = []

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "chat prompt"

            def __call__(self, **kwargs):
                return {"input_ids": [[1, 2, 3]]}

            def batch_decode(self, generated, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                return [" qwen vl video decoded answer "]

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return [[1, 2, 3, 4, 5]]

        with unittest.mock.patch("annotation_feature.qa_quality.benchmark._sample_video_frames_with_opencv") as mock_sample:
            mock_sample.return_value = (["decoded-frame"], 30.0)
            with unittest.mock.patch("annotation_feature.qa_quality.benchmark.process_vision_info") as mock_process:
                def fake_process(messages, return_video_kwargs=True):
                    process_messages.append(messages)
                    return ([], ["video"], {"fps": 1.0})

                mock_process.side_effect = fake_process
                adapter = QwenVLVideoAnswerAdapter(
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
                    Path("scene_day_rgb.mp4"),
                )

        self.assertEqual(answer, "qwen vl video decoded answer")
        self.assertEqual(process_messages[0][0]["content"][0]["video"], ["decoded-frame"])
        mock_sample.assert_called_once_with(Path("scene_day_rgb.mp4"), 1.0)

    def test_qwen_vl_video_answer_benchmark_output_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "valid.json"
            output_dir = root / "benchmarks"
            dataset_root = root / "aligned_dataset"
            items = []
            for index in range(2):
                items.append(
                    {
                        "qa_id": f"qa-{index}",
                        "modality": "rgb",
                        "section": "test",
                        "pair_key": f"aligned_dataset/scene_split/Seg{index + 1}/rgb",
                        "question": "What is shown?",
                        "answer": "a test action",
                        "caption": "A person performs a test action.",
                    }
                )
                segment = dataset_root / "scene_split" / f"Seg{index + 1}"
                segment.mkdir(parents=True, exist_ok=True)
                (segment / "scene_day_rgb.mp4").write_bytes(b"fake")
            input_path.write_text(json.dumps({"valid_qa": items}), encoding="utf-8")

            run_qwen_vl_video_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=1,
                dataset_root=dataset_root,
                adapter=StaticQwenVLVideoAdapter(),
            )
            run_qwen_vl_video_answer_benchmark(
                input_path=input_path,
                output_dir=output_dir,
                max_items=2,
                dataset_root=dataset_root,
                adapter=StaticQwenVLVideoAdapter(),
            )

            output_json = output_dir / "aligned_qa_video_answers_Qwen_Qwen3-VL-4B-Instruct.json"
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["metadata"]["provider"], "qwen_vl")
            self.assertEqual(payload["metadata"]["benchmark_type"], "video_input_answer_generation")
            self.assertEqual(payload["metadata"]["answered_items"], 2)
            self.assertFalse(payload["metadata"]["judge_enabled"])
            self.assertNotIn("max_video_frames", payload["metadata"])
            self.assertNotIn("max_video_frames", next(iter(payload["results"].values())))

    def test_option_72_is_registered(self):
        from annotation_feature.cli.actions.aligned_choices import ALIGNED_QA_QWEN_VL_VIDEO_ANSWER_BENCHMARK
        from annotation_feature.cli.actions.aligned_qa_quality import build_aligned_qa_quality_actions

        actions = build_aligned_qa_quality_actions(confirm=lambda prompt: False)

        self.assertEqual(ALIGNED_QA_QWEN_VL_VIDEO_ANSWER_BENCHMARK, "72")
        self.assertIn("72", actions)
        self.assertEqual(actions["72"].action_id, "aligned.qa_quality.qwen_vl_video_answer_benchmark")
        self.assertEqual(actions["72"].section, "FRAME INPUT ANSWER BENCHMARK")


if __name__ == "__main__":
    unittest.main()
