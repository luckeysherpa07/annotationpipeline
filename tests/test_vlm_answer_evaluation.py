import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from annotation_feature.qa_quality.answer_judge import (
    judge_prompt_sha256,
    run_llm_judge,
)
from annotation_feature.qa_quality.answer_metrics import (
    anls,
    boolean_accuracy,
    deterministic_metrics,
    numeric_accuracy,
    rouge_l_f1,
    token_prf,
)
from annotation_feature.qa_quality.evaluation_report import (
    pairwise_judge_comparisons,
    score_records,
    write_evaluation_outputs,
)
from annotation_feature.qa_quality.metric_router import route_metric
from annotation_feature.qa_quality.result_loader import (
    EvaluationRecord,
    load_evaluation_records,
    load_result_file,
)


def record(record_id="record-1", qa_id="qa-1", answer="two"):
    return EvaluationRecord(
        record_id=record_id,
        source_path="result.json",
        qa_id=qa_id,
        provider="model",
        model_name="model-name",
        input_type="frame",
        benchmark_type="test",
        modality="rgb",
        section="counting",
        pair_key="pair",
        question="How many cups are visible?",
        ground_truth_answer="Two",
        model_answer=answer,
        status="answered",
        reason="",
        latency_seconds=1.0,
        baseline_gpu_gb=1.0,
        peak_gpu_gb=2.0,
        incremental_peak_gpu_gb=1.0,
        source_metadata={},
    )


class VLMAnswerEvaluationTests(unittest.TestCase):
    def test_deterministic_short_answer_metrics(self):
        self.assertEqual(token_prf("A ceramic teapot", "teapot")["f1"], 2 / 3)
        self.assertEqual(boolean_accuracy("Yes.", "true"), 1.0)
        self.assertEqual(numeric_accuracy("two", "2"), 1.0)
        self.assertGreater(rouge_l_f1("open the door then enter", "open door enter"), 0.7)
        self.assertGreater(anls("mailbox", "mail box"), 0.8)

    def test_metric_router_uses_task_specific_metrics(self):
        metrics = deterministic_metrics("two", "2")
        routed = route_metric("dynamic_counting", "How many?", "two", metrics)
        self.assertEqual(routed["metric"], "numeric_accuracy")
        self.assertEqual(routed["score"], 1.0)

    def test_loader_supports_results_and_items(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key in ("results", "items"):
                path = root / f"{key}.json"
                path.write_text(
                    json.dumps(
                        {
                            key: {
                                "qa-1": {
                                    "qa_id": "qa-1",
                                    "provider": "test",
                                    "model_name": "model",
                                    "question": "Question?",
                                    "ground_truth_answer": "answer",
                                    "model_answer": "answer",
                                    "status": "answered",
                                    "frame_paths": ["frame.jpg"],
                                }
                            },
                            "metadata": {"benchmark_type": "frame_test"},
                        }
                    ),
                    encoding="utf-8",
                )
                loaded = load_result_file(path)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0].input_type, "frame")

    def test_directory_loader_skips_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text('{"items":[]}', encoding="utf-8")
            (root / "result.json").write_text(
                json.dumps(
                    {
                        "items": {
                            "qa": {
                                "qa_id": "qa",
                                "ground_truth_answer": "yes",
                                "model_answer": "yes",
                                "status": "answered",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            records, skipped = load_evaluation_records([root])
            self.assertEqual(len(records), 1)
            self.assertEqual(skipped, [])

    def test_loader_backfills_ground_truth_from_metadata_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa_path = root / "valid.json"
            qa_path.write_text(
                json.dumps(
                    {
                        "valid_qa": [
                            {
                                "qa_id": "qa-1",
                                "question": "How many cups?",
                                "answer": "Two.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "results": {
                            "qa-1": {
                                "qa_id": "qa-1",
                                "question": "How many cups?",
                                "model_answer": "2",
                                "status": "answered",
                            }
                        },
                        "metadata": {"input_path": qa_path.as_posix()},
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_result_file(result_path)
            self.assertEqual(loaded[0].ground_truth_answer, "Two.")
            self.assertEqual(
                loaded[0].source_metadata["ground_truth_source_path"],
                qa_path.as_posix(),
            )

    def test_llm_judge_cache_and_normalization(self):
        class Response:
            text = json.dumps(
                {
                    "items": [
                        {
                            "record_id": "record-1",
                            "label": "correct",
                            "reason": "Equivalent.",
                            "error_type": "none",
                        }
                    ]
                }
            )

        class Models:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **_kwargs):
                self.calls += 1
                return Response()

        class Client:
            def __init__(self):
                self.models = Models()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judge.json"
            client = Client()
            results = run_llm_judge(
                [record()],
                output,
                batch_size=1,
                api_key_list_path=Path(directory) / "missing",
                client_factory=lambda _key: client,
            )
            self.assertEqual(results["record-1"]["score"], 1.0)
            self.assertTrue(results["record-1"]["evaluation_input_sha256"])
            self.assertEqual(results["record-1"]["model_name"], "model-name")
            run_llm_judge(
                [record()],
                output,
                batch_size=1,
                api_key_list_path=Path(directory) / "missing",
                client_factory=lambda _key: client,
            )
            self.assertEqual(client.models.calls, 1)

    def test_llm_judge_rejudges_changed_answer_and_reuses_moved_record(self):
        class Models:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                prompt = kwargs["contents"][0]
                record_id = json.loads(prompt.split("Items:\n", 1)[1])[0]["record_id"]
                return type(
                    "Response",
                    (),
                    {
                        "text": json.dumps(
                            {
                                "items": [
                                    {
                                        "record_id": record_id,
                                        "label": "correct",
                                        "reason": "Equivalent.",
                                        "error_type": "none",
                                    }
                                ]
                            }
                        )
                    },
                )()

        class Client:
            def __init__(self):
                self.models = Models()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judge.json"
            client = Client()
            options = {
                "batch_size": 1,
                "api_key_list_path": Path(directory) / "missing",
                "client_factory": lambda _key: client,
            }
            run_llm_judge([record()], output, **options)
            run_llm_judge([record(record_id="moved-record")], output, **options)
            self.assertEqual(client.models.calls, 1)

            changed = record(record_id="moved-record", answer="three")
            results = run_llm_judge([changed], output, **options)
            self.assertEqual(client.models.calls, 2)
            self.assertEqual(results["moved-record"]["record_id"], "moved-record")

    def test_llm_judge_does_not_trust_legacy_cache_without_input_hash(self):
        class Models:
            calls = 0

            def generate_content(self, **_kwargs):
                self.calls += 1
                return type(
                    "Response",
                    (),
                    {
                        "text": json.dumps(
                            {
                                "items": [
                                    {
                                        "record_id": "record-1",
                                        "label": "incorrect",
                                        "reason": "Re-evaluated.",
                                        "error_type": "wrong_count",
                                    }
                                ]
                            }
                        )
                    },
                )()

        class Client:
            def __init__(self):
                self.models = Models()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judge.json"
            output.write_text(
                json.dumps(
                    {
                        "items": {
                            "record-1": {
                                "record_id": "record-1",
                                "label": "correct",
                                "score": 1.0,
                            }
                        },
                        "metadata": {
                            "judge_model": "gemini-3.1-flash-lite",
                            "judge_prompt_version": "reference_guided_vlm_answer_judge_v1",
                            "judge_prompt_sha256": judge_prompt_sha256(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            client = Client()
            results = run_llm_judge(
                [record()],
                output,
                batch_size=1,
                api_key_list_path=Path(directory) / "missing",
                client_factory=lambda _key: client,
            )
            self.assertEqual(client.models.calls, 1)
            self.assertEqual(results["record-1"]["label"], "incorrect")

    def test_llm_judge_warns_when_response_omits_an_item(self):
        calls = 0

        class Client:
            class Models:
                @staticmethod
                def generate_content(**_kwargs):
                    nonlocal calls
                    calls += 1
                    returned_ids = ["record-1"] if calls == 1 else ["record-2"]
                    return type(
                        "Response",
                        (),
                        {
                            "text": json.dumps(
                                {
                                    "items": [
                                        {
                                            "record_id": record_id,
                                            "label": "correct",
                                            "reason": "Equivalent.",
                                            "error_type": "none",
                                        }
                                        for record_id in returned_ids
                                    ]
                                }
                            )
                        },
                    )()

            models = Models()

        with tempfile.TemporaryDirectory() as directory:
            warning_output = io.StringIO()
            with redirect_stderr(warning_output):
                results = run_llm_judge(
                    [record(), record(record_id="record-2", qa_id="qa-2")],
                    Path(directory) / "judge.json",
                    batch_size=2,
                    api_key_list_path=Path(directory) / "missing",
                    client_factory=lambda _key: Client(),
                )
            self.assertIn(
                "WARNING: Gemini judge response has 1 missing record_id(s): record-2",
                warning_output.getvalue(),
            )
            self.assertEqual(results["record-2"]["label"], "unjudgeable")
            self.assertEqual(results["record-2"]["reason"], "Judge omitted this item.")

            results = run_llm_judge(
                [record(), record(record_id="record-2", qa_id="qa-2")],
                Path(directory) / "judge.json",
                batch_size=2,
                api_key_list_path=Path(directory) / "missing",
                client_factory=lambda _key: Client(),
            )
            self.assertEqual(calls, 2)
            self.assertEqual(results["record-2"]["label"], "correct")

    def test_llm_judge_retries_transient_network_error(self):
        class Models:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Server disconnected without sending a response.")
                return type(
                    "Response",
                    (),
                    {
                        "text": json.dumps(
                            {
                                "items": [
                                    {
                                        "record_id": "record-1",
                                        "label": "correct",
                                        "reason": "Equivalent.",
                                        "error_type": "none",
                                    }
                                ]
                            }
                        )
                    },
                )()

        class Client:
            def __init__(self):
                self.models = Models()

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            delays = []
            results = run_llm_judge(
                [record()],
                Path(directory) / "judge.json",
                batch_size=1,
                max_retries=3,
                retry_delay_seconds=2,
                retry_sleep=delays.append,
                api_key_list_path=Path(directory) / "missing",
                client_factory=lambda _key: client,
            )
            self.assertEqual(client.models.calls, 2)
            self.assertEqual(delays, [2])
            self.assertEqual(results["record-1"]["label"], "correct")

    def test_llm_judge_checkpoints_after_retry_exhaustion(self):
        class Client:
            class Models:
                calls = 0

                def generate_content(self, **_kwargs):
                    self.calls += 1
                    raise RuntimeError("Server disconnected without sending a response.")

            models = Models()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "judge.json"
            client = Client()
            with self.assertRaisesRegex(RuntimeError, "Server disconnected"):
                run_llm_judge(
                    [record()],
                    output,
                    batch_size=1,
                    max_retries=3,
                    retry_delay_seconds=0,
                    api_key_list_path=Path(directory) / "missing",
                    client_factory=lambda _key: client,
                )
            self.assertEqual(client.models.calls, 3)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["stopped_reason"], "judge_error")
            self.assertEqual(payload["metadata"]["evaluated_items"], 0)

    def test_report_outputs_and_pairwise_comparison(self):
        left = record("left", "qa-1")
        right = EvaluationRecord(**{**left.to_dict(), "record_id": "right", "model_name": "other"})
        judgments = {
            "left": {"label": "correct", "score": 1.0},
            "right": {"label": "incorrect", "score": 0.0},
        }
        rows = score_records([left, right], judgments)
        comparison = pairwise_judge_comparisons(rows)
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0]["model_a_only_correct"], 1)
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_evaluation_outputs(
                directory,
                rows,
                bootstrap_samples=20,
            )
            self.assertTrue(outputs["summary_json"].is_file())
            summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(len(summary["models"]), 2)


if __name__ == "__main__":
    unittest.main()
