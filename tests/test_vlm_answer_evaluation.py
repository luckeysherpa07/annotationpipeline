import json
import tempfile
import unittest
from pathlib import Path

from annotation_feature.qa_quality.answer_judge import run_llm_judge
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
            run_llm_judge(
                [record()],
                output,
                batch_size=1,
                api_key_list_path=Path(directory) / "missing",
                client_factory=lambda _key: client,
            )
            self.assertEqual(client.models.calls, 1)

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
