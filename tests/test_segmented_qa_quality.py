from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from annotation_feature.cli.actions.segmented_qa_quality import (
    build_segmented_qa_quality_actions,
)
from annotation_feature.qa_quality.segmented_cleaner import (
    clean_segmented_qa_dataset,
)
from annotation_feature.qa_quality.segmented_evaluator import evaluate_segmented_qa
from annotation_feature.qa_quality.segmented_llm_evaluator import (
    run_segmented_qa_llm_evaluation,
)


class _FakeModels:
    def generate_content(self, model, contents):
        items = json.loads(contents[0].split("Items:\n", 1)[1])
        response_items = [
            {
                "qa_id": item["qa_id"],
                "status": "pass",
                "answerable_from_caption": True,
                "answer_matches_question": True,
                "caption_supports_answer": True,
                "modality_appropriate": True,
                "single_question": True,
                "segment_consistent": True,
                "hallucination_risk": "low",
                "reason": "Supported by the segment caption.",
            }
            for item in items
        ]
        return SimpleNamespace(text=json.dumps({"items": response_items}))


class SegmentedQaQualityTests(unittest.TestCase):
    def test_rule_evaluation_preserves_segment_metadata_and_unique_pair_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "segmented.json"
            output_dir = root / "outputs"
            input_path.write_text(
                json.dumps(
                    {
                        "segment_001": {
                            "source_prefix": "check_mailbox",
                            "side": "day",
                            "task_label": "open mailbox",
                            "start_seconds": 1.0,
                            "end_seconds": 4.0,
                            "start_timestamp": "00:00:01",
                            "end_timestamp": "00:00:04",
                            "source_files": {"rgb": "dataset/day_rgb.mp4"},
                            "evidence_units": [
                                {
                                    "modality": "rgb",
                                    "section": "appearance",
                                    "caption": "A hand opens the mailbox.",
                                    "question": "What opens the mailbox?",
                                    "answer": "A person's hand opens it.",
                                    "source_unit_index": 0,
                                    "pair_index": 1,
                                },
                                {
                                    "modality": "rgb",
                                    "section": "appearance",
                                    "caption": "The mailbox door moves outward.",
                                    "question": "How does the mailbox door move?",
                                    "answer": "It moves outward.",
                                    "source_unit_index": 0,
                                    "pair_index": 2,
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            outputs = evaluate_segmented_qa(input_path=input_path, output_dir=output_dir)
            cleaned = json.loads(outputs["cleaned_json"].read_text(encoding="utf-8"))

            self.assertEqual(2, len(cleaned["items"]))
            self.assertEqual(2, len({item["qa_id"] for item in cleaned["items"]}))
            self.assertEqual("day", cleaned["items"][0]["side"])
            self.assertEqual("open mailbox", cleaned["items"][0]["task_label"])
            self.assertEqual("dataset/day_rgb.mp4", cleaned["items"][0]["source_media"])
            self.assertEqual(input_path.as_posix(), cleaned["metadata"]["source_file"])

    def test_llm_evaluation_and_strict_cleaning_require_segment_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cleaned.json"
            llm_json = root / "llm.json"
            llm_csv = root / "llm.csv"
            valid_path = root / "valid.json"
            base_item = {
                "qa_id": "segment_001__rgb__appearance__0__1__01",
                "segment_id": "segment_001",
                "source_prefix": "check_mailbox",
                "side": "day",
                "task_label": "open mailbox",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "start_timestamp": "00:00:01",
                "end_timestamp": "00:00:04",
                "source_files": {"rgb": "dataset/day_rgb.mp4"},
                "source_media": "dataset/day_rgb.mp4",
                "modality": "rgb",
                "section": "appearance",
                "caption": "A hand opens the mailbox.",
                "question": "What opens the mailbox?",
                "answer": "A person's hand opens it.",
                "transform": "none",
                "source_severity": "pass",
                "source_flags": [],
            }
            input_path.write_text(
                json.dumps({"items": [base_item], "metadata": {}}),
                encoding="utf-8",
            )

            run_segmented_qa_llm_evaluation(
                input_path=input_path,
                output_json=llm_json,
                output_csv=llm_csv,
                client=SimpleNamespace(models=_FakeModels()),
                delay_between_batches=0,
            )
            evaluated = json.loads(llm_json.read_text(encoding="utf-8"))
            result = next(iter(evaluated["items"].values()))
            self.assertTrue(result["segment_consistent"])
            self.assertEqual("dataset/day_rgb.mp4", result["source_media"])

            result["segment_consistent"] = False
            llm_json.write_text(
                json.dumps({"items": {result["qa_id"]: result}, "metadata": {}}),
                encoding="utf-8",
            )
            cleaned = clean_segmented_qa_dataset(llm_json, valid_path)
            self.assertEqual(0, cleaned["summary"]["total_valid"])

    def test_menu_registers_segmented_quality_actions(self):
        actions = build_segmented_qa_quality_actions(confirm=lambda _: False)
        self.assertEqual("segmented.qa_quality.evaluate", actions["73"].action_id)
        self.assertEqual("segmented.qa_quality.llm_eval", actions["74"].action_id)
        self.assertEqual("segmented.qa_quality.clean", actions["75"].action_id)


if __name__ == "__main__":
    unittest.main()
