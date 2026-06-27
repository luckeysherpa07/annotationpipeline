import unittest

from scripts.build_cross_modality_comparison import (
    build_rgb_ir_pairs,
    grouped_rgb_ir_pairs,
)


def row(qa_id, modality, label, score, section="event_action"):
    return {
        "provider": "provider",
        "model_name": "model",
        "source_qa_id": qa_id,
        "input_modality": modality,
        "source_modality": "event",
        "source_section": section,
        "judge_label": label,
        "judge_score": score,
    }


class CrossModalityComparisonTests(unittest.TestCase):
    def test_rgb_ir_pairs_are_matched_by_source_qa_and_section_is_normalized(self):
        pairs = build_rgb_ir_pairs(
            [
                row("qa-1", "rgb", "correct", 1.0),
                row("qa-1", "ir", "incorrect", 0.0),
                row("qa-2", "rgb", "incorrect", 0.0, "depth_counting"),
                row("qa-2", "ir", "correct", 1.0, "depth_counting"),
                row("qa-3", "rgb", "correct", 1.0),
            ]
        )

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["semantic_section"], "action")
        self.assertEqual(pairs[1]["semantic_section"], "counting")

        overall = grouped_rgb_ir_pairs(pairs, ("provider", "model_name"))[0]
        self.assertEqual(overall["paired_total"], 2)
        self.assertEqual(overall["paired_judgeable"], 2)
        self.assertEqual(overall["rgb_only_correct"], 1)
        self.assertEqual(overall["ir_only_correct"], 1)
        self.assertEqual(overall["rgb_minus_ir_strict_pp"], 0.0)

    def test_unjudgeable_pair_is_excluded_from_accuracy(self):
        pairs = build_rgb_ir_pairs(
            [
                row("qa-1", "rgb", "correct", 1.0),
                row("qa-1", "ir", "unjudgeable", None),
            ]
        )

        overall = grouped_rgb_ir_pairs(pairs, ("provider", "model_name"))[0]
        self.assertEqual(overall["paired_total"], 1)
        self.assertEqual(overall["paired_judgeable"], 0)
        self.assertEqual(overall["paired_excluded"], 1)
        self.assertIsNone(overall["rgb_strict_accuracy"])


if __name__ == "__main__":
    unittest.main()
