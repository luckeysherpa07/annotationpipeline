import unittest

from scripts.build_metric_modality_tables import (
    build_composite_table,
    build_metric_table,
    dense_ranks,
)


def row(qa_id, modality, score, question="What happened?"):
    return {
        "source_qa_id": qa_id,
        "input_modality": modality,
        "judge_score": str(score),
        "task_aware_score": str(score),
        "token_f1": str(score),
        "rouge_l_f1": str(score),
        "meteor": str(score),
        "bleu_4": str(score),
        "question": question,
        "source_modality": "rgb",
        "source_section": "action",
        "ground_truth_answer": "answer",
    }


class MetricModalityTableTests(unittest.TestCase):
    def test_dense_ranks_use_same_rank_for_ties(self):
        ranks = dense_ranks({"rgb": 0.9, "ir": 0.5, "event": 0.9, "depth": 0.1})

        self.assertEqual(ranks, {"rgb": 1, "ir": 2, "event": 1, "depth": 3})

    def test_metric_table_pivots_scores_and_ranks_by_input_modality(self):
        table = build_metric_table(
            [
                row("qa-1", "rgb", 0.9),
                row("qa-1", "ir", 0.5),
                row("qa-1", "event", 0.9),
                row("qa-1", "depth", 0.1),
            ],
            "judge_score",
        )

        self.assertEqual(len(table), 1)
        item = table[0]
        self.assertEqual(item["source_qa_id"], "qa-1")
        self.assertEqual(item["rgb_score"], 0.9)
        self.assertEqual(item["event_score"], 0.9)
        self.assertEqual(item["rgb_rank"], 1)
        self.assertEqual(item["ir_rank"], 2)
        self.assertEqual(item["event_rank"], 1)
        self.assertEqual(item["depth_rank"], 3)

    def test_composite_table_uses_weighted_metric_scores(self):
        rows = [
            row("qa-1", "rgb", 0.0),
            row("qa-1", "ir", 0.0),
        ]
        rows[0]["judge_score"] = "1.0"
        rows[0]["rouge_l_f1"] = "0.0"
        rows[1]["judge_score"] = "0.0"
        rows[1]["rouge_l_f1"] = "1.0"

        table = build_composite_table(
            rows,
            {"llm_judge": 0.75, "rouge_l": 0.25},
            modalities=("rgb", "ir"),
        )

        self.assertEqual(table[0]["rgb_composite_score"], 0.75)
        self.assertEqual(table[0]["ir_composite_score"], 0.25)
        self.assertEqual(table[0]["best_input_modalities"], "rgb")
        self.assertFalse(table[0]["is_tie"])

    def test_composite_table_retains_tied_best_modalities(self):
        rows = [
            row("qa-1", "rgb", 1.0),
            row("qa-1", "ir", 1.0),
        ]

        table = build_composite_table(
            rows,
            {"llm_judge": 1.0},
            modalities=("rgb", "ir"),
        )

        self.assertEqual(table[0]["best_input_modalities"], "rgb;ir")
        self.assertTrue(table[0]["is_tie"])


if __name__ == "__main__":
    unittest.main()
