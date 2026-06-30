# Composite Modality Scoring Method

The composite score is a weighted average of per-question modality scores. All included metrics are higher-is-better and already scaled to the 0-1 range.

| Metric | Field | Weight |
|---|---|---:|
| llm_judge | judge_score | 0.7000 |
| rouge_l | rouge_l_f1 | 0.1000 |
| meteor | meteor | 0.1500 |
| bleu_4 | bleu_4 | 0.0500 |

Rationale: LLM judge receives the largest weight because it best captures semantic correctness; ROUGE-L and METEOR keep lexical overlap visible; BLEU-4 receives the smallest weight because it is brittle for short answers.

## Best Modality Definition

`best_input_modalities` contains every input modality tied for the highest composite score for that question. Ties are retained instead of force-resolved, because a sensitivity check with deterministic tie-breakers changed only 9 of 5331 assignments (0.17%). `is_tie` is true when more than one modality shares the maximum composite score.
