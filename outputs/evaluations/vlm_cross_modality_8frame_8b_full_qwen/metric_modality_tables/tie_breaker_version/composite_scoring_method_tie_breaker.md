# Composite Modality Scoring Method

- for composite_modality_scores.csv

The composite score is a weighted average of per-question modality scores. All included metrics are higher-is-better and already scaled to the 0-1 range.

| Metric | Field | Weight |
|---|---|---:|
| llm_judge | judge_score | 0.7000 |
| rouge_l | rouge_l_f1 | 0.1000 |
| meteor | meteor | 0.1500 |
| bleu_4 | bleu_4 | 0.0500 |

Default rationale: LLM judge receives the largest weight because it best captures semantic correctness; ROUGE-L and METEOR keep lexical overlap visible; BLEU-4 receives the smallest weight because it is brittle for short answers.

## Tie Breakers

`primary_best_modalities` stores all modalities tied by the raw composite score. `best_input_modalities` applies the following tie-breakers and keeps multiple modalities only if the tie remains unresolved.

| Order | Tie-breaker field | Meaning |
|---:|---|---|
| 1 | composite_score | Composite weighted score |
| 2 | judge_score | LLM judge score |
| 3 | task_aware_score | Task-aware score |
| 4 | text_metric_mean | Mean of ROUGE-L, METEOR, and BLEU-4 |
| 5 | token_f1 | Token F1 |
