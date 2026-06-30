# Per-Question Best Input Modality Report

## Goal

The goal of this analysis is to compare the answer quality of the same QA item under four different input modalities: RGB, IR, Event, and Depth. For each question, we identify the input modality or modalities that achieve the best answer quality.

Importantly, this analysis does not force every question to have a single winner. If multiple modalities achieve the same highest composite score, all of them are retained as the best input modality set.

## Data Setup

- Model: Qwen3-VL-8B-Instruct
- Benchmark: 8-frame cross-modality QA
- Number of source questions: 5331
- Input modalities per question: rgb, ir, event, depth
- Total evaluated answers: 21324

This is a paired comparison setting: the same question is answered using different input modalities, so modality effects can be compared directly at the per-question level.

## Composite Score

For each answer, the composite score is defined as:

```text
composite_score =
0.70 * judge_score
+ 0.10 * rouge_l_f1
+ 0.15 * meteor
+ 0.05 * bleu_4
```

Rationale:

- `judge_score` receives the largest weight because it best captures semantic correctness.
- `rouge_l_f1` and `meteor` preserve lexical-overlap information.
- `bleu_4` receives the smallest weight because it is brittle for short answers and paraphrases.

For each question, `best_input_modalities` contains every input modality tied for the highest composite score. If more than one modality shares the maximum score, `is_tie` is set to `True`.

## Main Results

```text
Total questions: 5331
Unique best modality: 1245
Tied best modalities: 4086
```

Unique-best distribution:

```text
rgb   412
event 298
depth 277
ir    258
```

Simplified best-group distribution:

```text
all_modalities 2395
multi_best     1691
rgb             412
event           298
depth           277
ir              258
```

Here, `all_modalities` means that all four input modalities are tied for the highest score, while `multi_best` means that two or three modalities are tied for the highest score.

## Interpretation

The main finding is that most questions cannot be assigned to a single clearly best input modality.

```text
Unique best: 1245 / 5331 = 23.4%
Multi-best: 4086 / 5331 = 76.6%
```

This suggests that many QA items can be answered equally well from multiple modalities. Therefore, modality specialization should not be interpreted only from global winner counts. It should also be analyzed by question type, source section, and whether the item has a unique winner or multiple tied winners.

Among the unique-best subset, RGB has the largest count:

```text
rgb > event > depth > ir
```

However, this should not be interpreted as RGB being universally best, because most questions are multi-best rather than unique-best.

## Why There Are Many Ties

Many ties are caused by short-answer, closed-form, low-entropy questions such as yes/no questions, small-count answers, left/right directions, and abnormal-object absence checks.

The number of questions where all four modalities produce exactly the same answer is:

```text
2048 / 5331 = 38.4%
```

The most common identical answers are:

```text
no    1066
yes   212
0     192
1     124
2      89
```

By task-aware metric, identical-answer rates are highest for:

```text
boolean_accuracy: 1223 / 1954 = 62.6%
numeric_accuracy: 360 / 1043 = 34.5%
```

Sections with the highest four-modality identical-answer rates include:

```text
event_non_common:        56 / 56  = 100%
depth_non_common:        54 / 56  = 96.4%
non_common:             105 / 123 = 85.4%
depth_navigation:        37 / 53  = 69.8%
dynamic_counting:       298 / 434 = 68.7%
depth_spatial_reasoning: 33 / 50  = 66.0%
```

This indicates that `non_common`, `dynamic_counting`, and boolean-style reasoning questions often do not strongly separate input modalities. Their answer spaces are small, and the model frequently gives the same answer under different visual inputs.

## Tie-Breaker Sensitivity Check

We tested a deterministic tie-breaking procedure:

```text
composite_score
-> judge_score
-> task_aware_score
-> text_metric_mean
-> token_f1
```

The tie-breaker changed only:

```text
9 / 5331 = 0.17%
```

Therefore, the main analysis uses the simpler and more interpretable composite-only definition, retaining multi-best modality sets rather than force-resolving ties.

The tie-breaker version is archived under:

```text
tie_breaker_version/
```

## Visualization Files

TensorFlow Projector exports are available:

```text
projector/              full best_input_modalities, 15 classes
projector_best_group/   simplified best_group, 6 classes
projector_unique_best/  unique-best questions only, 4 classes
```

For presentation, the recommended version is:

```text
projector_best_group/
Label by: question
Color by: best_group
```

This simplifies colors into:

```text
rgb
ir
event
depth
multi_best
all_modalities
```

If the goal is to show only the four unique-best modality classes, use:

```text
projector_unique_best/
Color by: best_input_modalities
```

## Current Conclusion

RGB has the largest number of unique-best questions, but the dominant result is that most questions are multi-best: several input modalities can achieve the same highest score.

Therefore, the most accurate wording is:

> For each question, we identify the best input modality set, rather than forcing a unique best input modality.

Recommended next steps:

1. Analyze unique-best questions by `source_section`.
2. Analyze whether multi-best questions are concentrated in boolean, counting, and `non_common` sections.
3. Separately analyze or filter yes/no and simple counting questions to obtain a sharper view of modality specialization.
4. Repeat the same analysis on other models such as InternVL and Molmo2 to test whether the pattern is model-dependent.

