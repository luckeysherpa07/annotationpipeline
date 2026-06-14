# VLM Answer Evaluation

This evaluator scores saved VLM answers without rerunning model inference. It
supports frame-input and video-input result files that contain the question,
ground-truth answer, model answer, task section, modality, latency, and GPU
memory measurements.

## Metrics

- Normalized exact match
- Token precision, recall, and F1
- ROUGE-L F1 as a secondary lexical diagnostic
- ANLS and character F1 for OCR/text answers
- Boolean accuracy for yes/no tasks
- Numeric accuracy for counting tasks
- Set F1 for multi-concept answers
- Sequence-order score for temporal or ordered answers
- Repetition and excessive-length diagnostics
- Optional blinded LLM Judge: strict accuracy and partial-credit accuracy
- Answer rate, failure counts, latency, throughput, and peak GPU memory
- Micro averages, modality/task macro averages, and 95% bootstrap intervals
- Pairwise disagreement and McNemar tests when multiple models share QA IDs

The task-aware score is selected from the deterministic metrics according to
the QA section and question type. It is a reproducible diagnostic rather than
a replacement for semantic judging.

## Deterministic Evaluation

Evaluate all result JSON files in a benchmark directory:

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic
```

Multiple files or directories may be passed after `--input`. Manifest and
summary files are ignored automatically.

## Evaluation With LLM Judge

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic,llm_judge \
  --judge-model gemini-2.5-flash \
  --judge-batch-size 20 \
  --api-key-list api_keys.txt
```

Judge results are stored in `judge_cache.json`. Re-running the same command
only sends records that are not already cached. Model identity is excluded
from the judge prompt. The judge returns `correct`, `partially_correct`,
`incorrect`, or `unjudgeable`, plus a short reason and error type.

Run a small validation before the full judge:

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b_smoke \
  --metrics deterministic,llm_judge \
  --max-records 20 \
  --judge-max-items 20
```

## Outputs

- `per_item_scores.json` and `per_item_scores.csv`: one row per answer
- `summary.json` and `summary.csv`: overall model results
- `modality_scores.csv`: results by modality
- `section_scores.csv`: results by task section
- `pairwise_comparisons.csv`: paired model comparisons when judge scores exist
- `failures.csv`: failed or unanswered records
- `report.md`: compact comparison table
- `judge_cache.json`: resumable LLM Judge cache, when enabled

---

# VLM 回答评估

该工具直接评估已经保存的模型回答，不会重新执行模型推理。它同时兼容帧输入和视频输入结果。

确定性指标包括精确匹配、Token F1、ROUGE-L、ANLS、字符 F1，以及按任务选择的布尔、计数、集合和顺序指标。可选的 LLM Judge 会在隐藏模型身份的情况下判断回答为正确、部分正确、错误或无法判断。报告还会统计回答率、失败数、延迟、吞吐量、峰值显存、模态与任务宏平均及 95% bootstrap 置信区间。

正式执行确定性评估：

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic
```

增加 LLM Judge：

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic,llm_judge \
  --judge-model gemini-2.5-flash \
  --judge-batch-size 20 \
  --api-key-list api_keys.txt
```

`judge_cache.json` 支持断点续跑；再次执行时不会重复评估已经缓存的回答。
