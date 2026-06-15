# VLM Answer Evaluation

This evaluator scores saved VLM answers without rerunning model inference. It
supports frame-input and video-input result files that contain the question,
ground-truth answer, model answer, task section, modality, latency, and GPU
memory measurements.

1. BLEU ROUGE Meteor, LLM-as-a-judge
2. observation 4B to 8B, modality, attribute
3. Ego challenge
4. Pr: structure, describe the dataset, QA attributes
5. detail analysis
6. is Molmo2 better in video QA? as it stated in its paper.
7. For challenge, QA quality is important

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

### Metric Reference

All accuracy and similarity scores are in `[0, 1]`, where a larger value is
better unless stated otherwise.

| Output field | Meaning | Recommended use |
|---|---|---|
| `normalized_exact_match` | Exact equality after lowercasing, removing punctuation, normalizing whitespace, and removing English articles | Very strict short-answer diagnostic |
| `token_precision` | Fraction of candidate tokens that overlap the reference | Detects unsupported or excessive candidate content |
| `token_recall` | Fraction of reference tokens covered by the candidate | Detects missing reference content |
| `token_f1` | Harmonic mean of token precision and recall | General lexical similarity for short open-ended answers |
| `rouge_l_f1` | F1 based on the longest common token subsequence | Secondary diagnostic when word order matters |
| `anls` | Character-edit similarity, set to zero below a similarity threshold of `0.5` | OCR and text-reading answers |
| `character_f1` | Character overlap F1 after normalization | OCR spelling and transcription diagnostic |
| `boolean_accuracy` | Whether parsed yes/no polarity matches exactly | Boolean questions |
| `numeric_accuracy` | Whether the first parsed number matches; number words such as `two` match `2` | Counting questions |
| `set_f1` | F1 over concepts split by commas, semicolons, `/`, `and`, or `then` | Multi-concept answers where order is not primary |
| `sequence_order_score` | Fraction of reference concepts preserved in candidate order | Temporal, navigation, and ordered answers |
| `task_aware_score` | One deterministic metric selected according to the QA type | Primary reproducible deterministic diagnostic |
| `repetition_ratio` | Estimated fraction of excessive repeated unigrams and bigrams | Response-quality diagnostic; lower is better |
| `repetition_flag` | `true` when repetition ratio is at least `0.3` | Counts clearly repetitive responses |
| `conciseness_violation` | Candidate is over 100 characters and over four times the reference length | Detects excessively verbose answers |
| `answer_length_chars` | Candidate answer length in characters | Descriptive response-length statistic |

`task_aware_score` routes items as follows:

| QA type | Selected metric |
|---|---|
| Boolean reference | `boolean_accuracy` |
| Counting section or “how many” question | `numeric_accuracy`, with `token_f1` fallback |
| Text/OCR/reading section | `anls` |
| Sequence/navigation/order/temporal section | `sequence_order_score` |
| Multi-concept reference | `set_f1` |
| Other open-ended answer | `token_f1` |

### LLM Judge Metrics

The blinded judge assigns `correct = 1.0`, `partially_correct = 0.5`,
`incorrect = 0.0`, or `unjudgeable = null`.

| Output field | Meaning |
|---|---|
| `judge_evaluated` | Number of answered items with a judge label, including `unjudgeable` |
| `judge_label_counts` | Counts of the four judge labels |
| `judge_strict_accuracy` | Fraction labeled `correct`; `unjudgeable` items are excluded from the denominator |
| `judge_soft_accuracy` | Mean of numeric judge scores; partial answers receive half credit and `unjudgeable` is excluded |
| `judge_unjudgeable_rate` | Fraction of judged items labeled `unjudgeable` |
| `judge_reason` | Short per-item explanation from the judge |
| `judge_error_type` | Error category such as wrong count, contradiction, missing detail, or wrong order |
| `judge_model` | Judge model that produced the individual judgment |

For headline answer quality, use `judge_strict_accuracy` and
`judge_soft_accuracy`. Use `task_aware_score` as a reproducible diagnostic and
the lexical metrics to explain particular failure modes; token or character
similarity alone should not be described as semantic accuracy.

### Aggregate And Efficiency Fields

| Output field | Meaning |
|---|---|
| `answer_rate` | Answered, non-empty items divided by all input items |
| `repetition_rate` | Fraction of answered items with `repetition_flag = true`; lower is better |
| `conciseness_violation_rate` | Fraction of answered items violating the length rule; lower is better |
| `latency_mean_seconds` | Mean per-item inference latency |
| `latency_median_seconds` | Median per-item inference latency |
| `latency_p95_seconds` | 95th-percentile per-item latency |
| `throughput_qa_per_hour` | Answered items divided by summed latency, scaled to one hour |
| `peak_gpu_gb_max` | Maximum recorded total GPU-memory peak |
| `incremental_peak_gpu_gb_max` | Maximum GPU-memory increase above the recorded baseline |
| `*_ci95` | Seeded bootstrap 95% confidence interval for the corresponding mean |
| `modality_macro_*` | Unweighted mean across modalities |
| `section_macro_*` | Unweighted mean across QA sections |
| `label_disagreement_rate` | Fraction of shared QA IDs receiving different judge labels between two models |
| `mcnemar_exact_p_value` | Exact paired test using items where only one of two models is judged correct |

`summary.csv` contains one row per model, `modality_scores.csv` one row per
model and modality, and `section_scores.csv` one row per model and QA section.

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

### 30-Frame 4B Evaluation

The three complete 30-frame 4B result files can be evaluated together with the
same deterministic metrics, LLM Judge, modality summaries, section summaries,
and pairwise comparisons used by the fixed 8-frame experiments:

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input \
    outputs/benchmarks/aligned_qa_frame_answers_allenai_Molmo2-4B.json \
    outputs/benchmarks/aligned_qa_frame_answers_OpenGVLab_InternVL2_5-4B.json \
    outputs/benchmarks/aligned_qa_frame_answers_Qwen_Qwen3-VL-4B-Instruct.json \
  --output outputs/evaluations/vlm_30frame_aligned_4b \
  --require-frame-count 30 \
  --metrics deterministic,llm_judge \
  --judge-model gemini-3.1-flash-lite \
  --judge-batch-size 150
```

`--require-frame-count 30` validates every answered input record before any
judge request is sent. Reports retain both the configured maximum and observed
frame counts, so 8-frame and 30-frame experiments remain distinguishable.

## Evaluation With LLM Judge

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic,llm_judge \
  --judge-model gemini-3.1-flash-lite \
  --judge-batch-size 100 \
  --judge-max-retries 3 \
  --judge-service-unavailable-max-retries 8 \
  --api-key-list api_keys.txt
```

Judge results are stored in `llm_judge_cache.json`. Re-running the same command
only sends records whose evaluation input is not already cached. The cache
fingerprint covers the QA ID, modality, section, question, reference answer,
and candidate answer, so edited answers are evaluated again while moved input
files can reuse matching results. Model identity is excluded from the judge
prompt but retained in the cache for auditing. The judge returns `correct`,
`partially_correct`, `incorrect`, or `unjudgeable`, plus a short reason and
error type. If a batch response omits, duplicates, or returns unexpected record
IDs, the evaluator prints a warning with the affected count and sample IDs.
Omitted items are not treated as completed cache entries and are retried the
next time the evaluator runs.

The cache can be continued with a different judge model. Existing judgments
remain reusable, while every item records the judge model that produced it.
Cache metadata reports the active judge model and per-model item counts. This
supports quota-driven model changes, but aggregate scores from a mixed-judge
cache should be interpreted as using more than one evaluator.
Transient request and response errors are retried three times by default with
an increasing delay. Configure this with `--judge-max-retries` and
`--judge-retry-delay-seconds`. HTTP 503/model-high-demand errors use a separate
policy: eight attempts with a 15-second increasing delay by default. Configure
it with `--judge-service-unavailable-max-retries` and
`--judge-service-unavailable-retry-delay-seconds`.

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
- `llm_judge_cache.json`: resumable, content-validated LLM Judge cache

---

# VLM 回答评估

该工具直接评估已经保存的模型回答，不会重新执行模型推理。它同时兼容帧输入和视频输入结果。

确定性指标包括精确匹配、Token F1、ROUGE-L、ANLS、字符 F1，以及按任务选择的布尔、计数、集合和顺序指标。可选的 LLM Judge 会在隐藏模型身份的情况下判断回答为正确、部分正确、错误或无法判断。报告还会统计回答率、失败数、延迟、吞吐量、峰值显存、模态与任务宏平均及 95% bootstrap 置信区间。

## 指标说明

除非特别说明，准确率与相似度指标都位于 `[0, 1]`，数值越高越好。

| 输出字段 | 含义 | 主要用途 |
|---|---|---|
| `normalized_exact_match` | 小写化、去标点、标准化空格并去除英文冠词后完全一致 | 最严格的短回答检查 |
| `token_precision` | 模型回答中的 token 有多少与标准答案重合 | 检查多余或无依据内容 |
| `token_recall` | 标准答案中的 token 有多少被模型回答覆盖 | 检查遗漏内容 |
| `token_f1` | Token precision 与 recall 的调和平均 | 一般开放式短回答的词汇相似度 |
| `rouge_l_f1` | 基于最长公共 token 子序列的 F1 | 对词序敏感的辅助指标 |
| `anls` | 字符编辑相似度；低于 `0.5` 时记为 0 | OCR 与文字识别 |
| `character_f1` | 标准化后的字符重合 F1 | 拼写和转录质量 |
| `boolean_accuracy` | 解析后的 Yes/No 极性是否一致 | 布尔问题 |
| `numeric_accuracy` | 首个解析数字是否一致，`two` 与 `2` 等价 | 计数问题 |
| `set_f1` | 将答案拆成多个概念后计算集合 F1 | 不强调顺序的多概念回答 |
| `sequence_order_score` | 按正确顺序保留的标准答案概念比例 | 时序、导航和步骤问题 |
| `task_aware_score` | 根据 QA 类型选择的一个确定性指标 | 主要的可复现确定性诊断 |
| `repetition_ratio` | 过度重复 unigram 和 bigram 的估计比例 | 回答质量诊断，越低越好 |
| `repetition_flag` | 重复比例不低于 `0.3` 时为 `true` | 统计明显重复回答 |
| `conciseness_violation` | 回答超过 100 字符，且超过标准答案长度四倍 | 检查回答过长 |
| `answer_length_chars` | 模型回答字符数 | 描述性长度统计 |

`task_aware_score` 的选择规则：

| QA 类型 | 使用指标 |
|---|---|
| 布尔标准答案 | `boolean_accuracy` |
| Counting section 或 “how many” 问题 | `numeric_accuracy`，不可用时回退到 `token_f1` |
| Text/OCR/reading section | `anls` |
| Sequence/navigation/order/temporal section | `sequence_order_score` |
| 包含多个概念的标准答案 | `set_f1` |
| 其他开放式回答 | `token_f1` |

### LLM Judge 指标

Judge 标签映射为：`correct = 1.0`、`partially_correct = 0.5`、
`incorrect = 0.0`、`unjudgeable = null`。

| 输出字段 | 含义 |
|---|---|
| `judge_evaluated` | 有 Judge 标签的回答数量，包括 `unjudgeable` |
| `judge_label_counts` | 四种 Judge 标签的数量 |
| `judge_strict_accuracy` | `correct` 比例；分母排除 `unjudgeable` |
| `judge_soft_accuracy` | Judge 数值分数平均值；部分正确计 0.5，排除 `unjudgeable` |
| `judge_unjudgeable_rate` | Judge 结果中 `unjudgeable` 的比例 |
| `judge_reason` | Judge 对单条结果给出的简短原因 |
| `judge_error_type` | 错误数量、矛盾、缺少细节、顺序错误等分类 |
| `judge_model` | 产生该条评分的 Judge 模型 |

最终回答质量建议主要报告 `judge_strict_accuracy` 和
`judge_soft_accuracy`。`task_aware_score` 适合作为可复现辅助指标；Token 或字符
相似度只能用于解释错误，不应单独称为语义准确率。

### 汇总与效率指标

| 输出字段 | 含义 |
|---|---|
| `answer_rate` | 非空且状态为 answered 的条目占全部输入的比例 |
| `repetition_rate` | 明显重复回答比例，越低越好 |
| `conciseness_violation_rate` | 过长回答比例，越低越好 |
| `latency_mean_seconds` | 单条推理平均延迟 |
| `latency_median_seconds` | 单条推理延迟中位数 |
| `latency_p95_seconds` | 单条推理延迟第 95 百分位 |
| `throughput_qa_per_hour` | 按累计延迟换算的每小时回答数量 |
| `peak_gpu_gb_max` | 最大总显存峰值 |
| `incremental_peak_gpu_gb_max` | 相对基线的最大新增显存 |
| `*_ci95` | 对相应均值进行固定随机种子 bootstrap 得到的 95% 置信区间 |
| `modality_macro_*` | 各模态指标的不加权平均 |
| `section_macro_*` | 各 QA section 指标的不加权平均 |
| `label_disagreement_rate` | 两模型在共享 QA 上 Judge 标签不同的比例 |
| `mcnemar_exact_p_value` | 仅使用“只有一个模型正确”的配对样本进行精确检验 |

`summary.csv` 每个模型一行，`modality_scores.csv` 每个“模型 + 模态”一行，
`section_scores.csv` 每个“模型 + QA section”一行。

正式执行确定性评估：

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic
```

### 4B 模型 30 帧评估

三份完整的 4B 模型 30 帧结果可以复用与固定 8 帧实验完全相同的确定性指标、
LLM Judge、模态汇总、任务汇总和模型配对比较：

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input \
    outputs/benchmarks/aligned_qa_frame_answers_allenai_Molmo2-4B.json \
    outputs/benchmarks/aligned_qa_frame_answers_OpenGVLab_InternVL2_5-4B.json \
    outputs/benchmarks/aligned_qa_frame_answers_Qwen_Qwen3-VL-4B-Instruct.json \
  --output outputs/evaluations/vlm_30frame_aligned_4b \
  --require-frame-count 30 \
  --metrics deterministic,llm_judge \
  --judge-model gemini-3.1-flash-lite \
  --judge-batch-size 150
```

`--require-frame-count 30` 会在发送任何 Judge 请求前验证所有已回答记录。
报告会保留配置的最大帧数和实际帧数，因此 8 帧与 30 帧实验可以明确区分。

增加 LLM Judge：

```bash
.venv/bin/python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic,llm_judge \
  --judge-model gemini-3.1-flash-lite \
  --judge-batch-size 100 \
  --judge-max-retries 3 \
  --judge-service-unavailable-max-retries 8 \
  --api-key-list api_keys.txt
```

`llm_judge_cache.json` 支持断点续跑，并使用问题、标准答案和模型回答等内容的
哈希校验缓存。回答内容修改后会自动重新评估；仅移动输入文件不会导致重复评估。
如果批量响应缺失、重复或返回了意外的记录 ID，终端会立即打印数量和 ID 示例。
缺失项不会被视为已经完成，下次执行相同命令时会自动补评。
缓存允许切换 judge 模型后继续运行。已有评分仍会复用，每条评分会记录实际使用的
judge 模型，元数据同时保存当前模型及各模型的评分数量。混合 judge 的汇总结果应
明确按多个评估器共同产生来解释。
网络断连、超时和临时响应错误默认最多尝试三次，并逐次增加等待时间。
HTTP 503 或模型高负载错误默认最多尝试八次，等待时间从 15 秒起逐次增加。
