# 每题最佳输入 Modality 分析报告

## 目标

本分析的目标是：对同一个 QA 问题，比较 RGB、IR、Event、Depth 四种输入 modality 下的回答质量，从而找出每个问题对应的最佳输入 modality 集合，并进一步观察不同 modality 更擅长的问题类型。

这里的结论不是“强行给每个问题分配唯一 modality”，而是记录所有达到最高综合分的输入 modality。也就是说，一个问题可以有多个 best input modalities。

## 数据设置

- 模型：Qwen3-VL-8B-Instruct
- Benchmark：8-frame cross-modality QA
- 原始问题数：5331
- 每个问题的输入 modality：rgb、ir、event、depth
- 总评估回答数：21324

该设置是 paired comparison：同一个问题在四种输入 modality 下分别回答，因此可以直接比较不同输入对同一问题的影响。

## 综合分数定义

每个回答的综合分数为：

```text
composite_score =
0.70 * judge_score
+ 0.10 * rouge_l_f1
+ 0.15 * meteor
+ 0.05 * bleu_4
```

权重选择理由：

- `judge_score` 权重最高，因为它最接近语义正确性。
- `rouge_l_f1` 和 `meteor` 用于保留文本相似度信息。
- `bleu_4` 对短答案和同义表达较敏感，因此权重最低。

对于每个问题，`best_input_modalities` 定义为所有 composite score 最高的输入 modality。如果多个 modality 并列最高，则全部保留，并将 `is_tie` 标为 `True`。

## 主结果

```text
总问题数: 5331
唯一 best modality: 1245
并列 best modalities: 4086
```

唯一 best modality 分布：

```text
rgb   412
event 298
depth 277
ir    258
```

按简化 best group 统计：

```text
all_modalities 2395
multi_best     1691
rgb             412
event           298
depth           277
ir              258
```

其中 `all_modalities` 表示四种输入 modality 全部并列最高，`multi_best` 表示有两个或三个 modality 并列最高。

## 结果解释

当前最重要的发现是：大多数问题并不能被明确归因到单一最佳输入 modality。

```text
唯一 best: 1245 / 5331 = 23.4%
multi-best: 4086 / 5331 = 76.6%
```

这说明很多问题可以由多个 modality 达到相同的最高回答质量。也就是说，modality specialization 不应只看全局 winner count，而应该进一步按问题类型、source section、是否存在并列等维度分析。

在唯一 best 的问题中，RGB 数量最多，但差距不大：

```text
rgb > event > depth > ir
```

这说明 RGB 在唯一胜出的子集中略占优势，但不能简单得出“RGB 全面最好”的结论，因为大多数问题是 multi-best。

## 为什么并列很多

大量并列的主要原因是：数据中有很多短答案、闭合问题、低熵答案问题。例如 yes/no、0/1/2、left/right、是否存在异常物体等。

四种 modality 输出完全相同答案的问题有：

```text
2048 / 5331 = 38.4%
```

这些同答问题中最常见的答案是：

```text
no    1066
yes   212
0     192
1     124
2      89
```

按 task-aware metric 看，同答率最高的是：

```text
boolean_accuracy: 1223 / 1954 = 62.6%
numeric_accuracy: 360 / 1043 = 34.5%
```

最容易出现四模态同答的 section 包括：

```text
event_non_common:        56 / 56  = 100%
depth_non_common:        54 / 56  = 96.4%
non_common:             105 / 123 = 85.4%
depth_navigation:        37 / 53  = 69.8%
dynamic_counting:       298 / 434 = 68.7%
depth_spatial_reasoning: 33 / 50  = 66.0%
```

这说明 `non_common`、`dynamic_counting`、boolean reasoning 等问题经常无法有效区分 modality，因为答案空间本身很小，模型也容易在不同输入下给出相同回答。

## Tie-breaker 敏感性检查

我们曾测试过一个 deterministic tie-breaker：

```text
composite_score
-> judge_score
-> task_aware_score
-> text_metric_mean
-> token_f1
```

结果显示，tie-breaker 只改变：

```text
9 / 5331 = 0.17%
```

因此，主结果采用更简单、更可解释的 composite-only 定义，并保留 multi-best，而不是强行消除并列。

带 tie-breaker 的版本已归档在：

```text
tie_breaker_version/
```

## 可视化文件

已生成 TensorFlow Projector 可视化文件：

```text
projector/              完整 best_input_modalities，15 类
projector_best_group/   简化 best_group，6 类
projector_unique_best/  只保留唯一 best 问题，4 类
```

推荐汇报时优先使用：

```text
projector_best_group/
Label by: question
Color by: best_group
```

该版本将颜色简化为：

```text
rgb
ir
event
depth
multi_best
all_modalities
```

如果只想展示四种唯一 best modality 的分布，可以使用：

```text
projector_unique_best/
Color by: best_input_modalities
```

## 当前结论

当前结果表明，RGB 在唯一 best 的问题中数量最多，但整体上更重要的发现是：大部分问题存在 multi-best，多个输入 modality 可以达到相同最高分。

因此，目前更准确的表述是：

> 我们为每个问题找到的是 best input modality set，而不是强制唯一 best modality。

这对后续分析很重要。下一步应该重点关注：

1. 唯一 best 问题在不同 `source_section` 中的分布。
2. multi-best 问题是否集中在 boolean、counting、non_common 等低熵问题类型。
3. 去除或单独分析 yes/no 和简单 counting 问题后，modality specialization 是否更明显。
4. 在 InternVL、Molmo2 等其他模型上重复该分析，检查结论是否稳定。

