对，你的直觉是对的：**不同 modality 的答案并不是都相同**。

我刚刚反查了一下：

```text
四个 modality 答案完全相同的问题: 2048
四个 modality 有 2 种不同答案的问题: 1896
四个 modality 有 3 种不同答案的问题: 790
四个 modality 有 4 种不同答案的问题: 597
```

所以并不是“答案都一样”。

真正发生的是：

```text
tie-breaker 后仍并列的 4077 个问题里，
并列 best modality 的 metric tuple 完全一样。
```

也就是对于仍然并列的候选 modality，它们在这些字段上完全一样：

```text
judge_score
task_aware_score
rouge_l_f1
meteor
bleu_4
token_f1
```

但答案文本不一定一样：

```text
4077 个 unresolved ties 中：
3594 个：并列 best 的答案文本相同
378 个：并列 best 里有 2 种不同答案
93 个：并列 best 里有 3 种不同答案
12 个：并列 best 里有 4 种不同答案
```

所以情况是：

> 很多答案不同，但当前 metrics 给了它们完全一样的分数。

举个真实例子：

```text
ground truth: A doorway.

rgb answer:   door
ir answer:    door
event answer: doorframe
depth answer: doorframe
```

这些答案不完全相同，但 judge 都认为语义正确，`judge_score = 1.0`；同时 token/text metrics 可能也给出一样或同样低的分数。于是它们仍然无法被当前 tie-breaker 区分。

这说明不是代码逻辑明显反常，而是 **metrics 的分辨率不够细**，尤其是短答案场景：

```text
door vs doorway vs doorframe
yes vs yes
person vs man
room vs doorway
```

这些在语义上可能都被 judge 认为正确，但词面指标又很粗，所以会大量并列。

结论：目前 tie-breaker 合理，但它只能利用已有 metric 信息。若想继续减少并列，需要引入新的区分信号，例如：

```text
1. answer semantic preference judge，让 LLM 在并列答案之间二选一/多选一
2. 更细粒度人工规则，比如优先更接近 ground truth 的 normalized answer
3. 加 answer-level similarity，比如 embedding similarity
4. 保留 multi-best，把它当作“多个 modality 都擅长该问题”
```

对于你的研究目的，我反而建议保留 `primary_best_modalities` / `best_input_modalities` 的 multi-best 信息，因为“多个 modality 都能答好这个问题”本身也是有意义的发现。


对，这个现象不是“所有不同 modality 都真的提供了同样信息”，而是这些问题本身有明显特点：**短答案、闭合问题、低熵答案很多**。

我统计到：

```text
四个 modality 答案完全相同: 2048 / 5331 = 38.4%
```

这些同答问题的典型答案非常集中：

```text
no    1066
yes   212
0     192
1     124
2     89
```

也就是说，光 `yes/no/0/1/2` 就占了很大部分。尤其是 `no` 特别多。

按任务 metric 看，同答率最高的是：

```text
boolean_accuracy: 1223 / 1954 = 62.6%
numeric_accuracy: 360 / 1043 = 34.5%
token_f1:          291 / 1244 = 23.4%
sequence_order:    111 / 705  = 15.7%
anls/OCR:           33 / 218  = 15.1%
set_f1:             30 / 167  = 18.0%
```

所以四模态同答主要来自 **Boolean 问题** 和 **计数问题**。

最容易同答的 section：

```text
event_non_common:        56 / 56  = 100%
depth_non_common:        54 / 56  = 96.4%
non_common:             105 / 123 = 85.4%
depth_navigation:        37 / 53  = 69.8%
dynamic_counting:       298 / 434 = 68.7%
depth_spatial_reasoning: 33 / 50  = 66.0%
```

这很符合直觉：`non_common` 这类问题经常问：

```text
Are there any floating/impossible objects?
Are there any abnormal objects?
```

答案通常就是 `no`。模型在不同 modality 下都会给出同一个保守答案。

还有一类是计数：

```text
How many people...
How many moving objects...
```

模型经常统一回答 `0`、`1`、`2`。

同答问题里，judge 分布也说明它们不是全都“正确”：

```text
四模态都 judge_score = 1.0: 1579 个问题
四模态都 judge_score = 0.0: 439 个问题
四模态都 judge_score = 0.5: 23 个问题
```

所以有两种情况：

```text
1. 四个 modality 都答对了同一个简单答案
2. 四个 modality 都犯了同一个错误
```

为什么会这样：

```text
1. 问题答案空间很小，比如 yes/no、0/1/2、left/right。
2. 同一个 VLM 面对同一个问题，有强语言先验，视觉输入差异不足以改变输出。
3. 很多高层语义在 RGB/IR/event/depth 中都可见，所以答案自然一致。
4. 对 absence/异常检测问题，模型倾向回答 no。
5. 对计数问题，模型倾向回答常见小数字，尤其 0。
```

所以 2048 个同答不是异常，反而说明这批 QA 里有相当一部分问题不太能区分 modality 擅长性。真正更有分析价值的是另外 3283 个“不同 modality 给出不同答案”的问题，尤其是其中最终 winner 不同的部分。