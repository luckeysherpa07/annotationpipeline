# Per-Question Best Input Modality Report

For each answer, the composite score is defined as:

composite_score = 0.70 * judge_score + 0.10 * rouge_l_f1 + 0.15 * meteor + 0.05 * bleu_4

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

## Current Conclusion

RGB has the largest number of unique-best questions, but the dominant result is that most questions are multi-best: several input modalities can achieve the same highest score.