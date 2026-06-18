# VLM Model and Size Comparison

## Same Input And Size, Different Models

| Input | Size | Model | Judge strict | Judge soft | BLEU-4 | ROUGE-L | METEOR |
|---|---:|---|---:|---:|---:|---:|---:|
| frame 8 | 4B | InternVL3_5-4B-Instruct | 0.4611 | 0.4813 | 0.4704 | 0.2868 | 0.1435 |
| frame 8 | 4B | Molmo2-4B | 0.4516 | 0.4694 | 0.4422 | 0.2534 | 0.1382 |
| frame 8 | 4B | Qwen3-VL-4B-Instruct | 0.5563 | 0.5781 | 0.4730 | 0.3035 | 0.1579 |
| frame 8 | 8B | InternVL3_5-8B-Instruct | 0.4942 | 0.5202 | 0.4711 | 0.2726 | 0.1394 |
| frame 8 | 8B | Molmo2-8B | 0.4878 | 0.5058 | 0.4587 | 0.2651 | 0.1430 |
| frame 8 | 8B | Qwen3-VL-8B-Instruct | 0.5749 | 0.5948 | 0.4759 | 0.3113 | 0.1610 |
| video | 4B | InternVL3_5-4B-Instruct | 0.4898 | 0.5114 | 0.4685 | 0.3089 | 0.1535 |
| video | 4B | Molmo2-4B | 0.4966 | 0.5141 | 0.4674 | 0.2685 | 0.1411 |
| video | 4B | Qwen3-VL-4B-Instruct | 0.5360 | 0.5548 | 0.4708 | 0.2921 | 0.1503 |
| video | 8B | InternVL3_5-8B-Instruct | 0.4897 | 0.5124 | 0.4623 | 0.2637 | 0.1323 |
| video | 8B | Molmo2-8B | 0.4727 | 0.4917 | 0.4505 | 0.2534 | 0.1313 |
| video | 8B | Qwen3-VL-8B-Instruct | 0.5420 | 0.5653 | 0.4650 | 0.2975 | 0.1509 |

## Same Input And Model, Different Sizes

Deltas are calculated as `8B - 4B`.

| Input | Model family | 4B strict | 8B strict | Strict delta | 4B soft | 8B soft | Soft delta |
|---|---|---:|---:|---:|---:|---:|---:|
| frame 8 | InternVL3.5 | 0.4611 | 0.4942 | 0.0332 | 0.4813 | 0.5202 | 0.0389 |
| frame 8 | Molmo2 | 0.4516 | 0.4878 | 0.0362 | 0.4694 | 0.5058 | 0.0364 |
| frame 8 | Qwen3-VL | 0.5563 | 0.5749 | 0.0187 | 0.5781 | 0.5948 | 0.0167 |
| video | InternVL3.5 | 0.4898 | 0.4897 | -0.0002 | 0.5114 | 0.5124 | 0.0009 |
| video | Molmo2 | 0.4966 | 0.4727 | -0.0239 | 0.5141 | 0.4917 | -0.0224 |
| video | Qwen3-VL | 0.5360 | 0.5420 | 0.0060 | 0.5548 | 0.5653 | 0.0105 |
