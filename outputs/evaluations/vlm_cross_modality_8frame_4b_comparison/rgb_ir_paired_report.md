# RGB vs IR Paired LLM-Judge Comparison

Each pair contains the RGB and IR answers for the same model and `source_qa_id`. Pairs with an `unjudgeable` or missing label are excluded from strict and soft accuracy.

## Overall

| Model | N | RGB strict | IR strict | RGB - IR (pp) | RGB-only correct | IR-only correct |
|---|---:|---:|---:|---:|---:|---:|
| InternVL3_5-4B-Instruct | 5331 | 0.4590 | 0.4671 | -0.8066 | 306 | 349 |
| Molmo2-4B | 5327 | 0.5001 | 0.4830 | 1.7083 | 461 | 370 |
| Qwen3-VL-4B-Instruct | 5328 | 0.5826 | 0.5685 | 1.4077 | 418 | 343 |

## By QA Section

`semantic_section` removes the `event_` and `depth_` prefixes, so sections with the same QA purpose are combined across their source modalities.

| Model | QA section | N | RGB strict | IR strict | RGB - IR (pp) | RGB-only correct | IR-only correct |
|---|---|---:|---:|---:|---:|---:|---:|
| InternVL3_5-4B-Instruct | action | 719 | 0.4701 | 0.4812 | -1.1127 | 41 | 49 |
| InternVL3_5-4B-Instruct | counting | 674 | 0.5801 | 0.5964 | -1.6320 | 49 | 60 |
| InternVL3_5-4B-Instruct | dynamic_counting | 677 | 0.4106 | 0.4668 | -5.6130 | 22 | 60 |
| InternVL3_5-4B-Instruct | dynamic_recognition | 563 | 0.4014 | 0.4281 | -2.6643 | 19 | 34 |
| InternVL3_5-4B-Instruct | light_change | 448 | 0.4196 | 0.3772 | 4.2411 | 38 | 19 |
| InternVL3_5-4B-Instruct | light_recognition | 104 | 0.4231 | 0.4038 | 1.9231 | 12 | 10 |
| InternVL3_5-4B-Instruct | navigation | 532 | 0.3534 | 0.3515 | 0.1880 | 19 | 18 |
| InternVL3_5-4B-Instruct | non_common | 235 | 0.8553 | 0.8851 | -2.9787 | 5 | 12 |
| InternVL3_5-4B-Instruct | object_recognition | 201 | 0.4428 | 0.3980 | 4.4776 | 23 | 14 |
| InternVL3_5-4B-Instruct | scene_sequence | 574 | 0.3641 | 0.3659 | -0.1742 | 35 | 36 |
| InternVL3_5-4B-Instruct | spatial_reasoning | 203 | 0.6108 | 0.5911 | 1.9704 | 16 | 12 |
| InternVL3_5-4B-Instruct | text_recognition | 401 | 0.4264 | 0.4214 | 0.4988 | 27 | 25 |
| Molmo2-4B | action | 719 | 0.5800 | 0.5716 | 0.8345 | 64 | 58 |
| Molmo2-4B | counting | 674 | 0.3353 | 0.2967 | 3.8576 | 78 | 52 |
| Molmo2-4B | dynamic_counting | 677 | 0.5908 | 0.5968 | -0.5908 | 38 | 42 |
| Molmo2-4B | dynamic_recognition | 562 | 0.5356 | 0.5409 | -0.5338 | 30 | 33 |
| Molmo2-4B | light_change | 448 | 0.5000 | 0.4710 | 2.9018 | 33 | 20 |
| Molmo2-4B | light_recognition | 104 | 0.4423 | 0.4135 | 2.8846 | 17 | 14 |
| Molmo2-4B | navigation | 532 | 0.5038 | 0.4737 | 3.0075 | 44 | 28 |
| Molmo2-4B | non_common | 235 | 0.5447 | 0.5319 | 1.2766 | 20 | 17 |
| Molmo2-4B | object_recognition | 200 | 0.5400 | 0.4600 | 8.0000 | 32 | 16 |
| Molmo2-4B | scene_sequence | 572 | 0.3881 | 0.3899 | -0.1748 | 48 | 49 |
| Molmo2-4B | spatial_reasoning | 203 | 0.6305 | 0.6158 | 1.4778 | 15 | 12 |
| Molmo2-4B | text_recognition | 401 | 0.4888 | 0.4564 | 3.2419 | 42 | 29 |
| Qwen3-VL-4B-Instruct | action | 719 | 0.6147 | 0.5883 | 2.6426 | 66 | 47 |
| Qwen3-VL-4B-Instruct | counting | 674 | 0.5430 | 0.5490 | -0.5935 | 49 | 53 |
| Qwen3-VL-4B-Instruct | dynamic_counting | 677 | 0.7179 | 0.6883 | 2.9542 | 38 | 18 |
| Qwen3-VL-4B-Instruct | dynamic_recognition | 563 | 0.5204 | 0.5275 | -0.7105 | 30 | 34 |
| Qwen3-VL-4B-Instruct | light_change | 448 | 0.5625 | 0.5379 | 2.4554 | 36 | 25 |
| Qwen3-VL-4B-Instruct | light_recognition | 103 | 0.5922 | 0.4660 | 12.6214 | 23 | 10 |
| Qwen3-VL-4B-Instruct | navigation | 531 | 0.5386 | 0.5499 | -1.1299 | 36 | 42 |
| Qwen3-VL-4B-Instruct | non_common | 235 | 0.9277 | 0.9234 | 0.4255 | 2 | 1 |
| Qwen3-VL-4B-Instruct | object_recognition | 200 | 0.5500 | 0.4950 | 5.5000 | 25 | 14 |
| Qwen3-VL-4B-Instruct | scene_sequence | 574 | 0.4599 | 0.4599 | 0.0000 | 56 | 56 |
| Qwen3-VL-4B-Instruct | spatial_reasoning | 203 | 0.5714 | 0.5468 | 2.4631 | 19 | 14 |
| Qwen3-VL-4B-Instruct | text_recognition | 401 | 0.5237 | 0.5012 | 2.2444 | 38 | 29 |
