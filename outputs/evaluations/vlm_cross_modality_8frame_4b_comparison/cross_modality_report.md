# Cross-Modality VLM Comparison

Same questions are evaluated with different input modalities. `source_modality` is where the QA was generated; `input_modality` is the frames shown to the model.

## By Input Modality

| Provider | Model | Input modality | N | Answered | Judge strict | Judge soft | Task-aware |
|---|---|---|---:|---:|---:|---:|---:|
| internvl | InternVL3_5-4B-Instruct | depth | 5331 | 5331 | 0.4091 | 0.4257 | 0.3440 |
| internvl | InternVL3_5-4B-Instruct | event | 5331 | 5331 | 0.4095 | 0.4286 | 0.3448 |
| internvl | InternVL3_5-4B-Instruct | ir | 5331 | 5331 | 0.4671 | 0.4882 | 0.3819 |
| internvl | InternVL3_5-4B-Instruct | rgb | 5331 | 5331 | 0.4590 | 0.4849 | 0.3752 |
| molmo2 | Molmo2-4B | depth | 5331 | 5331 | 0.3137 | 0.3200 | 0.2703 |
| molmo2 | Molmo2-4B | event | 5331 | 5331 | 0.4161 | 0.4277 | 0.3385 |
| molmo2 | Molmo2-4B | ir | 5331 | 5331 | 0.4828 | 0.4959 | 0.3716 |
| molmo2 | Molmo2-4B | rgb | 5331 | 5331 | 0.5001 | 0.5176 | 0.3865 |
| qwen_vl | Qwen3-VL-4B-Instruct | depth | 5331 | 5331 | 0.3999 | 0.4129 | 0.3329 |
| qwen_vl | Qwen3-VL-4B-Instruct | event | 5331 | 5331 | 0.4908 | 0.5110 | 0.4057 |
| qwen_vl | Qwen3-VL-4B-Instruct | ir | 5331 | 5331 | 0.5684 | 0.5895 | 0.4554 |
| qwen_vl | Qwen3-VL-4B-Instruct | rgb | 5331 | 5331 | 0.5825 | 0.6059 | 0.4667 |

## Source By Input Matrix

| Provider | Model | Source modality | Input modality | N | Judge strict | Judge soft | Task-aware |
|---|---|---|---|---:|---:|---:|---:|
| internvl | InternVL3_5-4B-Instruct | depth | depth | 764 | 0.4188 | 0.4300 | 0.3722 |
| internvl | InternVL3_5-4B-Instruct | depth | event | 764 | 0.4372 | 0.4496 | 0.3988 |
| internvl | InternVL3_5-4B-Instruct | depth | ir | 764 | 0.5170 | 0.5308 | 0.4589 |
| internvl | InternVL3_5-4B-Instruct | depth | rgb | 764 | 0.4817 | 0.4987 | 0.4304 |
| internvl | InternVL3_5-4B-Instruct | event | depth | 800 | 0.4600 | 0.4756 | 0.4068 |
| internvl | InternVL3_5-4B-Instruct | event | event | 800 | 0.4600 | 0.4794 | 0.4093 |
| internvl | InternVL3_5-4B-Instruct | event | ir | 800 | 0.5188 | 0.5394 | 0.4543 |
| internvl | InternVL3_5-4B-Instruct | event | rgb | 800 | 0.5050 | 0.5325 | 0.4361 |
| internvl | InternVL3_5-4B-Instruct | ir | depth | 1500 | 0.3987 | 0.4160 | 0.3289 |
| internvl | InternVL3_5-4B-Instruct | ir | event | 1500 | 0.3667 | 0.3890 | 0.3035 |
| internvl | InternVL3_5-4B-Instruct | ir | ir | 1500 | 0.4333 | 0.4560 | 0.3462 |
| internvl | InternVL3_5-4B-Instruct | ir | rgb | 1500 | 0.4300 | 0.4610 | 0.3401 |
| internvl | InternVL3_5-4B-Instruct | rgb | depth | 2267 | 0.3948 | 0.4131 | 0.3223 |
| internvl | InternVL3_5-4B-Instruct | rgb | event | 2267 | 0.4107 | 0.4299 | 0.3311 |
| internvl | InternVL3_5-4B-Instruct | rgb | ir | 2267 | 0.4543 | 0.4771 | 0.3539 |
| internvl | InternVL3_5-4B-Instruct | rgb | rgb | 2267 | 0.4543 | 0.4793 | 0.3582 |
| molmo2 | Molmo2-4B | depth | depth | 764 | 0.3573 | 0.3599 | 0.3226 |
| molmo2 | Molmo2-4B | depth | event | 764 | 0.4542 | 0.4647 | 0.4068 |
| molmo2 | Molmo2-4B | depth | ir | 764 | 0.5393 | 0.5465 | 0.4580 |
| molmo2 | Molmo2-4B | depth | rgb | 764 | 0.5426 | 0.5531 | 0.4668 |
| molmo2 | Molmo2-4B | event | depth | 800 | 0.3125 | 0.3231 | 0.2862 |
| molmo2 | Molmo2-4B | event | event | 800 | 0.4275 | 0.4369 | 0.3699 |
| molmo2 | Molmo2-4B | event | ir | 800 | 0.4838 | 0.4969 | 0.3973 |
| molmo2 | Molmo2-4B | event | rgb | 800 | 0.5075 | 0.5244 | 0.4128 |
| molmo2 | Molmo2-4B | ir | depth | 1500 | 0.2993 | 0.3050 | 0.2473 |
| molmo2 | Molmo2-4B | ir | event | 1500 | 0.4200 | 0.4320 | 0.3206 |
| molmo2 | Molmo2-4B | ir | ir | 1500 | 0.4907 | 0.5053 | 0.3598 |
| molmo2 | Molmo2-4B | ir | rgb | 1500 | 0.4847 | 0.5070 | 0.3584 |
| molmo2 | Molmo2-4B | rgb | depth | 2267 | 0.3089 | 0.3153 | 0.2622 |
| molmo2 | Molmo2-4B | rgb | event | 2267 | 0.3967 | 0.4091 | 0.3162 |
| molmo2 | Molmo2-4B | rgb | ir | 2267 | 0.4583 | 0.4722 | 0.3413 |
| molmo2 | Molmo2-4B | rgb | rgb | 2267 | 0.4934 | 0.5104 | 0.3687 |
| qwen_vl | Qwen3-VL-4B-Instruct | depth | depth | 764 | 0.4359 | 0.4490 | 0.3983 |
| qwen_vl | Qwen3-VL-4B-Instruct | depth | event | 764 | 0.5445 | 0.5635 | 0.5017 |
| qwen_vl | Qwen3-VL-4B-Instruct | depth | ir | 764 | 0.5890 | 0.6054 | 0.5312 |
| qwen_vl | Qwen3-VL-4B-Instruct | depth | rgb | 764 | 0.6073 | 0.6211 | 0.5429 |
| qwen_vl | Qwen3-VL-4B-Instruct | event | depth | 800 | 0.3488 | 0.3638 | 0.3220 |
| qwen_vl | Qwen3-VL-4B-Instruct | event | event | 800 | 0.5075 | 0.5319 | 0.4615 |
| qwen_vl | Qwen3-VL-4B-Instruct | event | ir | 800 | 0.5687 | 0.5906 | 0.4984 |
| qwen_vl | Qwen3-VL-4B-Instruct | event | rgb | 800 | 0.5713 | 0.5975 | 0.5063 |
| qwen_vl | Qwen3-VL-4B-Instruct | ir | depth | 1500 | 0.4200 | 0.4357 | 0.3417 |
| qwen_vl | Qwen3-VL-4B-Instruct | ir | event | 1500 | 0.5043 | 0.5257 | 0.3975 |
| qwen_vl | Qwen3-VL-4B-Instruct | ir | ir | 1500 | 0.5964 | 0.6197 | 0.4620 |
| qwen_vl | Qwen3-VL-4B-Instruct | ir | rgb | 1500 | 0.5813 | 0.6103 | 0.4503 |
| qwen_vl | Qwen3-VL-4B-Instruct | rgb | depth | 2267 | 0.3926 | 0.4030 | 0.3088 |
| qwen_vl | Qwen3-VL-4B-Instruct | rgb | event | 2267 | 0.4579 | 0.4762 | 0.3591 |
| qwen_vl | Qwen3-VL-4B-Instruct | rgb | ir | 2267 | 0.5428 | 0.5638 | 0.4103 |
| qwen_vl | Qwen3-VL-4B-Instruct | rgb | rgb | 2267 | 0.5788 | 0.6009 | 0.4379 |
