# Detailed VLM Comparison

The tables use blinded LLM-judge accuracy as the headline semantic metric. `N` is shown because modality-section groups have different sample counts.

## Modality Comparison

| Input | Size | Model | Modality | N | Judge strict | Judge soft | Task-aware |
|---|---:|---|---|---:|---:|---:|---:|
| frame 8 | 4B | InternVL3_5-4B-Instruct | depth | 764 | 0.4411 | 0.4529 | 0.3905 |
| frame 8 | 4B | InternVL3_5-4B-Instruct | event | 822 | 0.4331 | 0.4556 | 0.3831 |
| frame 8 | 4B | InternVL3_5-4B-Instruct | ir | 1561 | 0.4785 | 0.4974 | 0.3759 |
| frame 8 | 4B | InternVL3_5-4B-Instruct | rgb | 2318 | 0.4658 | 0.4890 | 0.3583 |
| frame 8 | 4B | Molmo2-4B | depth | 764 | 0.3521 | 0.3586 | 0.3226 |
| frame 8 | 4B | Molmo2-4B | event | 822 | 0.4100 | 0.4276 | 0.3667 |
| frame 8 | 4B | Molmo2-4B | ir | 1561 | 0.4824 | 0.5019 | 0.3637 |
| frame 8 | 4B | Molmo2-4B | rgb | 2318 | 0.4784 | 0.4987 | 0.3695 |
| frame 8 | 4B | Qwen3-VL-4B-Instruct | depth | 764 | 0.4319 | 0.4483 | 0.3955 |
| frame 8 | 4B | Qwen3-VL-4B-Instruct | event | 822 | 0.5073 | 0.5298 | 0.4586 |
| frame 8 | 4B | Qwen3-VL-4B-Instruct | ir | 1561 | 0.6060 | 0.6262 | 0.4652 |
| frame 8 | 4B | Qwen3-VL-4B-Instruct | rgb | 2318 | 0.5811 | 0.6057 | 0.4404 |
| frame 8 | 8B | InternVL3_5-8B-Instruct | depth | 764 | 0.4935 | 0.5105 | 0.4500 |
| frame 8 | 8B | InternVL3_5-8B-Instruct | event | 822 | 0.4696 | 0.4982 | 0.4267 |
| frame 8 | 8B | InternVL3_5-8B-Instruct | ir | 1561 | 0.5170 | 0.5423 | 0.4148 |
| frame 8 | 8B | InternVL3_5-8B-Instruct | rgb | 2318 | 0.4879 | 0.5164 | 0.3886 |
| frame 8 | 8B | Molmo2-8B | depth | 764 | 0.4149 | 0.4221 | 0.3705 |
| frame 8 | 8B | Molmo2-8B | event | 822 | 0.4367 | 0.4501 | 0.3817 |
| frame 8 | 8B | Molmo2-8B | ir | 1561 | 0.5080 | 0.5295 | 0.3760 |
| frame 8 | 8B | Molmo2-8B | rgb | 2318 | 0.5164 | 0.5371 | 0.3834 |
| frame 8 | 8B | Qwen3-VL-8B-Instruct | depth | 764 | 0.5013 | 0.5085 | 0.4520 |
| frame 8 | 8B | Qwen3-VL-8B-Instruct | event | 822 | 0.5560 | 0.5773 | 0.4971 |
| frame 8 | 8B | Qwen3-VL-8B-Instruct | ir | 1561 | 0.6111 | 0.6329 | 0.4605 |
| frame 8 | 8B | Qwen3-VL-8B-Instruct | rgb | 2318 | 0.5815 | 0.6038 | 0.4454 |
| video | 4B | InternVL3_5-4B-Instruct | depth | 764 | 0.4450 | 0.4568 | 0.4037 |
| video | 4B | InternVL3_5-4B-Instruct | event | 822 | 0.4209 | 0.4416 | 0.3908 |
| video | 4B | InternVL3_5-4B-Instruct | ir | 1561 | 0.5388 | 0.5573 | 0.4158 |
| video | 4B | InternVL3_5-4B-Instruct | rgb | 2318 | 0.4961 | 0.5233 | 0.3828 |
| video | 4B | Molmo2-4B | depth | 764 | 0.4529 | 0.4640 | 0.4192 |
| video | 4B | Molmo2-4B | event | 822 | 0.4939 | 0.5170 | 0.4225 |
| video | 4B | Molmo2-4B | ir | 1561 | 0.5048 | 0.5247 | 0.4039 |
| video | 4B | Molmo2-4B | rgb | 2318 | 0.5065 | 0.5224 | 0.3944 |
| video | 4B | Qwen3-VL-4B-Instruct | depth | 764 | 0.4280 | 0.4313 | 0.3902 |
| video | 4B | Qwen3-VL-4B-Instruct | event | 822 | 0.5255 | 0.5462 | 0.4694 |
| video | 4B | Qwen3-VL-4B-Instruct | ir | 1561 | 0.5637 | 0.5846 | 0.4377 |
| video | 4B | Qwen3-VL-4B-Instruct | rgb | 2318 | 0.5565 | 0.5785 | 0.4311 |
| video | 8B | InternVL3_5-8B-Instruct | depth | 764 | 0.4338 | 0.4443 | 0.4021 |
| video | 8B | InternVL3_5-8B-Instruct | event | 822 | 0.4732 | 0.4982 | 0.4190 |
| video | 8B | InternVL3_5-8B-Instruct | ir | 1561 | 0.5215 | 0.5416 | 0.4066 |
| video | 8B | InternVL3_5-8B-Instruct | rgb | 2318 | 0.4924 | 0.5201 | 0.3853 |
| video | 8B | Molmo2-8B | depth | 764 | 0.4385 | 0.4457 | 0.4122 |
| video | 8B | Molmo2-8B | event | 822 | 0.4562 | 0.4751 | 0.4055 |
| video | 8B | Molmo2-8B | ir | 1561 | 0.4865 | 0.5042 | 0.3674 |
| video | 8B | Molmo2-8B | rgb | 2318 | 0.4806 | 0.5043 | 0.3690 |
| video | 8B | Qwen3-VL-8B-Instruct | depth | 764 | 0.4450 | 0.4548 | 0.4047 |
| video | 8B | Qwen3-VL-8B-Instruct | event | 822 | 0.5243 | 0.5444 | 0.4546 |
| video | 8B | Qwen3-VL-8B-Instruct | ir | 1561 | 0.5567 | 0.5862 | 0.4261 |
| video | 8B | Qwen3-VL-8B-Instruct | rgb | 2318 | 0.5703 | 0.5951 | 0.4264 |

## Best Configuration By Modality And Section

This is a descriptive maximum across all compared input types, sizes, and models, not a paired significance test.

| Modality | Section | Best input | Size | Model | N | Judge strict | Judge soft |
|---|---|---|---:|---|---:|---:|---:|
| depth | action | frame 8 | 8B | Qwen3-VL-8B-Instruct | 151 | 0.4106 | 0.4272 |
| depth | counting | frame 8 | 4B | Molmo2-4B | 103 | 0.2718 | 0.2718 |
| depth | dynamic_counting | frame 8 | 8B | Qwen3-VL-8B-Instruct | 118 | 0.5254 | 0.5254 |
| depth | dynamic_recognition | frame 8 | 8B | InternVL3_5-8B-Instruct | 131 | 0.5267 | 0.5496 |
| depth | navigation | frame 8 | 8B | Qwen3-VL-8B-Instruct | 53 | 0.7736 | 0.7736 |
| depth | non_common | frame 8 | 4B | Qwen3-VL-4B-Instruct | 56 | 0.9643 | 0.9643 |
| depth | object_recognition | frame 8 | 4B | InternVL3_5-4B-Instruct | 49 | 0.2653 | 0.2959 |
| depth | scene_sequence | frame 8 | 4B | InternVL3_5-4B-Instruct | 53 | 0.7358 | 0.7453 |
| depth | spatial_reasoning | video | 4B | InternVL3_5-4B-Instruct | 50 | 0.7000 | 0.7000 |
| event | action | video | 8B | Qwen3-VL-8B-Instruct | 155 | 0.5871 | 0.6290 |
| event | counting | frame 8 | 8B | InternVL3_5-8B-Instruct | 139 | 0.5755 | 0.5755 |
| event | dynamic_counting | frame 8 | 8B | Qwen3-VL-8B-Instruct | 130 | 0.6462 | 0.6462 |
| event | dynamic_recognition | frame 8 | 4B | InternVL3_5-4B-Instruct | 123 | 0.6423 | 0.6748 |
| event | navigation | frame 8 | 8B | Qwen3-VL-8B-Instruct | 54 | 0.4259 | 0.4630 |
| event | non_common | frame 8 | 4B | Qwen3-VL-4B-Instruct | 57 | 0.9825 | 0.9912 |
| event | object_recognition | video | 4B | Molmo2-4B | 54 | 0.6296 | 0.6574 |
| event | scene_sequence | video | 8B | Qwen3-VL-8B-Instruct | 55 | 0.2545 | 0.2909 |
| event | spatial_reasoning | frame 8 | 4B | Qwen3-VL-4B-Instruct | 55 | 0.6000 | 0.6000 |
| ir | action | frame 8 | 8B | Qwen3-VL-8B-Instruct | 163 | 0.6258 | 0.6748 |
| ir | counting | frame 8 | 8B | InternVL3_5-8B-Instruct | 177 | 0.6667 | 0.6695 |
| ir | dynamic_counting | frame 8 | 8B | Qwen3-VL-8B-Instruct | 170 | 0.7471 | 0.7500 |
| ir | dynamic_recognition | frame 8 | 4B | Qwen3-VL-4B-Instruct | 165 | 0.6121 | 0.6273 |
| ir | light_change | video | 4B | InternVL3_5-4B-Instruct | 176 | 0.6023 | 0.6080 |
| ir | light_recognition | frame 8 | 8B | Qwen3-VL-8B-Instruct | 52 | 0.5962 | 0.6538 |
| ir | navigation | frame 8 | 8B | Qwen3-VL-8B-Instruct | 165 | 0.6000 | 0.6152 |
| ir | non_common | frame 8 | 4B | InternVL3_5-4B-Instruct | 56 | 0.9107 | 0.9286 |
| ir | object_recognition | frame 8 | 8B | Qwen3-VL-8B-Instruct | 51 | 0.5490 | 0.6667 |
| ir | scene_sequence | frame 8 | 8B | Qwen3-VL-8B-Instruct | 177 | 0.5480 | 0.5932 |
| ir | spatial_reasoning | frame 8 | 8B | Qwen3-VL-8B-Instruct | 50 | 0.6600 | 0.6600 |
| ir | text_recognition | video | 4B | Qwen3-VL-4B-Instruct | 159 | 0.6918 | 0.7075 |
| rgb | action | frame 8 | 8B | Qwen3-VL-8B-Instruct | 266 | 0.6767 | 0.7143 |
| rgb | counting | frame 8 | 8B | InternVL3_5-8B-Instruct | 273 | 0.6154 | 0.6154 |
| rgb | dynamic_counting | frame 8 | 4B | Qwen3-VL-4B-Instruct | 279 | 0.7993 | 0.8065 |
| rgb | dynamic_recognition | frame 8 | 8B | Molmo2-8B | 157 | 0.5032 | 0.5159 |
| rgb | light_change | frame 8 | 4B | Qwen3-VL-4B-Instruct | 288 | 0.5486 | 0.5521 |
| rgb | light_recognition | frame 8 | 8B | Qwen3-VL-8B-Instruct | 54 | 0.6667 | 0.6852 |
| rgb | navigation | frame 8 | 8B | Qwen3-VL-8B-Instruct | 273 | 0.5714 | 0.6117 |
| rgb | non_common | frame 8 | 4B | Qwen3-VL-4B-Instruct | 68 | 0.8824 | 0.8824 |
| rgb | object_recognition | video | 4B | Qwen3-VL-4B-Instruct | 50 | 0.6800 | 0.7300 |
| rgb | scene_sequence | video | 8B | Qwen3-VL-8B-Instruct | 303 | 0.4752 | 0.5281 |
| rgb | spatial_reasoning | video | 8B | Qwen3-VL-8B-Instruct | 51 | 0.6667 | 0.6765 |
| rgb | text_recognition | video | 8B | Qwen3-VL-8B-Instruct | 256 | 0.4961 | 0.5098 |
