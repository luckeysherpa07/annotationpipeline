# 8B VLM Frame-Input Benchmark Protocol

This document records the data, sampling procedure, runtime environment, model
configuration, and output format used to evaluate Qwen3-VL-8B, InternVL3-8B,
and Molmo2-8B on aligned QA with frame inputs.

## 1. Evaluation Objective

This benchmark compares the models under identical visual evidence.

- All models answer the same QA items.
- All models receive exactly the same eight images for a given QA item.
- Native model-specific video sampling is not evaluated here.
- Native full-video input must be reported as a separate experiment.

## 2. Data Sources

QA source:

```text
outputs/aligned_qa_valid_items.json
```

Frame-cache root:

```text
aligned_dataset
```

Visual modalities and cache directories:

| Modality | Cache directory |
|---|---|
| RGB | `aligned_dataset/.frames_cache` |
| IR | `aligned_dataset/.frames_cache_ir` |
| Event | `aligned_dataset/.frames_cache_event` |
| Depth | `aligned_dataset/.frames_cache_marigold` |

Audio is not part of this frame-input benchmark.

Each aligned segment normally contains:

- one 30-second day/with-light video;
- one 30-second night/no-light video;
- approximately 900 original frames per side at 30 FPS;
- approximately 30 cached images per side, normally extracted at 1 FPS.

The formal QA set contains 5,465 items:

| Modality | QA count |
|---|---:|
| RGB | 2318 |
| IR | 1561 |
| Event | 822 |
| Depth | 764 |

## 3. Fixed Manifests

QA manifest:

```text
outputs/benchmarks/vlm_8frame_aligned/fixed_qa_manifest.json
```

This file fixes the QA items, ground-truth answers, and aligned segments used
by the experiment. It does not select the final eight frames.

Eight-frame input manifest:

```text
outputs/benchmarks/vlm_8frame_aligned/manifests/frames_8.json
```

This file records the eight image paths supplied for every QA item. All models
must read the same manifest.

Current manifest SHA-256:

```text
ce1b15ad21ec8e429b71a7b71e5e2ab4d08453ec74ce76409efde3d9082ce8b3
```

Formal result files must record and match this hash. The manifest must not be
regenerated or modified between models unless the experiment itself is
redefined.

## 4. Frame-Sampling Procedure

Each QA item receives eight frames:

```text
day/with-light: 4 frames
night/no-light: 4 frames
```

Algorithm version:

```text
referenced_or_nearest_then_stratified_uniform_v2
```

Rules:

1. Day and night candidates are processed independently; no mixed candidate
   pool is used.
2. Candidate frames are deduplicated and stably sorted by original frame
   number and path.
3. If the question explicitly references a frame number, that frame is
   selected first.
4. If the exact frame is absent from the cache, the nearest cached frame is
   selected. Ties are resolved in favor of the lower frame number.
5. Remaining slots are filled by deterministic uniform sampling from the
   remaining candidates on that side.
6. Input order is fixed: four day frames followed by four night frames.
7. Sampling does not use randomness, an LLM, a VLM, or a motion-scoring model.

Twelve questions in the formal manifest explicitly reference a frame number:

- 11 references match an exact cached frame;
- one question references `frame 700`, which is absent from the 1 FPS cache,
  so the nearest cached frame, `frame 690`, is used.

## 5. Inference Rules

Shared resource and generation rules:

| Setting | Value |
|---|---|
| GPU | One NVIDIA GeForce RTX 4090, 24 GB |
| Batch size | 1 QA item |
| Input images | 8 |
| Day/night allocation | 4 + 4 |
| `max_new_tokens` | 128 |
| `do_sample` | `false` |
| CPU offload | Not used as an active capacity extension |
| Concurrent models | 1 |
| Resume support | Enabled |
| CUDA allocator | `expandable_segments:True` |

Models must not run concurrently. OOM errors, empty answers, and other failures
must be recorded. A failed QA item must not be rerun with fewer frames and then
mixed into the same result set.

## 6. Model Configurations

### Qwen3-VL-8B-Instruct

```text
Model: models/qwen/Qwen3-VL-8B-Instruct
Weights: bitsandbytes 4-bit NF4
Compute dtype: float16
```

### InternVL3-8B

```text
Model: models/internvl/InternVL3-8B
Weights: bitsandbytes 8-bit
Image size: 448 x 448
Maximum tiles per frame: 1
```

The one-tile limit is part of the formal configuration. InternVL's default
maximum of 12 tiles per image produces substantially more visual tokens and
is not suitable for a shared eight-frame benchmark on the current 24 GB GPU.

When FlashAttention2 is unavailable, InternVL falls back to eager attention.
This does not change the selected frames, but it can affect runtime and memory
usage and must therefore be reported.

### Molmo2-8B

```text
Model: models/molmo2/Molmo2-8B
Weight and compute dtype: bfloat16
Image processor: use_fast=false
```

Generic full-model bitsandbytes 4-bit loading triggers a Byte LayerNorm
compatibility error. The formal benchmark therefore uses the verified BF16
configuration.

## 7. Software and Hardware Environment

Environment recorded on June 13, 2026:

```text
Operating system: Ubuntu 22.04.5 LTS
Kernel: Linux 6.8.0-124-generic x86_64
Python: 3.10.12
PyTorch: 2.12.0+cu130
CUDA runtime: 13.0
Transformers: 4.57.1
bitsandbytes: 0.49.2
Pillow: 12.2.0
OpenCV: 4.13.0
GPU: NVIDIA GeForce RTX 4090, 24564 MiB
```

Model versions are determined by the contents of the local model directories.
For archival reproducibility, the Git commit and model source revisions or
model-file hashes should also be saved. The current working tree has not yet
been identified by a commit containing this complete experiment setup.

## 8. Execution

Benchmark script:

```text
scripts/run_vlm_8frame_smoke.py
```

Formal execution for one model:

```bash
.venv/bin/python scripts/run_vlm_8frame_smoke.py \
  --frame-counts 8 \
  --items-per-modality 0 \
  --experiment-dir outputs/benchmarks/vlm_8frame_aligned \
  --qa-manifest outputs/benchmarks/vlm_8frame_aligned/fixed_qa_manifest.json \
  --models qwen_vl
```

Replace the final value with `internvl` or `molmo2` for the other models. Do
not add:

```text
--no-resume
--rebuild-qa-manifest
```

For a preflight smoke test, add:

```text
--max-items-per-modality 1
```

This runs one QA item from each of the four modalities. Remove the option to
continue with the full dataset.

## 9. Result Format

Each model produces one JSON file and one CSV file under:

```text
outputs/benchmarks/vlm_8frame_aligned/frames_8/
```

Each result item stores:

- QA ID, modality, section, and pair key;
- question, ground-truth answer, and model answer;
- day/night frame counts and complete frame paths;
- status and failure reason;
- per-item latency;
- baseline, peak, and incremental peak GPU memory;
- processor input statistics when available;
- generation settings;
- model name and quantization configuration.

A valid completed result has:

```text
status = answered
reason = ""
model_answer is non-empty
```

`failed` and `oom` results must remain in the output and must not be silently
removed.

## 10. Known Limitations

- Eight frames are the shared stable input on the current hardware and model
  configurations, not the theoretical maximum of every model.
- Eight frames can miss brief actions and fine-grained temporal information
  from a 30-second video.
- The models use different numerical precision settings. Results represent
  each model together with its current deployable configuration.
- InternVL is restricted to one tile per frame, limiting its native dynamic
  tiling capability.
- This benchmark does not measure native video decoding, sampling, or temporal
  compression.
- Some answers exceed the requested concise format and should be scored as
  produced by the downstream judge.
- Answer generation and correctness judging are separate stages. This script
  only generates answers.

## 11. Completed Runs

As of June 13, 2026:

| Model | Completed | Failed/OOM | Mean latency | Mean peak GPU memory |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 5465 | 0 | 1.896 s | 8.46 GB |
| InternVL3-8B | 5465 | 0 | 0.765 s | 10.09 GB |
| Molmo2-8B | Formal run pending confirmation | - | Smoke test: about 2.8-3.5 s | Smoke test: about 21.01 GB |

This table records answer-generation completion only; it does not report answer
accuracy.

---

# 中文版

# 8B VLM 帧输入评估规范

本文记录本项目对 Qwen3-VL-8B、InternVL3-8B 和 Molmo2-8B 进行 aligned QA
帧输入评估时采用的数据、采样规则、运行环境和结果格式。

## 1. 评估目标

本评估用于比较三个模型在获得相同视觉证据时的问答能力。

- 所有模型回答相同的 QA。
- 同一条 QA 对三个模型使用完全相同的 8 张图像。
- 本评估不包含模型原生的视频采样能力。
- 原生完整视频输入应作为另一套独立实验报告。

## 2. 数据源

QA 来源：

```text
outputs/aligned_qa_valid_items.json
```

帧缓存根目录：

```text
aligned_dataset
```

视觉 modality 与缓存目录：

| Modality | 缓存目录 |
|---|---|
| RGB | `aligned_dataset/.frames_cache` |
| IR | `aligned_dataset/.frames_cache_ir` |
| Event | `aligned_dataset/.frames_cache_event` |
| Depth | `aligned_dataset/.frames_cache_marigold` |

Audio 不属于本次帧输入评估。

每个 aligned segment 通常包含：

- 一段 30 秒的 day/with-light 视频；
- 一段 30 秒的 night/no-light 视频；
- 原视频通常为 30 FPS，每侧约 900 个原始视频帧；
- 缓存通常按 1 FPS 提取，每侧约 30 张图像。

正式 QA 清单共 5465 条：

| Modality | QA 数量 |
|---|---:|
| RGB | 2318 |
| IR | 1561 |
| Event | 822 |
| Depth | 764 |

## 3. 固定清单

QA 清单：

```text
outputs/benchmarks/vlm_8frame_aligned/fixed_qa_manifest.json
```

它固定本次实验使用的 QA、ground truth 和 aligned segment，不负责决定具体帧。

8 帧输入清单：

```text
outputs/benchmarks/vlm_8frame_aligned/manifests/frames_8.json
```

它记录每条 QA 最终输入模型的 8 个帧路径。所有模型必须读取同一份清单。

当前 manifest SHA-256：

```text
ce1b15ad21ec8e429b71a7b71e5e2ab4d08453ec74ce76409efde3d9082ce8b3
```

正式结果必须保存并匹配该哈希。除非重新定义实验，否则不得在模型之间重新生成
或修改 manifest。

## 4. 帧采样规则

每条 QA 输入 8 帧：

```text
day/with-light：4 帧
night/no-light：4 帧
```

算法版本：

```text
referenced_or_nearest_then_stratified_uniform_v2
```

具体规则：

1. day 和 night 候选帧分开处理，不使用混合候选池。
2. 候选帧按原始 frame number 和路径稳定排序并去重。
3. 如果问题明确引用 frame number，优先选择该编号。
4. 如果缓存中没有精确编号，选择时间上最近的缓存帧；编号距离相同时选择较小编号。
5. 剩余名额从对应侧的剩余候选帧中进行确定性均匀采样。
6. 最终输入顺序固定为 day 4 帧在前，night 4 帧在后。
7. 采样不使用随机数、LLM、VLM 或运动强度模型。

正式 manifest 中有 12 条问题引用明确帧号：

- 11 条命中精确缓存帧；
- 1 条引用 `frame 700`，缓存中不存在该编号，因此使用最近的 `frame 690`。

## 5. 推理规则

统一规则：

| 配置 | 值 |
|---|---|
| GPU | 单张 NVIDIA GeForce RTX 4090，24 GB |
| Batch size | 1 条 QA |
| 输入图像数 | 8 |
| Day/night | 4 + 4 |
| `max_new_tokens` | 128 |
| `do_sample` | `false` |
| CPU offload | 不作为主动扩容手段 |
| 并行模型数 | 1 |
| 断点续跑 | 启用 |
| CUDA allocator | `expandable_segments:True` |

模型不得同时运行，以免互相占用 GPU 显存。OOM、空答案和普通异常必须记录，
不得针对失败 QA 临时降低帧数后混入同一结果。

## 6. 模型配置

### Qwen3-VL-8B-Instruct

```text
模型：models/qwen/Qwen3-VL-8B-Instruct
权重：bitsandbytes 4-bit NF4
计算 dtype：float16
```

### InternVL3-8B

```text
模型：models/internvl/InternVL3-8B
权重：bitsandbytes 8-bit
图像尺寸：448 x 448
每帧最大 tile：1
```

限制为每帧 1 tile 是正式配置的一部分。默认最多 12 tile 会显著增加视觉 token，
不适合当前 24 GB GPU 上的 8 帧统一评估。

未安装 FlashAttention2 时，InternVL 使用 eager attention。该回退不改变输入帧，
但会影响速度和显存，因此应在报告中注明。

### Molmo2-8B

```text
模型：models/molmo2/Molmo2-8B
权重和计算 dtype：bfloat16
image processor：use_fast=false
```

通用 bitsandbytes 全模型 4-bit 加载会触发 Byte LayerNorm 兼容错误，因此正式评估
使用已验证可运行的 BF16 配置。

## 7. 软件环境

本轮实验环境记录于 2026-06-13：

```text
操作系统：Ubuntu 22.04.5 LTS
内核：Linux 6.8.0-124-generic x86_64
Python：3.10.12
PyTorch：2.12.0+cu130
CUDA runtime：13.0
Transformers：4.57.1
bitsandbytes：0.49.2
Pillow：12.2.0
OpenCV：4.13.0
GPU：NVIDIA GeForce RTX 4090，24564 MiB
```

模型版本由本地模型目录内容决定。正式归档时应同时保存 Git commit、模型目录来源
或模型文件哈希；当前工作树尚未以 commit 标识本轮配置。

## 8. 执行方式

脚本：

```text
scripts/run_vlm_8frame_smoke.py
```

正式运行一个模型：

```bash
.venv/bin/python scripts/run_vlm_8frame_smoke.py \
  --frame-counts 8 \
  --items-per-modality 0 \
  --experiment-dir outputs/benchmarks/vlm_8frame_aligned \
  --qa-manifest outputs/benchmarks/vlm_8frame_aligned/fixed_qa_manifest.json \
  --models qwen_vl
```

将最后一项分别替换为 `internvl` 或 `molmo2`。不要添加：

```text
--no-resume
--rebuild-qa-manifest
```

正式运行前的小批量测试使用：

```text
--max-items-per-modality 1
```

该参数会运行四种 modality 各 1 条；去掉后继续全集评估。

## 9. 结果格式

每个模型分别生成 JSON 和 CSV：

```text
outputs/benchmarks/vlm_8frame_aligned/frames_8/
```

每条结果保存：

- QA ID、modality、section 和 pair key；
- question、ground truth answer 和 model answer；
- day/night 帧数量及完整帧路径；
- 状态和失败原因；
- 单题耗时；
- 基线、峰值和增量峰值 GPU 显存；
- processor 可提供的输入统计；
- 生成参数；
- 模型名称及量化配置。

合法完成状态为：

```text
status = answered
reason = ""
model_answer 非空
```

`failed` 和 `oom` 必须保留在结果中，不应静默删除。

## 10. 已知限制

- 8 帧是三个模型在当前硬件和配置下共同稳定的输入，不是所有模型的理论最大帧数。
- 8 帧会丢失 30 秒视频中的短暂动作和细粒度时序信息。
- 三个模型采用不同精度配置，结果代表“模型加当前可运行部署配置”的系统能力。
- InternVL 每帧固定 1 tile，限制了其原生动态切图能力。
- 当前基准不能衡量模型原生视频解码、视频采样或时序压缩能力。
- 少量模型答案可能超过“简短回答”的要求，应由后续 judge 据实评分。
- 回答生成与正确性评分是两个阶段；本脚本只生成答案，不执行 judge。

## 11. 已完成运行

截至 2026-06-13：

| 模型 | 完成数量 | 失败/OOM | 平均单题耗时 | 平均峰值显存 |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 5465 | 0 | 1.896 秒 | 8.46 GB |
| InternVL3-8B | 5465 | 0 | 0.765 秒 | 10.09 GB |
| Molmo2-8B | 正式运行待确认 | - | 小批量约 2.8-3.5 秒 | 小批量约 21.01 GB |

该表仅记录回答生成是否完成，不表示答案正确率。
