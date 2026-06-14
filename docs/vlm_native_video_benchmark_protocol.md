# 8B VLM Native-Video Input Benchmark Protocol

This document records the data, video-processing rules, runtime environment,
model configuration, execution procedure, and output format used to evaluate
Qwen3-VL-8B, InternVL3-8B, and Molmo2-8B on aligned QA with video inputs.

## 1. Evaluation Objective

This benchmark evaluates each deployed model system with day and night videos
instead of a shared set of preselected images.

- All models answer the same 5,465 aligned QA items.
- Each QA item is associated with one day/with-light video and one
  night/no-light video from the same segment and modality.
- Batch size, output length, GPU, QA set, and failure policy are shared.
- Video decoding, temporal sampling, and visual tokenization follow the
  supported path of each model adapter.
- The three adapters do not produce identical visual evidence. Results measure
  the complete model-plus-video-adapter system, not a strictly controlled
  frame-input comparison.
- Results from this benchmark must be reported separately from the fixed
  eight-frame benchmark.

The fixed eight-frame protocol is documented in:

```text
docs/vlm_8frame_benchmark_protocol.md
```

## 2. Data Sources

QA source:

```text
outputs/aligned_qa_valid_items.json
```

Dataset root:

```text
aligned_dataset
```

Each selected QA item contains:

- QA ID, modality, section, pair key, question, and ground-truth answer;
- one day/with-light video;
- one night/no-light video.

The visual modalities and QA counts are:

| Modality | QA count |
|---|---:|
| RGB | 2318 |
| IR | 1561 |
| Event | 822 |
| Depth | 764 |
| Total | 5465 |

Audio is not included. Models must answer using only the visual videos and the
question.

Each source video is normally 30 seconds long at 30 FPS, with approximately
900 original frames. A QA item therefore normally references two 30-second
videos from the same aligned segment.

## 3. Fixed Video Manifest

Video manifest:

```text
outputs/benchmarks/vlm_native_video/video_manifest.json
```

Manifest type:

```text
aligned_native_day_night_video_v1
```

Current manifest SHA-256:

```text
357d81f8e694c514641c2e3aca52b7feeec9a7933776fbcf3aaaf5d4ea8cb7b0
```

The manifest fixes the QA items, ground-truth answers, day/night video paths,
video metadata, modality, and input order. Formal result files must record and
match this hash.

The input order is always:

```text
Video 1: day/with-light
Video 2: night/no-light
```

Do not rebuild or edit the manifest between model runs unless the experiment
is intentionally redefined.

## 4. Shared Prompt and Inference Rules

The prompt:

- labels the first observation as day/with-light;
- labels the second observation as night/no-light;
- identifies the modality and QA section;
- instructs the model to use only the supplied visual evidence;
- prohibits captions, hidden metadata, audio, and outside knowledge;
- requests one concise answer without an explanation.

Shared inference settings:

| Setting | Value |
|---|---|
| Batch size | 1 QA item |
| Source videos per QA | 2 |
| Video order | Day, then night |
| `max_new_tokens` | 128 |
| `do_sample` | `false` |
| Concurrent models | 1 |
| CPU/GPU offload | Not used as an active capacity extension |
| Resume | Enabled |
| Soft timeout | 300 seconds |
| CPU preprocessing cache | LRU, 4 video pairs |
| CUDA allocator | `expandable_segments:True` |

The soft timeout is recorded after a call returns; it does not forcibly
terminate a GPU kernel.

OOM, context overflow, empty answers, ordinary failures, and soft timeouts must
remain in the result file. A failed item must not be rerun with a lower input
budget and mixed into the same formal result.

## 5. Model-Specific Video Processing

The model-specific paths below are intentional and must be disclosed when
comparing accuracy.

### Qwen3-VL-8B

```text
Model: models/qwen/Qwen3-VL-8B-Instruct
Weights: bitsandbytes 4-bit NF4
Video path: qwen-vl-utils native video processing
```

Qwen receives the two source video paths through `qwen-vl-utils`.
`process_vision_info` performs decoding and returns sampled video tensors,
video metadata, and processor keyword arguments. The adapter passes the
returned metadata, including timing information, to the model processor.

Sampling is controlled by the Qwen video utility and processor rather than by
a shared fixed-frame manifest. The exact sampled metadata and tensor shape are
saved in each result item.

### InternVL3-8B

```text
Model: models/internvl/InternVL3-8B
Weights: bitsandbytes 8-bit
Sampling: deterministic uniform temporal segments
Segments per source video: 12
Total input frames: 24
Image size: 448 x 448
Maximum tiles per frame: 1
```

InternVL does not receive encoded video files directly in this adapter. The
external adapter uses Decord to select 12 uniformly distributed frames from
each source video, producing:

```text
day: 12 frames
night: 12 frames
total: 24 frames
```

The prompt labels the selected inputs as `DayFrame1...12` followed by
`NightFrame1...12`. Frame indices and timestamps are saved for every QA item.

This result cannot be attributed entirely to InternVL's own video sampling,
because temporal sampling is performed by the external adapter. A 30-frame
configuration was tested on the current hardware and caused OOM; 24 frames is
the selected formal configuration.

When FlashAttention2 is unavailable, InternVL falls back to its available
attention implementation. This affects runtime and memory but not the selected
frames.

### Molmo2-8B

```text
Model: models/molmo2/Molmo2-8B
Weight and compute dtype: bfloat16
Image processor: use_fast=false
Sampling utility: molmo-utils
Maximum sampling rate per source video: 1.25 FPS
```

Molmo2's current processor path accepts one video marker. The adapter therefore:

1. asks `molmo-utils` to sample the day and night videos independently at a
   maximum of 1.25 FPS;
2. concatenates the sampled day frames followed by the sampled night frames
   on the CPU;
3. supplies the combined array and explicit combined-video metadata as one
   model video;
4. tells the model where the day sequence ends and the night sequence begins.

For the usual 30-second videos, this produces:

```text
day: 39 frames
night: 39 frames
combined: 78 frames
```

Capacity probes produced the following successful configurations on the
current RTX 4090:

| Maximum FPS per source | Approximate combined frames | Peak GPU memory |
|---:|---:|---:|
| 1.00 | 62 | 20.31 GB |
| 1.10 | 68 | 20.72 GB |
| 1.25 | 78 | 21.39 GB |
| 1.50 | 92 | 22.33 GB |

The formal experiment uses 1.25 FPS to retain memory headroom. A 2 FPS
configuration, approximately 122 combined frames, caused OOM.

## 6. CPU Preprocessing Cache

The runner uses a CPU-side LRU cache with a default capacity of four video
pairs. Its purpose is to avoid repeatedly decoding and sampling the same
day/night videos when one segment is associated with multiple questions.

Cache keys include:

- absolute day and night paths;
- file size and nanosecond modification time;
- model-specific sampling settings;
- InternVL image size where applicable.

Consequently, changing or replacing a source video invalidates its cache entry.
The cache stores only decoded or preprocessed CPU inputs. Every QA item still
executes an independent model forward/generation call, so model answers do not
share conversational state or visual encoder outputs.

Cache use may reduce wall-clock time. It does not change the selected video
content for a given model configuration. Cache hit status and preprocessing
time are saved in the input statistics.

## 7. Software and Hardware Environment

Environment recorded for the 8B video runs on June 13, 2026:

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
NVIDIA driver: 595.71.05
```

Only one model may use the GPU at a time. Latency and GPU-memory comparisons
are valid only when the hardware, software environment, cache policy, and
model processing configuration are also reported.

## 8. Execution

Benchmark script:

```text
scripts/run_vlm_native_video_smoke.py
```

Despite the filename, this script supports both smoke tests and full formal
runs.

### Preflight Smoke Test

The default inference limit is one item per modality. For example:

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models internvl
```

This processes at most four QA items. It is not a formal full run.

### Formal Runs

The `--max-items-per-modality 0` argument is mandatory for a full 5,465-item
run.

Qwen3-VL-8B:

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models qwen_vl \
  --max-items-per-modality 0
```

InternVL3-8B:

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models internvl \
  --max-items-per-modality 0
```

Molmo2-8B:

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models molmo2 \
  --molmo2-max-fps 1.25 \
  --max-items-per-modality 0 \
  --experiment-dir outputs/benchmarks/vlm_native_video_molmo_fps_1_25
```

Resume is enabled by default. After interruption, run the same command again.
Do not add `--no-resume` when continuing a formal run. Do not run two models
concurrently on the same GPU.

## 9. Result Format

Each model writes one JSON file and one CSV file under the selected experiment
directory's `results/` folder.

Each result item stores:

- QA ID, modality, section, and pair key;
- question, ground-truth answer, and model answer;
- complete day/night video paths and metadata;
- status and failure reason;
- per-item latency;
- baseline, peak, and incremental peak GPU memory;
- model-specific sampled-frame, timestamp, tensor, and cache statistics;
- generation settings, model path, and quantization configuration.

A valid completed result has:

```text
status = answered
reason = ""
model_answer is non-empty
```

The JSON metadata records the manifest hash, model configuration, total and
attempted item counts, status counts, cache policy, and video-processing
policy. The CSV is a flattened companion representation of the same results.

Answer generation and answer correctness evaluation are separate stages. This
runner records model answers and ground truth but does not calculate accuracy
or invoke an LLM judge.

## 10. Known Limitations

- The models do not receive identical sampled frames or equal visual-token
  budgets.
- Qwen uses its supported video utility, InternVL uses external deterministic
  sampling, and Molmo2 uses official sampling followed by adapter-side
  day/night concatenation.
- The benchmark therefore compares deployable video-input systems rather than
  isolated model weights under identical visual evidence.
- CPU preprocessing cache changes runtime, but not per-item inference
  independence or visual evidence.
- The same source videos are decoded again after cache eviction.
- Brief events may be missed by any temporal sampler.
- Molmo2's concatenated timeline is an adapter representation, not an original
  encoded day-plus-night video file.
- InternVL is restricted to one visual tile per sampled frame.
- Generation uses model-family-specific numerical precision.
- Accuracy must be calculated by a separate metric or judge, and the metric
  configuration must be reported independently.

## 11. Run Status

As of June 13, 2026:

| Model | Configuration | Status |
|---|---|---|
| Molmo2-8B | 1.25 FPS per source, 78 typical combined frames | Complete: 5465 answered, 0 failed |
| InternVL3-8B | 12 frames per source, 24 total | Formal run in progress |
| Qwen3-VL-8B | Qwen native video utility | Formal run pending |

The completed Molmo2 run used approximately 4 hours 57 minutes of summed
per-item latency, with a maximum observed GPU allocation of approximately
21.39 GB. These figures describe answer generation only, not answer accuracy.

---

# 中文版

# 8B VLM 原生视频输入评估规范

本文记录本项目使用视频输入评估 Qwen3-VL-8B、InternVL3-8B 和 Molmo2-8B
时采用的数据、视频处理规则、运行环境、模型配置、执行流程和结果格式。

## 1. 评估目标

本评估让各模型系统接收白天和夜间视频，而不是所有模型共享同一组预选图像。

- 所有模型回答相同的 5465 条 aligned QA。
- 每条 QA 对应同一 segment 和 modality 下的一段 day/with-light 视频以及一段
  night/no-light 视频。
- Batch size、最大输出长度、GPU、QA 集和失败处理规则保持一致。
- 视频解码、时间采样和视觉 token 化采用各模型适配器支持的路径。
- 三个适配器不会产生完全相同的视觉证据，因此结果衡量的是“模型 + 视频适配器”
  的完整系统，不是严格控制帧输入后的纯模型对比。
- 本评估必须与固定 8 帧评估分开报告。

固定 8 帧评估规范位于：

```text
docs/vlm_8frame_benchmark_protocol.md
```

## 2. 数据源

QA 来源：

```text
outputs/aligned_qa_valid_items.json
```

数据集根目录：

```text
aligned_dataset
```

每条 QA 包含：

- QA ID、modality、section、pair key、问题和 ground truth；
- 一段 day/with-light 视频；
- 一段 night/no-light 视频。

各视觉 modality 的 QA 数量如下：

| Modality | QA 数量 |
|---|---:|
| RGB | 2318 |
| IR | 1561 |
| Event | 822 |
| Depth | 764 |
| 总计 | 5465 |

本评估不包含 Audio。模型只能根据视频视觉内容和问题作答。

每个源视频通常为 30 秒、30 FPS，约含 900 个原始帧。因此，每条 QA 通常关联
同一 aligned segment 下的两段 30 秒视频。

## 3. 固定视频清单

视频 manifest：

```text
outputs/benchmarks/vlm_native_video/video_manifest.json
```

Manifest 类型：

```text
aligned_native_day_night_video_v1
```

当前 manifest SHA-256：

```text
357d81f8e694c514641c2e3aca52b7feeec9a7933776fbcf3aaaf5d4ea8cb7b0
```

该 manifest 固定 QA、ground truth、白天和夜间视频路径、视频 metadata、modality
以及输入顺序。正式结果必须保存并匹配该哈希。

输入顺序始终为：

```text
Video 1：day/with-light
Video 2：night/no-light
```

除非有意重新定义实验，否则不得在不同模型之间重建或修改 manifest。

## 4. 统一 Prompt 与推理规则

Prompt 会：

- 标记第一段视频为 day/with-light；
- 标记第二段视频为 night/no-light；
- 标明 modality 和 QA section；
- 要求只使用所提供的视觉证据；
- 禁止使用 caption、隐藏 metadata、音频和外部知识；
- 要求仅返回简短答案，不附带解释。

统一推理配置：

| 配置 | 值 |
|---|---|
| Batch size | 1 条 QA |
| 每条 QA 的源视频 | 2 段 |
| 视频顺序 | Day 在前，night 在后 |
| `max_new_tokens` | 128 |
| `do_sample` | `false` |
| 并行模型数 | 1 |
| CPU/GPU offload | 不作为主动扩容手段 |
| 断点续跑 | 启用 |
| Soft timeout | 300 秒 |
| CPU 预处理缓存 | LRU，4 组视频对 |
| CUDA allocator | `expandable_segments:True` |

Soft timeout 在调用返回后根据耗时标记，不会强制中断正在执行的 GPU kernel。

OOM、上下文溢出、空答案、普通失败和 soft timeout 都必须保留在结果文件中。
不得针对失败项降低输入预算后再混入同一正式结果。

## 5. 各模型的视频处理方式

下列模型差异是当前实验设计的一部分，比较准确率时必须明确说明。

### Qwen3-VL-8B

```text
模型：models/qwen/Qwen3-VL-8B-Instruct
权重：bitsandbytes 4-bit NF4
视频路径：qwen-vl-utils 原生视频处理
```

Qwen 通过 `qwen-vl-utils` 接收两段源视频路径。`process_vision_info` 负责解码，
并返回采样后的视频 tensor、视频 metadata 和 processor 参数。适配器会将包括
时间信息在内的 metadata 传递给模型 processor。

采样由 Qwen 视频工具和 processor 决定，而不是由共享的固定帧 manifest 决定。
每条结果会保存实际输入 metadata 和 tensor shape。

### InternVL3-8B

```text
模型：models/internvl/InternVL3-8B
权重：bitsandbytes 8-bit
采样：确定性时间均匀分段
每段源视频采样数：12
总输入帧数：24
图像尺寸：448 x 448
每帧最大 tile：1
```

当前适配器不会把编码后的视频文件直接传给 InternVL，而是通过 Decord 对每段
源视频均匀选择 12 帧：

```text
day：12 帧
night：12 帧
总计：24 帧
```

Prompt 将其标记为 `DayFrame1...12`，随后是 `NightFrame1...12`。每条结果都会
保存帧编号和时间戳。

由于时间采样发生在外部适配层，InternVL 的结果不能完全归因于模型自身的视频
采样能力。当前硬件上测试 30 帧配置会 OOM，因此正式配置选择 24 帧。

未安装 FlashAttention2 时，InternVL 使用可用的 attention 实现。这会影响速度
和显存，但不会改变选中的帧。

### Molmo2-8B

```text
模型：models/molmo2/Molmo2-8B
权重与计算 dtype：bfloat16
Image processor：use_fast=false
采样工具：molmo-utils
每段源视频的最大采样率：1.25 FPS
```

Molmo2 当前 processor 路径只接受一个 video marker，因此适配器会：

1. 使用 `molmo-utils` 分别按最高 1.25 FPS 采样 day 和 night 视频；
2. 在 CPU 上将 day 帧放在前面、night 帧放在后面进行拼接；
3. 将拼接后的数组和明确的 combined-video metadata 作为一段模型视频输入；
4. 在 prompt 中告知模型 day 序列结束位置和 night 序列开始位置。

对通常的 30 秒视频，输入约为：

```text
day：39 帧
night：39 帧
合计：78 帧
```

当前 RTX 4090 上的容量测试如下：

| 每段源视频最大 FPS | 近似总帧数 | 峰值显存 |
|---:|---:|---:|
| 1.00 | 62 | 20.31 GB |
| 1.10 | 68 | 20.72 GB |
| 1.25 | 78 | 21.39 GB |
| 1.50 | 92 | 22.33 GB |

正式实验采用 1.25 FPS，以保留显存余量。2 FPS 约产生 122 个合并帧，测试时
发生 OOM。

## 6. CPU 预处理缓存

Runner 默认使用容量为 4 组视频对的 CPU LRU 缓存。其目的在于：同一 segment
关联多条问题时，避免重复解码和采样完全相同的 day/night 视频。

缓存 key 包含：

- day 和 night 的绝对路径；
- 文件大小和纳秒级修改时间；
- 模型对应的采样配置；
- InternVL 的图像尺寸等参数。

因此，源视频被修改或替换后，旧缓存不会被错误复用。缓存只保存 CPU 侧已解码
或预处理的输入。每条 QA 仍然独立执行一次模型 forward/generation，不共享对话
状态，也不缓存视觉编码器输出。

缓存可以减少总运行时间，但不会改变同一模型配置下某条 QA 获得的视频内容。
每条结果会保存缓存命中状态和预处理耗时。

## 7. 软件与硬件环境

2026 年 6 月 13 日的 8B 视频评估环境：

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
NVIDIA driver：595.71.05
```

同一时间只能有一个模型使用 GPU。比较延迟和显存时，必须同时报告硬件、软件
环境、缓存策略和各模型的视频处理配置。

## 8. 执行方式

评估脚本：

```text
scripts/run_vlm_native_video_smoke.py
```

虽然文件名包含 `smoke`，该脚本同时支持小规模测试和完整正式评估。

### 小规模测试

脚本默认每个 modality 最多推理 1 条。例如：

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models internvl
```

该命令最多只处理 4 条 QA，不是完整正式评估。

### 正式评估

完整运行 5465 条 QA 时，必须添加 `--max-items-per-modality 0`。

Qwen3-VL-8B：

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models qwen_vl \
  --max-items-per-modality 0
```

InternVL3-8B：

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models internvl \
  --max-items-per-modality 0
```

Molmo2-8B：

```bash
.venv/bin/python scripts/run_vlm_native_video_smoke.py \
  --models molmo2 \
  --molmo2-max-fps 1.25 \
  --max-items-per-modality 0 \
  --experiment-dir outputs/benchmarks/vlm_native_video_molmo_fps_1_25
```

默认启用断点续跑。中断后执行完全相同的命令即可继续。继续正式评估时不要添加
`--no-resume`，同一张 GPU 上也不得同时运行两个模型。

## 9. 结果格式

每个模型在所选实验目录的 `results/` 下分别生成一份 JSON 和一份 CSV。

每条结果保存：

- QA ID、modality、section 和 pair key；
- question、ground truth 和 model answer；
- 完整 day/night 视频路径及 metadata；
- 状态和失败原因；
- 单题耗时；
- baseline、peak 和 incremental peak GPU memory；
- 模型对应的采样帧、时间戳、tensor 和缓存统计；
- 生成参数、模型路径和量化配置。

有效完成项满足：

```text
status = answered
reason = ""
model_answer 非空
```

JSON metadata 保存 manifest hash、模型配置、总条目数、已尝试条目数、状态统计、
缓存策略和视频处理策略。CSV 是同一批结果的扁平化配套表示。

回答生成与答案正确性评估属于两个阶段。该 runner 会保存模型答案和 ground truth，
但不会计算准确率，也不会调用 LLM Judge。

## 10. 已知限制

- 三个模型得到的采样帧和视觉 token 预算并不相同。
- Qwen 使用其视频工具；InternVL 使用外部确定性采样；Molmo2 使用官方采样后再
  由适配层拼接 day/night。
- 因此，本评估比较的是可部署视频输入系统，不是在完全相同视觉证据下隔离比较
  模型权重。
- CPU 缓存会改变运行时间，但不会改变单题推理独立性或视觉证据。
- 视频对被缓存淘汰后，再次出现时仍需要重新解码。
- 所有时间采样器都有可能遗漏短暂事件。
- Molmo2 的合并时间线是适配器表示，不是原始编码的 day+night 拼接视频文件。
- InternVL 每个采样帧被限制为一个视觉 tile。
- 不同模型家族使用不同的数值精度配置。
- 准确率需要由独立 metric 或 Judge 计算，并单独报告评估配置。

## 11. 当前运行状态

截至 2026 年 6 月 13 日：

| 模型 | 配置 | 状态 |
|---|---|---|
| Molmo2-8B | 每侧 1.25 FPS，通常合计 78 帧 | 已完成：5465 条回答，0 失败 |
| InternVL3-8B | 每侧 12 帧，合计 24 帧 | 正式评估进行中 |
| Qwen3-VL-8B | Qwen 原生视频工具 | 正式评估待运行 |

Molmo2 已完成结果的单题耗时总和约为 4 小时 57 分钟，观察到的最大 GPU allocation
约为 21.39 GB。以上数据只表示回答生成，不表示答案准确率。
