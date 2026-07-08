# 多模态视频 QA 标注与 VLM 评估项目说明

本文档面向第一次接触本仓库的同学，目标是快速说明项目做什么、数据如何组织、视频如何预处理、QA 如何生成，以及后续如何评估视觉语言模型（VLM）。

## 1. 项目目标

本项目围绕日常第一视角视频构建多模态问答数据，并用这些问答数据评估不同 VLM 在多模态场景下的理解能力。

核心链路如下：

```text
原始多模态视频
  -> 视频/音频预处理与时间对齐
  -> 按任务片段切分 aligned_dataset
  -> 各模态 QA 生成
  -> QA 清洗、质量检查、归一化与融合
  -> 构建 valid QA benchmark
  -> VLM 帧输入 / 原生视频输入评估
  -> 指标计算、跨模态比较、图表与报告
```

项目覆盖的主要模态包括：

| 模态 | 含义 | 典型用途 |
|---|---|---|
| RGB | 普通可见光视频 | 物体、动作、文字、场景、颜色、光照 |
| IR | 红外/低光视频 | 夜间可见性、低光条件下的物体和动作 |
| Event | 事件相机/事件视频 | 运动、动态变化、动作阶段 |
| Depth | 深度视频或由 Marigold 估计的深度 | 空间关系、距离、导航、布局 |
| Audio | 视频中的音频轨道或配套音频 | 声源、动作确认、环境声音、时间线 |

## 2. 目录结构

仓库中最重要的目录如下：

| 路径 | 作用 |
|---|---|
| `annotation_feature/` | 核心 Python 包，包含预处理、时间对齐、QA 生成、融合、质量评估等逻辑 |
| `annotation_feature/pipeline/` | 主标注流水线，包括 Gemini 客户端、各模态 pipeline、共享工具 |
| `annotation_feature/pipeline/modalities/` | RGB、IR、Event、Depth、Audio 的具体 QA 生成逻辑 |
| `annotation_feature/qa_quality/` | aligned QA 的规则检查、LLM 评估、清洗、benchmark 运行与打分 |
| `annotation_feature/reasoning/` | 将各模态 QA 归一化为 evidence units，并做分组、导出和融合 |
| `prompts/` | 各模态 caption/question/answer prompt |
| `scripts/` | 一次性实验脚本、benchmark 脚本、图表和报告生成脚本 |
| `docs/` | 实验协议、项目说明文档和 caption schema 语义定义 |
| `qa_pairs/aligned/` | 对齐后的各模态 QA JSON |
| `aligned_dataset/` | 对齐并切分后的多模态数据集，以及缓存帧 |
| `segmented_outputs/` | 任务片段级 QA 结果 |
| `temporal_alignment_json/` | 时间对齐结果 JSON |
| `temporal_alignment_plots/` | 时间对齐可视化图 |
| `outputs/` | 清洗后的 QA、benchmark 结果、评估结果、图表和报告 |

## 3. 数据与视频采集约定

项目假设原始数据按“动作/场景 + 白天/夜间 + 模态”的方式组织。代码通过文件名推断样本、模态和 day/night side。

常见文件名形态类似：

```text
<sample>_day_rgb.mp4
<sample>_night_rgb.mp4
<sample>_day_ir.mp4
<sample>_night_ir.mp4
<sample>_day_event.mp4
<sample>_night_event.mp4
<sample>_day_depth.mp4
<sample>_night_depth.mp4
```

其中 `<sample>` 是动作或场景名称，例如 `cut_carrot`、`check_mailbox`、`make_coffee`。仓库中已有的对齐与评估产物显示，项目使用了白天/夜间成对视频，并围绕同一动作场景采集 RGB、IR、Event、Depth、Audio 等模态。

视频采集阶段需要尽量保证：

- 同一动作在 day/night 两种环境下都有对应记录。
- 同一动作的多个模态尽量同步开始、同步结束。
- 文件名包含可被程序识别的 side 和 modality，例如 `_day_rgb`、`_night_ir`。
- 视频帧率、时长和编码尽量稳定；时间对齐脚本会读取 fps、frame count、duration 等元数据。
- 音频通常从 RGB 或带音频的视频中提取并参与 RGB-Audio 对齐。

文件名解析、模态识别和 day/night 判断主要在 `annotation_feature/pipeline/utils.py` 中完成。

## 4. 视频预处理与帧缓存

预处理入口主要在 `annotation_feature/video_preprocessor.py`：

- `extract_frames(video_path, fps=1)`：从单个视频按指定 fps 抽帧。
- `preprocess_videos(dataset_folder, fps=1, video_type="rgb")`：递归扫描数据集，按模态抽帧并缓存。

不同模态的帧缓存目录约定如下：

| 模态 | 缓存目录 |
|---|---|
| RGB | `.frames_cache` |
| IR | `.frames_cache_ir` |
| Event | `.frames_cache_event` |
| Depth | `.frames_cache_depth` 或 `.frames_cache_marigold` |

在 aligned benchmark 中，默认缓存根目录是：

```text
aligned_dataset
```

例如 RGB、IR、Event、Marigold Depth 的缓存通常位于：

```text
aligned_dataset/.frames_cache
aligned_dataset/.frames_cache_ir
aligned_dataset/.frames_cache_event
aligned_dataset/.frames_cache_marigold
```

这些缓存帧后续会被 QA 生成和 VLM frame-input benchmark 复用。

## 5. 时间对齐与任务片段切分

时间对齐逻辑位于 `annotation_feature/temporal_alignment.py`。项目以 RGB 为参考模态，对 Event、IR、Depth、Audio 等目标模态估计时间偏移。

主要方法包括：

- RGB 与 IR/Depth：使用活动强度或跨相关估计短时间偏移。
- RGB 与 Event：支持 DTW、光流、特征匹配等方法。
- RGB 与 Audio：通过音频能量曲线做 cross-correlation。
- 输出 JSON 记录 offset、duration、fps、质量指标等。
- 输出 plot 用于人工检查对齐是否合理。

相关产物：

```text
temporal_alignment_json/
temporal_alignment_plots/
aligned_dataset/
```

任务片段切分逻辑主要在：

```text
annotation_feature/task_slicing.py
annotation_feature/segmented_pipeline.py
```

切分后的任务片段会被导出到 `aligned_dataset/<sample>_split/SegN/` 这类目录中。后续 QA 和 VLM benchmark 主要基于这些 aligned segment 运行。

## 6. 单模态 QA 生成

各模态 QA 生成流程在 `annotation_feature/pipeline/modalities/` 下。

| 模态 | 主要文件 | 说明 |
|---|---|---|
| RGB | `modalities/rgb/pipeline.py` | 使用 night/day RGB frames 生成 caption、question、answer |
| IR | `modalities/ir/pipeline.py` | 面向红外/低光可见性生成 QA |
| Event | `modalities/event/pipeline.py` | 面向运动、事件和动态变化生成 QA |
| Depth | `modalities/depth/pipeline.py` | 面向空间、距离、导航和布局生成 QA |
| Marigold Depth | `modalities/marigold/pipeline.py` | 先由 RGB 估计深度，再基于深度帧生成 QA |
| Audio | `modalities/audio/pipeline.py` | 先生成 HIA，再生成带时间戳的 audio-visual caption，最后生成声学 QA |

Prompt 定义在：

```text
prompts/rgb_prompts.py
prompts/ir_prompts.py
prompts/event_prompts.py
prompts/depth_prompts.py
prompts/audio_prompts.py
```

视觉模态的典型输出结构是：

```json
{
  "pair_key": {
    "night_file": "...",
    "day_file": "...",
    "annotations": {
      "object_recognition": {
        "caption": "...",
        "question": "...",
        "answer": "..."
      }
    }
  }
}
```

Audio 的输出结构略有不同，会包含：

- `audio_hia`：人类交互描述。
- `audio_chronological_caption`：带时间戳的音频/视频描述。
- `categories`：声源识别、声音事件、环境理解等 QA 类别。

常见输出文件包括：

```text
rgb_qa_results.json
ir_qa_results.json
event_qa_results.json
depth_qa_results.json
marigold_depth_qa_results.json
audio_qa_results.json
```

aligned 版本位于：

```text
qa_pairs/aligned/rgb_qa_results_aligned.json
qa_pairs/aligned/ir_qa_results_aligned.json
qa_pairs/aligned/event_qa_results_aligned.json
qa_pairs/aligned/audio_qa_results_aligned.json
qa_pairs/aligned/marigold_depth_qa_results_aligned.json
```

## 7. 多模态、证据归一化与融合

单模态 QA 生成后，项目会把不同模态的 caption/question/answer 归一化为统一证据单元，便于分组、融合和构建跨模态 QA。

相关模块：

```text
annotation_feature/reasoning/normalize_evidence_units.py
annotation_feature/reasoning/normalizer.py
annotation_feature/reasoning/group_evidence.py
annotation_feature/reasoning/export_grouped_qa.py
annotation_feature/fusion.py
annotation_feature/multimodal_qa_pipeline.py
annotation_feature/multimodal_qa_verifier.py
```

主要产物：

```text
normalized_evidence_units.json
grouped_evidence.json
grouped_qa_pairs.json
fused_qa_results.json
fusion_qa_stats.json
fusion_diagnostics.json
outputs/implicit_multimodal_qa_candidates_gemini_v2.json
outputs/implicit_multimodal_qa_verified_gemini_v2.json
```

其中 `annotation_feature/multimodal_qa_pipeline.py` 用于生成 implicit cross-modal QA。它会基于不同模态组合构造问题，例如：

- RGB + Audio：用视觉上下文和声音证据互相验证。
- RGB + Event：用可见光上下文和运动事件互补。
- RGB + Depth：结合外观和空间关系。
- RGB + IR：比较可见光和低光红外能力。
- Event + Audio / Event + Depth / Event + IR：强调运动、声音、空间和夜间可见性的互补。

## 8. QA 质量评估与清洗

QA 质量控制代码在 `annotation_feature/qa_quality/`。

主要阶段：

1. `aligned_evaluator.py`：规则检查 aligned QA，发现空字段、prompt 泄漏、模态不匹配、过短答案、多问题/多答案等问题。
2. `llm_evaluator.py`：使用 LLM 对 QA 质量做进一步判断，可按模态均衡抽样。
3. `cleaner.py`：根据质量评估结果生成正式 benchmark 使用的 valid QA。
4. `benchmark.py`：既包含 caption-only benchmark，也包含 frame-input 和 video-input answer benchmark 的公共能力。
5. `answer_metrics.py`、`answer_judge.py`：对模型答案计算确定性指标和 LLM-as-a-judge 指标。

重要中间产物：

```text
outputs/aligned_qa_quality_report.json
outputs/aligned_qa_quality_items.csv
outputs/aligned_qa_llm_eval_results.json
outputs/aligned_qa_llm_eval_items.csv
outputs/aligned_qa_cleaned_items.json
outputs/aligned_qa_valid_items.json
```

`outputs/aligned_qa_valid_items.json` 是后续 VLM benchmark 默认读取的 QA 数据源。

## 9. VLM 模型评估

项目支持多种 VLM 评估方式。

### 9.1 Caption-only QA benchmark

Caption-only benchmark 只给模型 caption 和 question，不给原始帧或视频。入口在：

```text
annotation_feature/qa_quality/benchmark.py
```

默认输入：

```text
outputs/aligned_qa_valid_items.json
```

默认输出：

```text
outputs/benchmarks/
```

它支持 Gemini、OpenAI、Qwen 文本/视觉模型等 provider，并使用 judge model 对答案打分。

### 9.2 固定帧输入 benchmark

固定帧输入 benchmark 会为每个 QA item 提供相同数量、相同顺序的缓存帧，确保不同模型看到完全一致的视觉证据。

关键脚本：

```text
scripts/run_vlm_4b_aligned_frame_benchmark.py
scripts/run_vlm_4b_aligned_frame_benchmark_cu130.py
scripts/run_vlm_8frame_smoke.py
scripts/run_vlm_cross_modality_frame_benchmark.py
```

现有协议文档：

```text
docs/vlm_8frame_benchmark_protocol.md
```

8-frame benchmark 的固定规则包括：

- 每个 QA item 输入 8 张图。
- day/with-light 4 张，night/no-light 4 张。
- 输入顺序固定为 day frames 后接 night frames。
- 如果问题中引用了特定 frame number，优先选择该帧或最近缓存帧。
- 其余帧使用确定性均匀采样。
- 生成参数通常为 `max_new_tokens=128`、`do_sample=False`。

### 9.3 Cross-modality frame benchmark

跨模态 frame benchmark 会固定原始问题和标准答案，只替换输入视觉模态，从而观察模型在“问题来自某模态，但输入证据换成另一模态”时的表现。

入口脚本：

```text
scripts/run_vlm_cross_modality_frame_benchmark.py
```

默认输出：

```text
outputs/benchmarks/vlm_cross_modality_8frame/
```

后续比较脚本：

```text
scripts/build_cross_modality_comparison.py
scripts/build_metric_modality_tables.py
scripts/plot_modality_cluster.py
```

相关报告通常位于：

```text
outputs/evaluations/
```

### 9.4 Native video benchmark

Native video benchmark 会把视频片段本身作为模型输入，而不是固定抽帧。它用于评估模型原生视频理解能力。

入口脚本：

```text
scripts/run_vlm_native_video_smoke.py
scripts/run_vlm_cross_modality_video_benchmark.py
```

协议文档：

```text
docs/vlm_native_video_benchmark_protocol.md
```

默认输出示例：

```text
outputs/benchmarks/vlm_native_video/
outputs/benchmarks/vlm_cross_modality_native_video_4b/
```

### 9.5 答案评估指标

模型答案评估脚本是：

```text
scripts/evaluate_vlm_answers.py
```

评估协议文档是：

```text
docs/vlm_answer_evaluation.md
```

支持的指标包括：

- normalized exact match
- token precision/recall/F1
- BLEU-4
- ROUGE-L
- METEOR
- ANLS 和 character F1
- yes/no boolean accuracy
- counting numeric accuracy
- set F1
- sequence order score
- task-aware score
- repetition 和 conciseness diagnostics
- LLM-as-a-judge strict/soft accuracy
- latency、throughput、GPU memory
- modality/section macro average
- pairwise disagreement 和 McNemar test

常用命令示例：

```bash
python scripts/evaluate_vlm_answers.py \
  --input outputs/benchmarks/vlm_8frame_aligned_4b \
  --output outputs/evaluations/vlm_8frame_aligned_4b \
  --metrics deterministic,llm_judge \
  --judge-model gemini-3.1-flash-lite
```

## 10. 常用入口

交互式入口：

```bash
python main.py
```

`main.py` 注册了 aligned RGB、IR、Event、Audio、Marigold Depth 生成、修复、QA 质量检查、多模态 QA、时间对齐、图表生成等操作，适合本地逐步运行。

核心程序化入口：

```text
annotation_feature/pipeline/main.py
```

这里包含：

- `run_event`
- `run_depth`
- `run_marigold_depth_qa`
- `run_ir`
- `run_audio`
- 各模态 missing-section repair
- resume/filter/write results 等通用逻辑

## 11. 推荐阅读顺序

新同学建议按以下顺序读代码：

1. `docs/project_overview_cn.md`：先建立全局地图。
2. `docs/caption_entity_atom_semantics_cn.md`：理解 caption schema 中 Entity、Atom 和 evidence refs 的语义边界。
3. `main.py`：看项目有哪些可运行动作。
4. `annotation_feature/pipeline/main.py`：理解各模态 QA 生成如何串起来。
5. `annotation_feature/video_preprocessor.py`：理解帧缓存。
6. `annotation_feature/temporal_alignment.py`：理解对齐和 aligned dataset。
7. `annotation_feature/pipeline/modalities/rgb/pipeline.py`：先看一个视觉模态的完整 QA 生成模板。
8. `annotation_feature/pipeline/modalities/audio/pipeline.py`：看音频 HIA、timestamped caption 和 QA cascade。
9. `annotation_feature/qa_quality/aligned_evaluator.py`：看 QA 质量规则。
10. `annotation_feature/qa_quality/benchmark.py`：看 benchmark adapter、judge 和结果格式。
11. `scripts/run_vlm_cross_modality_frame_benchmark.py` 和 `scripts/run_vlm_cross_modality_video_benchmark.py`：看正式实验如何扩展和运行。

## 12. 典型产物对照表

| 阶段 | 主要输入 | 主要输出 |
|---|---|---|
| 原始采集 | 多模态 day/night 视频 | 原始 dataset |
| 抽帧缓存 | dataset videos | `.frames_cache*` |
| 时间对齐 | RGB + target modality videos/audio | `temporal_alignment_json/`、`temporal_alignment_plots/` |
| 任务切分 | 原始/对齐视频 | `aligned_dataset/<sample>_split/SegN/` |
| 单模态 QA | 缓存帧或音频视频 | `*_qa_results.json`、`qa_pairs/aligned/*_aligned.json` |
| 证据归一化 | 各模态 QA | `normalized_evidence_units.json` |
| 证据分组/融合 | normalized evidence | `grouped_evidence.json`、`fused_qa_results.json` |
| QA 质量控制 | aligned QA | `outputs/aligned_qa_valid_items.json` |
| VLM benchmark | valid QA + frames/videos | `outputs/benchmarks/...` |
| 答案评估 | benchmark answer JSON | `outputs/evaluations/...` |
| 报告图表 | evaluation CSV/JSON | `outputs/presentation/`、`outputs/figures/` |

## 13. 开发注意事项

- 多数 Gemini 调用需要 `.env` 或 API key list 配置，具体由 `annotation_feature/pipeline/client.py` 和 benchmark 脚本读取。
- 大模型本地推理依赖 GPU、CUDA、transformers、vLLM、bitsandbytes 等环境；可先跑 smoke 或小样本。
- benchmark 结果要保留 manifest hash、模型路径、frame count、generation config，避免不同实验混在一起。
- 不要随意修改 fixed manifest；如需修改，应视为新实验并单独命名输出目录。
- QA 数据质量会直接影响 VLM 评估结论，正式实验应使用 `outputs/aligned_qa_valid_items.json` 或经过同等质量控制的数据。
- Audio 和 native video benchmark 通常更依赖模型/SDK 对视频或音频输入的原生支持，环境差异可能影响可复现性。

