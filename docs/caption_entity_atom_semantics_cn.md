# Caption Schema 中 Entity 与 Atom 的语义定义

本文档定义 cross-modal disambiguation caption schema 中 `Entity`、`Atom` 以及二者关系的语义边界。它的目的不是描述某一次具体输出，而是作为后续 prompt、schema validation、QA 生成和 evidence graph 设计的共同约定。

## 1. 核心结论

当前 schema 应采用下面的概念模型：

```text
Physical world
  -> stable Entity scopes
  -> modality-local observation Atoms
  -> frame grounding
```

更具体地说：

```text
Entity(s)
   ^
   | entity_refs
   |
 Atom
   |
   | frame_keys
   v
Frame(s)
```

`Entity` 控制稳定语义范围，`Atom` 控制具体观察粒度。不要把 `Atom` 理解成更小的 `Entity`，也不要为了描述 group 内部的某个具体成员而不断创建 subset entity。

## 2. Entity 的定义

英文定义：

> A modality-independent, stable world-level referential scope representing an object, object group, region, or phenomenon that remains semantically consistent across the caption.

中文定义：

> Entity 是跨模态共享的、稳定的世界级指代范围，可以表示对象、对象组、区域或现象，并且在整个 caption 中保持身份和粒度不变。

Entity 回答的问题是：

```text
我们正在讨论现实世界中的哪一个稳定对象、对象组、区域或现象？
```

例如：

```text
entity_001 = rider / bicycle-rider system
entity_002 = parked vehicles group
entity_003 = drainage grate
entity_004 = tree-shadow phenomenon
```

Entity 不属于某个单独模态，不属于某个 frame，也不描述某一条具体事实。它只是一个稳定的 world-level referential scope。

因此，一旦定义：

```text
entity_002 = parked vehicles group
```

在整个 caption 中它都应保持这个粒度。它不应该后来变成：

```text
entity_002 = white BMW
```

也不应因为需要讨论这组车辆里的某一辆 BMW 而新增：

```text
entity_005 = white BMW
```

否则会形成：

```text
entity_005 subset-of entity_002
```

这会造成 entity scope 重叠，并让 downstream 无法判断两个 entity 是并列对象、父子对象，还是同一对象的不同粒度。

## 3. Atom 的定义

英文定义：

> A minimal, modality-local, frame-grounded observation proposition that states one directly supportable fact about one or more entities.

中文定义：

> Atom 是单模态局部的、由具体 frame 支撑的最小观察命题，用来表达关于一个或多个 Entity 的一条可直接支持的事实。

Atom 回答的问题是：

```text
在某个视频模态中，哪些 supplied frames 直接支持了哪一条最小观察事实？
```

Atom 的本质是事实，不是对象。例如下面这些都可以是 atoms：

- 存在事实：`A white sedan is parked on the right side.`
- 属性事实：`The parked sedan has a bright white body.`
- 空间事实：`The parked vehicles line the right side of the road.`
- 状态事实：`The drainage grate remains stationary near the curb.`
- 运动事实：`The rider moves forward past the parked vehicles.`
- 关系事实：`The rider passes the parked vehicles on the right.`
- 视觉结构事实：`Branching shadow patterns extend across the asphalt.`

因此，atom 不应是：

```text
BMW
tree shadow
rider
```

而应是：

```text
关于 BMW 的一条最小观察事实
关于 tree shadow 的一条最小观察事实
关于 rider 的一条最小观察事实
```

## 4. Group 与 Specific Member 的建模规则

如果一个 entity 已经表示对象组，例如：

```text
entity_002 = parked vehicles group
```

后续需要讨论组内的具体成员时，默认应创建更具体的 atom，而不是创建 subset entity。

推荐：

```json
{
  "atom_id": "v1_atom_004",
  "frame_keys": ["frame_000480"],
  "fact": "A white BMW sedan is parked nearest the foreground on the right.",
  "entity_refs": ["entity_002"]
}
```

```json
{
  "atom_id": "v1_atom_005",
  "frame_keys": ["frame_000510"],
  "fact": "A black Mercedes is parked farther along the same row.",
  "entity_refs": ["entity_002"]
}
```

不推荐：

```text
entity_002 = parked vehicles group
entity_005 = white BMW sedan
entity_006 = black Mercedes
```

这种做法会把 entity registry 变成任意粒度对象清单，削弱 entity 作为稳定 reasoning scope 的作用。

## 5. Entity 与 Atom 不是严格树结构

概念上可以说：

```text
Entity
  -> multiple observation atoms
```

但数据结构上不应实现成严格树，因为一个 atom 可以同时关于多个 entities。

例如：

```json
{
  "atom_id": "v1_atom_010",
  "frame_keys": ["frame_000510", "frame_000630"],
  "fact": "The rider passes the parked vehicles on the right side.",
  "entity_refs": ["entity_001", "entity_002"]
}
```

这里 `v1_atom_010` 同时关于：

```text
entity_001 = rider / bicycle-rider system
entity_002 = parked vehicles group
```

所以真实结构是 many-to-many graph：

```text
Entity_001 <---- Atom_010 ----> Entity_002
                    |
                    v
             frame_000510
             frame_000630
```

## 6. entity_refs 的语义

`entity_refs` 不表示：

```text
atom.fact 描述的对象与 entity 完全相同
```

它应表示：

```text
这个 atom 提供了关于哪些 Entity scope 的观察证据。
```

例如：

```json
{
  "atom_id": "v1_atom_004",
  "frame_keys": ["frame_000480"],
  "fact": "A white BMW sedan is parked on the right side of the street.",
  "entity_refs": ["entity_002"]
}
```

即使：

```text
entity_002 = parked vehicles group
```

这个引用仍然合理，因为 atom 描述的是该 group 内部的具体可观察内容。这里不能把 `entity_refs` 解释成：

```text
white BMW sedan == parked vehicles group
```

而应解释成：

```text
white BMW sedan 的观察事实是关于 parked vehicles group 这个稳定范围的证据。
```

## 7. Atom 最小性边界

Atom 应尽量表达一条可独立引用的观察命题。判断是否应拆分时，可以使用下面的规则：

- 如果两个事实可能被不同 downstream claim 单独引用，应拆成两个 atoms。
- 如果两个事实对应不同 entity_refs，通常应拆分，除非它们本身就是一个关系事实。
- 如果一个事实是另一个事实的推理结果，应把直接观察放进 atom，把推理放进 reasoning_events 或 ambiguity_events。
- 如果一句话同时包含对象存在、属性、空间关系和运动变化，通常太粗，应拆成更小 atoms。
- 如果一个事实必须由多个 frames 共同支持，可以保留为一个 atom，并在 `frame_keys` 中列出所有支持帧。
- 如果一个事实跨越多个 entities 但本身表达的是二者关系，可以保留为一个 atom，并引用多个 entity_refs。

示例：

```text
The rider passes the parked vehicles on the right side.
```

这是一个关系事实，可以作为一个 atom，并引用 rider 与 parked vehicles group。

但下面这句通常太粗：

```text
The rider passes the parked vehicles while a white BMW and a black Mercedes are visible along the curb.
```

它混合了 rider-vehicle 关系事实和具体车辆存在事实，更适合拆分。

## 8. 与下游字段的关系

`cross_modal_evidence_links`、`information_gain`、`reasoning_events`、`ambiguity_events` 和 `qa_relevant_details` 不应直接跳过 atoms 去引用 frames。它们应引用 atom IDs，并要求 atom 的 `fact` 本身直接支持对应 claim。

也就是说，证据链应是：

```text
downstream claim
  -> supporting atom refs
  -> atom.fact
  -> atom.frame_keys
```

而不是：

```text
downstream claim
  -> frame contains relevant visual content
```

如果 downstream claim 需要某个具体细节，而现有 atoms 没有显式表达这个细节，应先新增一个最小 atom，再引用它。

## 9. 当前实现状态

截至本文档创建时，`annotation_feature/aligned_multimodal_caption_pipeline.py` 中的 prompt 已经要求 information atoms 表达 directly observable、source-local、minimal factual claim，并要求 downstream claim 通过 atom refs 获得支持。

不过当前 schema example 中的 `information_atoms` 还没有显式 `entity_refs` 字段。本文档把 `entity_refs` 的目标语义先固定下来，供后续 schema/prompt/validator 演进时使用。
