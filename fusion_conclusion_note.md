# Late Fusion Design Note

## 1. Goal

The late fusion module is designed to construct a reliable multimodal QA ground truth set from modality-specific QA results. Its main purpose is not to generate a video summary. Instead, it aims to select, verify, and organize QA pairs that can be used to evaluate how well different models understand video inputs from one or more modalities.

In this project, a model may be given RGB video, infrared video, event data, depth video, audio, or a combination of these inputs. The fused QA set provides the ground truth questions and answers used to evaluate the model's understanding. Therefore, the quality and reliability of the QA set are more important than producing a short or balanced summary.

The current fusion design follows one key principle:

```text
category controls reliability;
section controls evaluation grouping.
```

This means that the original QA category determines how the QA is scored and filtered, while section labels describe what capability the QA is intended to evaluate.

## 2. Inputs and QA Evidence

The fusion module reads QA outputs from different modalities, including RGB, infrared, event, audio, and depth results. Each modality-specific result file contains QA entries grouped under category names such as `object_recognition`, `spatial_reasoning`, `dynamic_recognition`, `counting`, `text_recognition`, or `audio_hia`.

These category names are important. During fusion, the original category becomes the QA item's `annotation_key` and `category`. The downstream reliability rules depend on this category. As a result, the upstream modality-specific QA files play a decisive role in how each QA item is interpreted.

The fusion process converts each modality-specific entry into QA evidence. Each evidence item stores the source modality, original category, caption, question, answer, normalized tokens, and later reliability metadata.

If a QA field contains multiple numbered questions and answers in one block, the fusion module attempts to split the block into fine-grained QA evidence. This is important because reliability should be evaluated at the individual QA level. Treating a block of many QA pairs as one item would make scoring too coarse and could hide weak or incorrect QA pairs inside an otherwise useful block.

Caption fallback QA can still be used when a modality provides useful caption evidence but no explicit QA pair. However, fused text sections such as `cross_modal_details` and `final_unified_caption` should not be converted back into fallback QA evidence, because they are already fusion outputs rather than original modality evidence.

## 3. Category-Driven Reliability

The current reliability design is category-driven. Each QA category is associated with a reliability profile. The profile defines how the QA should be scored, what kind of modality evidence is trusted, whether cross-modal support is required, and whether lexical support is important.

The main reliability components are:

- `fusion_score`: an overall score for the QA evidence.
- `support_score`: a cross-modality support score.
- `gate`: the category-level filtering rule.
- `review_reasons`: reasons why a QA should be reviewed instead of directly selected.

The `fusion_score` estimates how reliable and useful a QA item is. It considers the source modality, the reliability of that modality, whether the QA has a valid question and answer, whether it was split into a fine-grained QA item, and whether it has support from other modalities.

The source modality is weighted according to the QA category. For example, text-related QA should trust RGB and infrared more than audio. Audio-related QA should trust audio more than visual modalities. Motion-related QA can give stronger weight to event data. This avoids using section labels as scoring rules.

The `support_score` measures whether other modalities provide consistent evidence for the same QA. It combines lexical support and semantic support. Lexical support is based on token overlap, while semantic support is based on sentence embedding similarity. Different categories combine these two signals differently. For example, counting and text recognition are more sensitive to lexical overlap, while action, temporal, and audio categories can benefit more from semantic similarity.

## 4. Category Gates

The fusion module uses category gates to decide whether a QA item should be selected, reviewed, or dropped. The main gate types are:

- `support_required`
- `support_soft`
- `support_exempt`
- `review_recommended`

`support_required` is used for categories where cross-modality evidence is important, such as object recognition, spatial reasoning, navigation, and counting. If support is too weak, the QA is likely to be dropped or sent to review.

`support_soft` is used for categories where cross-modality support is useful but not always mandatory. This applies to action, motion, dynamic recognition, and temporal sequence QA. A high-quality single-modality QA can still pass if it is otherwise reliable.

`support_exempt` is used for categories that may naturally rely on a single modality. Examples include text recognition, audio understanding, and lighting. For instance, some text may only be visible in RGB or infrared, and audio events may only be available in the audio stream.

`review_recommended` is used for categories such as non-common or anomaly-related QA. These QA items are not necessarily wrong, but they are more likely to require manual or LLM-based verification before being treated as final ground truth.

## 5. Review Queue

The review queue is designed for uncertain QA items. A QA item in the review queue is not automatically considered wrong. Instead, it means the automatic rules found a reason to be cautious.

A QA item may be sent to review when:

- the answer is too long or contains multiple clauses;
- the question is too long;
- the original numbered QA block was not cleanly split;
- semantic support is high but lexical support is very low;
- the category requires lexical support but lexical overlap is weak;
- a support-required category has no clear supporting modality;
- possible numeric, directional, or negation conflicts are detected.

This separation is important. Dropped QA items fail the automatic reliability rules. Review QA items may still be useful, but they should be inspected before being used as final benchmark ground truth.

## 6. Section Labels

Section labels are used for evaluation grouping, not reliability filtering.

Each QA item can belong to one or more sections. The mapping is derived from the QA category. For example:

```text
text_recognition -> text_and_symbols, objects_and_attributes
dynamic_counting -> motion_and_action, counting
navigation -> spatial_and_layout, motion_and_action
audio_chronological_caption -> audio_understanding, temporal_sequence
```

The output keeps three related fields:

- `section`: the primary section, kept for compatibility.
- `primary_section`: the main section label.
- `sections`: the full list of section labels.

This multi-section design is useful because a QA item can test more than one capability. For example, a dynamic counting question can test both motion understanding and counting. A text recognition question can test both text understanding and object-level recognition.

The important point is that section labels do not decide whether a QA is reliable. They are used after reliability filtering to organize QA items by evaluation dimension.

## 7. Output Interpretation

The main output file is `fused_qa_results.json`. It contains:

- `selected_reliable_qas`: QA items that passed the automatic reliability rules.
- `review_recommended_qas`: QA items that may be useful but need inspection.
- `qas_by_section`: selected QA grouped by section labels.
- `review_qas_by_section`: review QA grouped by section labels.

The diagnostics file, `fusion_diagnostics.json`, records the detailed filtering process, including candidate counts, selected counts, review counts, dropped counts, drop reasons, review reasons, and supporting modality information.

The analysis files, `fusion_qa_stats.json` and `fusion_qa_rows.csv`, provide aggregated and flattened views of the fusion results. These files are useful for checking QA counts by category, modality, section, gate, and review reason.

## 8. Assumptions and Limitations

The current fusion pipeline assumes that upstream modality-specific QA categories are mostly correct. Since both reliability profiles and section labels are derived from the category, incorrect upstream category assignment can affect both scoring and evaluation grouping.

The automatic reliability rules are useful but not perfect. Lexical overlap is simple and does not consider word order. Semantic embedding support is more flexible, but it is still not equivalent to human or LLM judgment. Some valid QA items may be placed in the review queue, and some weak QA items may still pass the automatic rules.

For higher-quality benchmark construction, the current automatic fusion pipeline can be combined with a manual or LLM-based review stage. The review queue is designed to support this future step.

## 9. Summary

The current late fusion process can be summarized as:

```text
modality QA files
-> fine-grained QA evidence
-> category-driven reliability profile
-> fusion score and support score
-> selected / review / dropped decision
-> section labels for evaluation grouping
```

The main design distinction is:

```text
category = reliability behavior
section = evaluation label
```

This design better matches the project goal of constructing reliable QA ground truth for multimodal video understanding evaluation.
