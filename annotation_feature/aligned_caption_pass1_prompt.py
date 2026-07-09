"""Prompt, schema example, and template caption builders for Pass 1 (evidence construction)."""

from __future__ import annotations

import json
from typing import Any

from annotation_feature.aligned_caption_schema import (
    ALLOWED_MISSING_ATTRIBUTE_TYPES,
    _enum_line,
    build_modality_constraint_block,
)

def _build_prompt_schema_example(task: Any) -> dict[str, Any]:
    fk = task.composite_frames[0].stem if task.composite_frames else "frame_000000"
    return {
        "global_scene": {
            "scene_summary": "A stationary vehicle is situated near a concrete barrier.",
            "environment": "urban",
            "temporal_progression": "The spatial relationship between the vehicle and barrier is maintained.",
            "physical_entities": [
                {
                    "entity_id": "entity_001",
                    "category": "vehicle",
                    "referential_scope": "stationary vehicle"
                },
                {
                    "entity_id": "entity_002",
                    "category": "barrier",
                    "referential_scope": "concrete barrier"
                }
            ]
        },
        "video1_analysis": {
            "modality": task.modality1,
            "detailed_caption": "A stationary vehicle rests beside a concrete barrier.",
            "information_atoms": [
                {
                    "atom_id": "v1_atom_001",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_001", "entity_002"],
                    "fact": "A stationary vehicle rests beside a concrete barrier."
                }
            ],
            "sensor_specific_cues": ["High contrast shadow line marks the barrier edge."],
            "sensor_limitations": ["Low ambient light obscures the surface details of the vehicle."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Partial outline of the vehicle.",
                    "missing_evidence": "Internal structural details.",
                    "hypotheses": [
                        {"hypothesis": "Small passenger car.", "confidence": "low"},
                        {"hypothesis": "Delivery van.", "confidence": "low"}
                    ]
                }
            ],
            "missing_key_attributes": [
                {
                    "attribute_type": "material",
                    "missing_attribute": "Material of entity_001.",
                    "why_missing": "Surface texture is not visible in this segment's lighting.",
                    "recoverable_evidence_refs": []
                }
            ]
        },
        "video2_analysis": {
            "modality": task.modality2,
            "detailed_caption": "A concrete barrier is visible.",
            "information_atoms": [
                {
                    "atom_id": "v2_atom_001",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_002"],
                    "fact": "A concrete barrier is visible."
                }
            ],
            "sensor_specific_cues": [],
            "sensor_limitations": [],
            "uncertain_observations": [],
            "missing_key_attributes": []
        }
    }


def _build_pass1_prompt(task: Any) -> str:
    frame_names = ", ".join(path.name for path in task.composite_frames)
    constraint_block = build_modality_constraint_block(task.modality1, task.modality2)
    schema_example_text = json.dumps(_build_prompt_schema_example(task), indent=2, ensure_ascii=False)

    return "\n".join(
        [
            "TASK: Construct PASS 1 evidence for an aligned multimodal video segment. Neither source is ground truth.",
            "SCOPE: Produce ONLY global_scene, video1_analysis, and video2_analysis. NO cross-modal fusion, information-gain, directional contribution, cross-source ambiguity resolution, QA, or recoverability reasoning.",
            "PRIORITIES: 1. source-local factual correctness; 2. valid provenance/references; 3. stable entity identity; 4. atom minimality; 5. evidence coverage; 6. caption completeness; 7. descriptive richness.",
            constraint_block,
            "RULES:",
            "1. SHARED ENTITIES: Reuse `entity_id` for the same physical referent. Never reuse an ID for different referents. Create entities for salient independent physical referents. Preserve minimality; do not hide genuinely salient referents.",
            "2. REGISTRY METADATA: Global registry metadata is organizational, not source-local evidence. Use conservative but useful identity descriptions. Do not automatically over-neutralize every referent to 'object' or 'unknown', but do not insert source-exclusive details that create leakage risk.",
            "3. ISOLATION & LEAKAGE: Cross-modal leakage occurs ONLY when a claim is not independently supported by the current source. For each source, use the strongest description independently justified by that source: no stronger, but also no weaker merely because another source supports the same interpretation. SOURCE-LOCAL TEST: Judge each claim using only the current source. Hide the opposite source and treat shared registry metadata as non-evidence. If the claim remains reasonably supported, it is allowed.",
            "4. ATOMS FIRST: Atoms are the source-local evidence inventory. Build `detailed_caption` from the completed same-source atom set; never invent caption content first. 1 atom = 1 minimal physical proposition. Unique `atom_id` (v1_atom_/v2_atom_ prefix), valid `frame_keys`, non-empty `entity_refs`. Primarily state source-local physical observations; avoid mechanism-oriented phrasing when an equivalent physical observation is available.",
            "5. CAPTION CLOSURE: Build each `detailed_caption` only from same-source atom facts. It may reorder, combine, or paraphrase them, but must not add motion, temporal progression, quantity, subtype, causal interpretation, or stronger semantics absent from the atoms. Temporal words (initially, later, then, moving forward, passing) require explicit same-source atom support.",
            "6. LENGTH: Meet minimum lengths using supported content only. Target 24-40 words for scene_summary and 35-60 words for each detailed_caption. Never add unsupported facts as filler.",
            "7. PHYSICAL WORDING: `detailed_caption`, `global_scene`, and `information_atoms` must describe physical-world observations. Do not use 'the sensor records', 'generates activations', or generic explanations. Forbidden: modality, rgb, ir, event, depth, edge map, blurry, noisy, pixel, grayscale.",
            "8. SENSORS & UNCERTAINTY: `sensor_specific_cues` may describe response patterns but must remain segment-specific; do not use a cue as evidence for an unsupported semantic interpretation. `sensor_limitations` must describe a limitation visibly manifested in this segment, not a generic property of the sensor or modality. Uncertain hypotheses must arise from current-source evidence.",
            "9. MISSING ATTRIBUTES: A missing-attribute description must never reveal the missing value. `why_missing` must cite segment-local absence of evidence, not generic sensor or modality capability.",
            "10. `recoverable_evidence_refs` MUST remain [] in PASS 1.",
            "OUTPUT CONTRACT: Return ONLY valid JSON matching this skeleton. No other top-level fields.",
            schema_example_text,
            "ALLOWED ENUMS:",
            "- confidence: low, medium, high",
            _enum_line("attribute_type / missing_attribute_type", ALLOWED_MISSING_ATTRIBUTE_TYPES),
            f"Segment: {task.segment_id}; side: {task.side}.",
            f"Composite frames ({len(task.composite_frames)} images): {frame_names}",
        ]
    )


def _template_caption_pass1(task: Any) -> dict[str, Any]:
    target_entity = "unresolved_target"
    return {
        "global_scene": {
            "scene_summary": "Template mode placeholder for scene summary. " * 5,
            "physical_entities": [
                {
                    "entity_id": target_entity,
                    "category": "unknown",
                    "referential_scope": "the unresolved target entity used by template mode",
                    "evidence_profile": {
                        "identity_evidence": ["Placeholder evidence"],
                        "observable_attributes": ["Placeholder attribute"],
                        "spatial_context": ["Placeholder context"]
                    }
                }
            ],
            "environment": "unknown",
            "temporal_progression": "Template mode placeholder for temporal progression. " * 3,
        },
        "video1_analysis": {
            "modality": task.modality1,
            "detailed_caption": "Template mode placeholder for detailed caption. " * 6,
            "information_atoms": [
                {
                    "atom_id": "v1_atom_001",
                    "frame_keys": [task.composite_frames[0].stem] if task.composite_frames else [],
                    "entity_refs": [target_entity],
                    "fact": "Template placeholder fact"
                }
            ],
            "sensor_specific_cues": ["Template mode placeholder."],
            "sensor_limitations": ["Template mode placeholder."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Template placeholder",
                    "hypotheses": [{"hypothesis": "Placeholder 1", "confidence": "low"}, {"hypothesis": "Placeholder 2", "confidence": "low"}],
                    "missing_evidence": "Placeholder"
                }
            ],
            "missing_key_attributes": [
                {
                    "attribute_type": "existence",
                    "missing_attribute": "Template placeholder",
                    "why_missing": "Template placeholder",
                    "recoverable_evidence_refs": []
                }
            ],
        },
        "video2_analysis": {
            "modality": task.modality2,
            "detailed_caption": "Template mode placeholder for detailed caption. " * 6,
            "information_atoms": [
                {
                    "atom_id": "v2_atom_001",
                    "frame_keys": [task.composite_frames[0].stem] if task.composite_frames else [],
                    "entity_refs": [target_entity],
                    "fact": "Template placeholder fact"
                }
            ],
            "sensor_specific_cues": ["Template mode placeholder."],
            "sensor_limitations": ["Template mode placeholder."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Template placeholder",
                    "hypotheses": [{"hypothesis": "Placeholder 1", "confidence": "low"}, {"hypothesis": "Placeholder 2", "confidence": "low"}],
                    "missing_evidence": "Placeholder"
                }
            ],
            "missing_key_attributes": [
                {
                    "attribute_type": "existence",
                    "missing_attribute": "Template placeholder",
                    "why_missing": "Template placeholder",
                    "recoverable_evidence_refs": []
                }
            ],
        }
    }
