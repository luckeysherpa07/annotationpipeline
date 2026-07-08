"""Prompt, schema example, and template caption builders."""

from __future__ import annotations

import json
from typing import Any

from annotation_feature.aligned_caption_schema import (
    ALLOWED_AMBIGUITY_DIRECTIONS,
    ALLOWED_CROSS_MODAL_CONTRIBUTION_DIRECTIONS,
    ALLOWED_GAIN_RATINGS,
    ALLOWED_GAIN_TYPES,
    ALLOWED_MISSING_ATTRIBUTE_TYPES,
    ALLOWED_QA_REASONING_PATTERNS,
    ALLOWED_REASONING_EVENT_TYPES,
    _enum_line,
    build_modality_constraint_block,
)

def _build_prompt_schema_example(task: Any) -> dict[str, Any]:
    frame_key = task.composite_frames[0].stem if task.composite_frames else "frame_000000"
    later_frame_key = task.composite_frames[min(1, len(task.composite_frames) - 1)].stem if task.composite_frames else frame_key
    return {
        "global_scene": {
            "scene_summary": "This is a fully compliant scene summary paragraph that describes the physical environment and ongoing actions in sufficient detail to exceed the minimum word count requirement of twenty words.",
            "environment": "urban",
            "temporal_progression": "The scene unfolds chronologically, demonstrating clear progression of events from start to finish without referencing any sensor or image quality artifacts.",
            "physical_entities": [
                {
                    "entity_id": "entity_001",
                    "category": "rider",
                    "referential_scope": "the bicycle rider tracked across the sampled interval",
                    "evidence_profile": {
                        "identity_evidence": ["Rider and bicycle form a persistent moving referent."],
                        "observable_attributes": ["The rider moves along the road."],
                        "spatial_context": ["The rider is near the curbside vehicle area."]
                    }
                },
                {
                    "entity_id": "entity_002",
                    "category": "vehicle",
                    "referential_scope": "the specific white BMW tracked across the sampled interval"
                },
                {
                    "entity_id": "entity_003",
                    "category": "parked_vehicle_group",
                    "referential_scope": "the other parked vehicles along the curb, excluding entity_002"
                }
            ]
        },
        "video1_analysis": {
            "modality": task.modality1,
            "detailed_caption": "The white BMW is stationary near the right curb. The same white BMW begins moving forward. The rider passes behind the turning white BMW. Other parked vehicles remain along the curbside area throughout the sampled interval.",
            "information_atoms": [
                {
                    "atom_id": "v1_atom_001",
                    "frame_keys": [frame_key],
                    "entity_refs": ["entity_002"],
                    "fact": "The white BMW is stationary near the right curb."
                },
                {
                    "atom_id": "v1_atom_002",
                    "frame_keys": [later_frame_key],
                    "entity_refs": ["entity_002"],
                    "fact": "The same white BMW begins moving forward."
                },
                {
                    "atom_id": "v1_atom_010",
                    "frame_keys": [later_frame_key],
                    "entity_refs": ["entity_001", "entity_002"],
                    "fact": "The rider passes behind the turning white BMW."
                },
                {
                    "atom_id": "v1_atom_011",
                    "frame_keys": [frame_key],
                    "entity_refs": ["entity_003"],
                    "fact": "Other parked vehicles remain along the curbside area."
                }
            ],
            "sensor_specific_cues": ["Specific visual cue observed."],
            "sensor_limitations": ["Limitation of visibility in this condition."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Partial visual evidence.", 
                    "missing_evidence": "Missing details.", 
                    "hypotheses": [
                        {"hypothesis": "First possible explanation.", "confidence": "low"},
                        {"hypothesis": "Second possible explanation.", "confidence": "low"}
                    ]
                }
            ],
            "missing_key_attributes": [
                {
                    "attribute_type": "existence", 
                    "missing_attribute": "Certain attribute missing.", 
                    "why_missing": "Not visible due to physical occlusion or condition.", 
                    "recoverable_evidence_refs": []
                }
            ]
        },
        "video2_analysis": {
            "modality": task.modality2,
            "detailed_caption": "The white BMW occupies a persistent curbside position. The same white BMW changes position later in the sampled interval. The rider and the white BMW form a close spatial relation. Other parked vehicles remain in the curbside row.",
            "information_atoms": [
                {
                    "atom_id": "v2_atom_001",
                    "frame_keys": [frame_key],
                    "entity_refs": ["entity_002"],
                    "fact": "The white BMW occupies a persistent curbside position."
                },
                {
                    "atom_id": "v2_atom_002",
                    "frame_keys": [later_frame_key],
                    "entity_refs": ["entity_002"],
                    "fact": "The same white BMW changes position later in the sampled interval."
                },
                {
                    "atom_id": "v2_atom_010",
                    "frame_keys": [later_frame_key],
                    "entity_refs": ["entity_001", "entity_002"],
                    "fact": "The rider and the white BMW form a close spatial relation."
                },
                {
                    "atom_id": "v2_atom_011",
                    "frame_keys": [frame_key],
                    "entity_refs": ["entity_003"],
                    "fact": "Other parked vehicles remain in the curbside row."
                }
            ],
            "sensor_specific_cues": ["Specific visual cue observed."],
            "sensor_limitations": ["Limitation of visibility in this condition."],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "cross_modal_evidence_links": [
            {
                "entity_id": "entity_002",
                "video1_evidence_refs": ["v1_atom_002"],
                "video2_evidence_refs": [],
                "shared_evidence": [],
                "unique_to_video1": ["Video 1 supports that the white BMW begins moving forward."],
                "unique_to_video2": [],
                "directional_contributions": [
                    {
                        "direction": "video1_improves_video2", 
                        "contribution": "Video 1 adds the forward movement described by v1_atom_002."
                    }
                ]
            }
        ],
        "information_gain": [],
        "reasoning_events": [
            {
                "event_id": "evt_001",
                "event_type": "confirmation",
                "participating_entities": ["entity_002"],
                "supporting_atom_refs": ["v1_atom_001", "v2_atom_001", "v1_atom_002", "v2_atom_002"],
                "description": "Both videos independently support the same grounded physical conclusion."
            }
        ],
        "ambiguity_events": [],
        "qa_relevant_details": [],
        "rejected_observations": [
            {"observation": "Observation without cross-modal value.", "reason": "Reason it was rejected."}
        ]
    }


def _build_caption_prompt(task: Any) -> str:
    frame_keys = ", ".join(f'"{path.stem}"' for path in task.composite_frames)
    frame_names = ", ".join(path.name for path in task.composite_frames)
    constraint_block = build_modality_constraint_block(task.modality1, task.modality2)
    schema_example_text = json.dumps(_build_prompt_schema_example(task), indent=2, ensure_ascii=False)

    return "\n".join(
        [
            "You are an expert multimodal perception analyst.",
            "You will receive multiple synchronized composite frames sampled from one aligned video segment.",
            f"Video 1 (left): {task.modality1}.",
            f"Video 2 (right): {task.modality2}.",
            "These two videos observe the same physical scene using different sensing modalities.",
            constraint_block,
            "Neither video is considered the reference or the ground truth.",
            "ROLE AND OBJECTIVE: Build a dense bidirectional multimodal evidence graph from directly observable evidence. Distinguish physical reality, video-local observations, and reasoning uncertainty. Do not invent objects, future events, intentions, identities, unreadable text, unsupported causes, or reverse-direction benefits.",
            "OBSERVABILITY / PHYSICAL-WORLD RULES: Describe the physical world as if standing there. global_scene.scene_summary and temporal_progression must not mention camera, sensor, modality, frame/image-processing, or image-quality artifacts. Forbidden wording includes modality, rgb, infrared, ir, thermal camera/image/frame/modality, event camera/stream/sensor/frame/modality, depth camera/sensor/map/frame/modality, edge map/edge-based/edge-like, heat signature/map, blurry, noisy, pixel/pixels, grayscale/greyscale, monochrome, overexposed, saturated. The words event, depth, edge, and heat are only forbidden in those sensor-specific phrases. scene_summary must be a full physical-world paragraph covering entities, appearance, spatial layout, environment, and ongoing actions.",
            "ENTITY REGISTRY CONSTRUCTION AND INVARIANTS: Build one shared modality-independent Entity Registry from physical referents implicated by both videos' provisional observations. 1. Stable referent identity: each Entity represents one stable physical referent and has one explicit referential_scope. 2. Same-object continuity: when observations across time or modalities refer to the same physical object and continuity is supported, reuse one entity_id; state, action, position, visibility, or modality-specific appearance changes do not create a new Entity by themselves. 3. Canonicalization: merge duplicate or equivalent referents; do not create one Entity per frame, observation, modality, or state. 4. Reasoning-relevant granularity: create only Entities needed for stable tracking, repeated reference, ambiguity resolution, interaction reasoning, or cross-modal identity. 5. Group-member handling: a specific group member may remain as a fact inside an Atom when it does not need independent identity; promote it only when independent reasoning requires stable identity. 6. Scope exclusivity: Entity scopes should be non-overlapping whenever possible; if a member becomes an independent Entity, exclude it from the residual group Entity. 7. Scope stability: never reuse one entity_id for a different object or granularity later in the caption.",
            "ATOM AND PROVENANCE INVARIANTS: 1. Each information atom is one minimal, modality-local, frame-grounded observation proposition. 2. Every atom contains atom_id, frame_keys, non-empty entity_refs, and fact. 3. atom.fact may describe one referenced Entity, multiple referenced Entities, or one directly observable relation, interaction, or joint event among referenced Entities. 4. A multi-entity atom is valid when one proposition directly describes a relation, interaction, or joint event among the referenced Entities. 5. If one atom combines multiple independently testable observations that do not form one directly observable relation, interaction, or joint event, split them into separate minimal atoms. 6. Evidence validity is determined by atom.fact semantics, not by anything else visible in atom.frame_keys. Frame co-occurrence does not imply semantic support. 7. Never assign entity_refs or cite an atom merely because an Entity or fact appears somewhere in the same frame. 8. Do not place intentions, unsupported causes, fusion conclusions, or multi-step inferences inside atoms. 9. No explicit atom does not mean the video cannot determine the fact.",
            "VIDEO-LOCAL ANALYSIS / TEMPORAL ALIGNMENT RULES: The supplied composite frames are temporally aligned, but Video 1 and Video 2 analyses do not need identical frame subsets, one-to-one atom pairs, equal atom counts, or placeholder atoms for unused frames. Track visibility, occlusion, interactions, temporal changes, and QA-relevant details only when directly supported. Be honest about uncertainty.",
            "CROSS-MODAL REASONING RULES: Asymmetric directional contributions, unidirectional disambiguation, mutual complementarity, and confirmation-only relations are valid when supported. Do not force every Entity into cross_modal_evidence_links or information_gain; include an Entity only when the evidence justifies that section. Do not invent bidirectional gain if one video only confirms the other or adds no unique information. Only include ambiguity_events when one video genuinely disambiguates the other; otherwise return an empty ambiguity_events list and explain rejected cases in rejected_observations. A segment may genuinely contain no reliable cross-modal reasoning value. If no cross-modal link, information gain, reasoning event, ambiguity event, or QA-relevant detail is sufficiently supported, return the corresponding lists as empty. Do not invent reasoning content merely to make the graph non-empty.",
            "GRAPH CONSTRUCTION AND EVIDENCE-CLOSURE WORKFLOW: Internally construct the graph in this order: 1. Draft provisional Video 1 source-local atoms. 2. Draft provisional Video 2 source-local atoms. 3. Build and reconcile the shared Entity Registry. 4. Assign final entity_refs to all atoms. 5. Draft candidate downstream claims. 6. Check every claim against atom.fact semantics and Entity-Atom links. 7. If required evidence is directly observable but missing, add a minimal source-local atom with valid frame_keys and correct entity_refs. 8. If evidence is not reliably observable, remove the claim or that video's contribution. 9. Emit only downstream positive claims whose evidence closure is complete.",
            "FIELD / ID / REFERENCE REQUIREMENTS: All referenceable IDs must be globally unique. information_atoms use v1_atom_ or v2_atom_; reasoning_events use evt_; ambiguity_events use amb_; qa_relevant_details use qa_detail_. Information atoms are the only structures with frame_keys; reasoning_events and ambiguity_events cite atom IDs for provenance. qa_relevant_details.supporting_refs must not reference another qa_detail. Every information atom must use valid frame_keys and non-empty entity_refs pointing to existing global_scene.physical_entities[].entity_id values, with no duplicate entity_refs.",
            f"VALID FRAME KEYS: [{frame_keys}]. information_atoms[].frame_keys MUST choose only from these exact values.",
            "ADDITIONAL FIELD RULES: If an attribute is missing from one video but recoverable from the other, missing_key_attributes[].recoverable_evidence_refs must list recovering atom IDs; otherwise use an empty list. evidence_profile fields must be omitted when there is no meaningful non-dynamic evidence for them; do not return empty lists or empty strings. detailed_caption and global_scene must use physical-world wording only.",
            "TARGETED QUALITY RULES — ATOM MINIMALITY AND SOURCE-LOCAL DETAILED CAPTIONS: Apply the following rules strictly. These rules are intended to improve evidence quality only. Do not force additional cross-modal links, directional gains, information gain, ambiguity events, reasoning events, or QA details. Any of those sections may remain empty when unsupported.",
            "1. STRICT ATOM MINIMALITY: Each information atom must express exactly one minimal, directly observable proposition. Use the INDEPENDENT TRUTH TEST: If two clauses could independently be true or false, they must be split into separate atoms. If removing one clause would still leave another complete independently testable observation, split them. Do not combine multiple sequential actions, multiple unrelated objects, or an action plus an unrelated visibility statement in one atom. The presence of several entities in the same frame does not justify combining them into one atom. A single atom may reference multiple entities only when the proposition itself is one directly observable relation, interaction, or joint event among those entities. Before emitting every atom, perform this check: (A) How many independently testable propositions are present? (B) Could one clause be false while another remains true? (C) Does every referenced entity participate in the single proposition stated by the atom? If the answer indicates more than one independent proposition, split the atom. Do not create placeholder atoms merely to use every supplied frame.",
            "2. DETAILED_CAPTION MUST DESCRIBE THE SPECIFIC PHYSICAL SCENE: For video1_analysis.detailed_caption and video2_analysis.detailed_caption: Describe only the concrete source-local scene observations and their temporal progression in the supplied segment. The detailed_caption should answer: What physical entities are present? Where are they located? What changes over time? Which object approaches, passes, appears, disappears, remains stationary, becomes occluded, or changes position? What directly observable spatial or temporal relations occur? Do not use the detailed_caption to explain how the sensing modality works. Especially for event-based input, do NOT describe generic sensor theory, signal-processing behavior, or modality mechanisms such as: high-frequency motion boundaries, transient intensity changes, intensity-change responses, event activity, contrast response, contrast transitions, spatial-frequency structure, edge response, signal response, camera translation producing events, camera-platform motion producing boundaries, the sensor captures changes, the modality highlights high-frequency structure, sharp profile boundaries (unless tied to a concrete physical object and stated as a direct scene observation).",
            "3. KEEP SENSOR-SPECIFIC ANALYSIS IN THE CORRECT FIELDS: If a modality-specific cue is useful, place it in: sensor_specific_cues. If a modality-specific limitation is useful, place it in: sensor_limitations. Do not move generic sensor theory into detailed_caption. Even inside sensor_specific_cues and sensor_limitations: remain segment-specific; refer to evidence actually present in the supplied segment; do not provide generic textbook explanations of the modality.",
            "4. PHYSICAL-WORLD WORDING PRIORITY: Prefer physical-world statements over representation-level statements. Prefer: 'The sedan is visible on the right.', 'The bicycle approaches the sedan.', 'The rider passes the parked car.', 'Tree-shadow contours cross the roadway.', 'The van appears farther ahead near the curb.' Avoid: 'The modality captures...', 'The sensor detects...', 'The stream represents...', 'The camera motion produces...', 'The signal highlights...', 'The high-frequency response shows...'",
            "5. DO NOT COMPENSATE BY INVENTING CROSS-MODAL REASONING: These repairs must not cause artificial downstream reasoning. After making atoms more minimal and detailed captions more scene-specific: do not force shared evidence; do not force unique evidence; do not force directional contributions; do not force information gain; do not force ambiguity events; do not force QA-relevant details. Emit empty lists whenever the evidence does not support a reliable claim. If a previously tempting downstream claim depends only on generic sensor-theory wording rather than a concrete physical-world atom, omit that claim.",
            "6. FINAL SELF-CHECK BEFORE OUTPUT: ATOM CHECK: Every atom contains one minimal proposition. No atom combines unrelated observations using 'and', 'while', 'with ... visible', or similar constructions unless they form one indivisible relation or interaction. Every entity_ref participates in the atom's single stated proposition. DETAILED CAPTION CHECK: Each detailed_caption describes this specific segment. Each detailed_caption focuses on physical entities, actions, locations, and temporal progression. No detailed_caption explains generic sensor behavior or signal-processing theory. Event-based descriptions do not use abstract signal-level wording when a concrete physical-world description is possible. DOWNSTREAM CHECK: No cross-modal gain is invented merely because the repaired atoms are more detailed. Unsupported reasoning sections remain empty.",
            "OUTPUT SCHEMA: Return ONLY a valid JSON object matching this exact skeleton.",
            schema_example_text,
            "ALLOWED enum values for fields:",
            "- confidence: " + ", ".join(sorted(ALLOWED_GAIN_RATINGS)),
            _enum_line("attribute_type / missing_attribute_type", ALLOWED_MISSING_ATTRIBUTE_TYPES),
            _enum_line("gain_rating", ALLOWED_GAIN_RATINGS),
            _enum_line("gain_type", ALLOWED_GAIN_TYPES),
            _enum_line("event_type", ALLOWED_REASONING_EVENT_TYPES),
            _enum_line("direction", ALLOWED_AMBIGUITY_DIRECTIONS),
            "- ambiguous_video / resolving_video: video1, video2",
            _enum_line("reasoning_pattern", ALLOWED_QA_REASONING_PATTERNS),
            _enum_line("cross_modal_evidence_links directional_contributions[].direction", ALLOWED_CROSS_MODAL_CONTRIBUTION_DIRECTIONS),
            "REASONING EVENT TYPE DEFINITIONS:",
            "- confirmation: Both videos independently support the same physical conclusion. Fusion mainly increases confidence.",
            "- cross_modal_complementarity: Each video contributes different useful evidence, but neither necessarily resolves a strict ambiguity.",
            "- unidirectional_disambiguation: One video resolves a concrete ambiguity present in the other.",
            "- temporal_change: A grounded change across multiple selected timestamps.",
            "- interaction: A grounded entity-entity or entity-object interaction.",
            "- occlusion_change: An entity's visibility/occlusion state changes across time.",
            "- spatial_transition: A grounded change in relative or absolute spatial configuration.",
            "- joint_fusion: The conclusion genuinely requires evidence from both videos and cannot be reduced to one video simply resolving the other.",
            "OPTIONAL NON-EMPTY ITEM SHAPES:",
            "These shapes are structural guidance only. They do not imply that any item must be emitted. Keep the corresponding list empty when unsupported. If a list is non-empty, every item must be fully populated; never emit partial placeholder objects.",
            "For information_gain[]:",
            "```json\n{\n  \"entity_id\": \"entity_...\",\n  \"video1_evidence_refs\": [],\n  \"video2_evidence_refs\": [],\n  \"video1_can_determine\": [],\n  \"video1_cannot_determine\": [],\n  \"video2_can_determine\": [],\n  \"video2_cannot_determine\": [],\n  \"fusion_additionally_reveals\": [],\n  \"gain_type\": \"confidence_gain\",\n  \"gain_rating\": \"low\"\n}\n```",
            "at least one evidence side must be non-empty for an actual gain item; 'confirmation' requires both sides; 'complementarity' requires both sides; asymmetric 'disambiguation', 'semantic_emergence', and 'confidence_gain' remain allowed where valid.",
            "For qa_relevant_details[]:",
            "```json\n{\n  \"detail_id\": \"qa_detail_001\",\n  \"reasoning_pattern\": \"temporal_integration\",\n  \"supporting_refs\": [\"evt_001\"],\n  \"why_question_worthy\": \"...\"\n}\n```",
            "For ambiguity_events[]:",
            "Required fields: ambiguity_id, target_entity, direction, ambiguous_video, resolving_video, low_confidence_observation, why_ambiguous_video_cannot_resolve, candidate_hypotheses, resolving_discriminative_evidence, eliminated_hypotheses, fusion_conclusion, missing_attribute_type, ambiguous_evidence_refs, resolving_evidence_refs",
            "For every optional reasoning list: either return [] or return fully populated valid items. Never emit partial objects with missing required fields or wrong field types.",
            "FINAL GRAPH CONSISTENCY SELF-CHECK: Before returning JSON, confirm these compact checks: every downstream positive claim has complete evidence closure; every cited Atom semantically supports the claim for which it is cited; no Entity is duplicated under equivalent normalized scope; no frame co-occurrence is used as semantic evidence; no unsupported symmetric contribution is invented; no optional ambiguity, information_gain, reasoning_event, or QA item is included without grounded evidence; do not add optional reasoning items merely to avoid empty reasoning sections.",
            f"Segment: {task.segment_id}; side: {task.side}.",
            f"Composite frames ({len(task.composite_frames)} images): {frame_names}",
        ]
    )



def _template_caption(task: Any) -> dict[str, Any]:
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
                    "recoverable_evidence_refs": ["v1_atom_001"]
                }
            ],
        },
        "cross_modal_evidence_links": [
            {
                "entity_id": target_entity,
                "video1_evidence_refs": ["v1_atom_001"],
                "video2_evidence_refs": ["v2_atom_001"],
                "shared_evidence": ["Placeholder"],
                "unique_to_video1": ["Placeholder"],
                "unique_to_video2": [],
                "directional_contributions": [{"direction": "video1_improves_video2", "contribution": "Placeholder"}]
            }
        ],
        "information_gain": [
            {
                "entity_id": target_entity,
                "video1_evidence_refs": ["v1_atom_001"],
                "video2_evidence_refs": ["v2_atom_001"],
                "video1_can_determine": ["Placeholder"],
                "video1_cannot_determine": ["Placeholder"],
                "video2_can_determine": ["Placeholder"],
                "video2_cannot_determine": ["Placeholder"],
                "fusion_additionally_reveals": ["Placeholder"],
                "gain_type": "confirmation",
                "gain_rating": "low",
            }
        ],
        "reasoning_events": [
            {
                "event_id": "evt_001",
                "event_type": "joint_fusion",
                "participating_entities": [target_entity],
                "supporting_atom_refs": ["v1_atom_001", "v2_atom_001"],
                "description": "Template placeholder event"
            }
        ],
        "ambiguity_events": [
            {
                "ambiguity_id": "amb_001",
                "target_entity": target_entity,
                "direction": "video1_resolves_video2",
                "ambiguous_video": "video2",
                "resolving_video": "video1",
                "low_confidence_observation": "Placeholder",
                "why_ambiguous_video_cannot_resolve": "Placeholder",
                "candidate_hypotheses": [
                    {"hypothesis": "Placeholder 1", "why_compatible_with_ambiguous": "Placeholder", "support_from_resolving": "Placeholder"},
                    {"hypothesis": "Placeholder 2", "why_compatible_with_ambiguous": "Placeholder", "support_from_resolving": "Placeholder"}
                ],
                "resolving_discriminative_evidence": "Placeholder",
                "eliminated_hypotheses": [{"hypothesis": "Placeholder 1", "why_eliminated": "Placeholder"}],
                "fusion_conclusion": "Placeholder",
                "missing_attribute_type": "existence",
                "ambiguous_evidence_refs": ["v2_atom_001"],
                "resolving_evidence_refs": ["v1_atom_001"]
            }
        ],
        "qa_relevant_details": [
            {
                "detail_id": "qa_detail_001",
                "reasoning_pattern": "joint_fusion",
                "supporting_refs": ["evt_001", "amb_001"],
                "why_question_worthy": "Placeholder"
            }
        ],
        "rejected_observations": [
            {
                "observation": "No rejected observation was evaluated in template mode.",
                "reason": "Gemini was not called in template mode."
            }
        ],
    }


