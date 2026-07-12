"""Prompt, schema example, and template caption builders for Pass 1 (evidence construction)."""

from __future__ import annotations

import json
from typing import Any

from annotation_feature.aligned_caption_schema import (
    ALLOWED_MISSING_ATTRIBUTE_TYPES,
    _enum_line,
    build_modality_constraint_block,
    normalize_modality_name,
    SUPPORTED_MODALITIES,
)
from types import MappingProxyType

MODALITY_PHYSICAL_GUIDANCE = MappingProxyType({
    "rgb": "### ACTIVE MODALITY GUIDANCE: RGB\n- Supported: Visible color, surface texture, paint and material appearance, visible lighting effects.\n- Avoid: Do not use representation-oriented wording. Do not claim absolute distance or hidden structure.\n- Valid: 'The vehicle is white.' 'The surface is smooth.'\n- Invalid: 'The RGB image captures a white vehicle.'",
    "event": (
        "### ACTIVE MODALITY GUIDANCE: EVENT\n"
        "- Conversion policy: When independently supported by the current Event source, infer the strongest defensible physical-world proposition: coarse object presence, changes in position, extent, or orientation, directly supported motion, temporal or spatial relations, coarse physical structure or layout, or visibility under difficult illumination.\n"
        "- A line, contour, silhouette, outline, boundary, or edge transition is a raw cue, not automatically a valid final atom proposition. Event-local atoms and captions must not describe how such structures are traced, highlighted, defined, resolved, encoded, captured, or represented. Never borrow an object identity from RGB to turn an Event cue into a semantic fact.\n"
        "- If no reliable physical proposition is supported, place only the segment-specific raw cue in `sensor_specific_cues` or omit it.\n"
        "- Valid physical facts, only when independently supported: 'Several parked vehicles remain distinguishable along the roadside.' 'The cyclist changes position relative to the parked vehicle.' 'A person moves from the left side of the roadway toward the center.' 'The building facade contains repeated rectangular structural regions.'\n"
        "- Invalid atom or caption wording: 'The vehicle silhouette is traced by sharp boundaries.' 'Edge transitions define the building facade.' 'The object is outlined against the background.' 'The wall is reduced to a single vertical line.' 'Motion boundaries capture the parked vehicles.'\n"
        "- Valid raw cue placement in `sensor_specific_cues`: 'A narrow vertical response remains in the final sampled time.' This cue alone does not justify a building or wall atom."
    ),
    "depth": "### ACTIVE MODALITY GUIDANCE: DEPTH\n- Supported: Relative depth, one entity standing in front of another, surface geometry.\n- Avoid: Do not use representation-oriented wording ('depth map represents'). Do not claim color, material, or exact metric distance unless directly supported.\n- Valid: 'The person stands in front of the wall.'\n- Invalid: 'A depth discontinuity separates the person and wall.'",
    "ir": "### ACTIVE MODALITY GUIDANCE: IR\n- Supported: Relative infrared appearance or contrast (e.g., one physical region being brighter/darker than another).\n- Avoid: Do not guess thermal IR from pixel appearance alone. Do not claim exact temperature or causal thermal conclusions unless metadata explicitly establishes thermal infrared. Do not use representation-oriented wording.\n- Valid: 'The person is brighter than the surrounding background.'\n- Invalid: 'The infrared channel highlights a warm object.'"
})

PAIR_PHYSICAL_GUIDANCE = MappingProxyType({
    frozenset({"rgb", "event"}): "### ACTIVE PAIR GUIDANCE: RGB + EVENT\nRGB may independently support visible color and surface appearance. Event may independently support temporal change and changing structure. Do not copy RGB-only color into Event-local text or shared fields."
})

def build_modality_physical_guidance(modality1: str, modality2: str) -> str:
    mod1 = normalize_modality_name(modality1)
    mod2 = normalize_modality_name(modality2)

    blocks = []

    unsupported = [raw for raw, normalized in ((modality1, mod1), (modality2, mod2)) if normalized not in SUPPORTED_MODALITIES]
    if unsupported:
        raise ValueError(f"Unsupported modality: {', '.join(map(str, unsupported))}")

    active_mods = []
    if mod1 in SUPPORTED_MODALITIES:
        active_mods.append(mod1)
    if mod2 in SUPPORTED_MODALITIES and mod2 != mod1:
        active_mods.append(mod2)

    for mod in active_mods:
        if mod in MODALITY_PHYSICAL_GUIDANCE:
            blocks.append(MODALITY_PHYSICAL_GUIDANCE[mod])

    pair_key = frozenset({mod1, mod2})
    if pair_key in PAIR_PHYSICAL_GUIDANCE:
        blocks.append(PAIR_PHYSICAL_GUIDANCE[pair_key])

    return "\n\n".join(blocks)

def _build_prompt_schema_example(task: Any, modality1: str | None = None, modality2: str | None = None) -> dict[str, Any]:
    fk = task.composite_frames[0].stem if task.composite_frames else "frame_000000"
    modality1 = normalize_modality_name(modality1 if modality1 is not None else task.modality1)
    modality2 = normalize_modality_name(modality2 if modality2 is not None else task.modality2)
    return {
        "global_scene": {
            "scene_summary": "A barrier stands beside a paved roadway in an urban setting, and the surrounding physical layout remains consistent throughout the observed segment.",
            "environment": "urban",
            "temporal_progression": "The barrier and adjacent paved roadway remain in a stable spatial arrangement across the sampled times.",
            "physical_entities": [
                {
                    "entity_id": "entity_001",
                    "category": "vehicle",
                    "referential_scope": "the vehicle"
                },
                {
                    "entity_id": "entity_002",
                    "category": "barrier",
                    "referential_scope": "the barrier"
                },
                {
                    "entity_id": "entity_003",
                    "category": "lighting_effect",
                    "referential_scope": "the shadow on the ground"
                },
                {
                    "entity_id": "entity_004",
                    "category": "roadway",
                    "referential_scope": "the paved surface surrounding the barrier"
                }
            ]
        },
        "video1_analysis": {
            "modality": modality1,
            "detailed_caption": "A vehicle rests beside a concrete barrier. A sharp shadow from the barrier falls across the adjacent paved road surface.",
            "information_atoms": [
                {
                    "atom_id": "v1_atom_001",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_001", "entity_002"],
                    "fact": "A vehicle rests beside a concrete barrier."
                },
                {
                    "atom_id": "v1_atom_002",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_002", "entity_003", "entity_004"],
                    "fact": "A sharp shadow of the barrier falls across the road surface."
                }
            ],
            "sensor_specific_cues": ["High contrast shadow line marks the barrier edge."],
            "sensor_limitations": ["Low ambient light obscures the surface details of the vehicle."],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": modality2,
            "detailed_caption": "A concrete barrier has a flat upper surface and a vertical side face. Individual cobblestones surround the base of the barrier.",
            "information_atoms": [
                {
                    "atom_id": "v2_atom_001",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_002"],
                    "fact": "A concrete barrier has a flat upper surface and a vertical side face."
                },
                {
                    "atom_id": "v2_atom_002",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_002", "entity_004"],
                    "fact": "Individual cobblestones surround the base of the barrier."
                }
            ],
            "sensor_specific_cues": [],
            "sensor_limitations": [],
            "uncertain_observations": [],
            "missing_key_attributes": []
        }
    }


def build_pass1_system_prompt() -> str:
    return """PASS 1 ROLE, OBJECTIVE, AND SCOPE
This is not ordinary captioning. Build dense, reasoning-relevant, frame-grounded evidence for two aligned sources. Neither source is ground truth, and only directly observable evidence may be asserted. Pass 1 constructs only `global_scene`, `video1_analysis`, and `video2_analysis`; it does not perform cross-modal fusion, information-gain or directional-contribution analysis, cross-source ambiguity resolution, reasoning-event construction, QA generation, or Pass 2 recoverability resolution.

EVIDENCE-CONSTRUCTION WORKFLOW / INTERNAL CONVERSION ORDER (FOLLOW IN ORDER)
1. Inspect Video 1 independently and draft only source-local observations.
2. Inspect Video 2 independently and draft only source-local observations.
3. Build and reconcile one conservative physical Entity Registry. Reuse an ID across sources or time only when continuity is supported.
4. Create minimal source-local `information_atoms` with valid frame and entity references. Atoms come before captions.
5. Check every source-local field with the opposite source and registry wording temporarily hidden; remove unsupported leakage.
6. Write each `detailed_caption` only as a lossy prose compression of its same-source atoms.
7. Add only genuine unresolved source-local uncertainty that remains consistent with those atoms.
8. Add a missing attribute only when it is a valid Pass 2 resolution target under the existing five-field contract.
9. Run the final schema, reference, grounding, wording, and scope checks below, then return only JSON.

CORE SEMANTIC INVARIANTS

ENTITY REGISTRY
- Assign one stable `entity_id` to each physical referent and never reuse it for a different referent. Do not merge distinct objects for brevity.
- Register salient independently distinguishable physical entities. In particular, an identifiable object occupying roughly 5% or more of a frame must not be omitted; also preserve smaller entities when they are important to relations, events, or later reasoning.
- Registry metadata organizes identity; it is not source-local evidence. Use a conservative but useful category and `referential_scope`. Do not leak source-exclusive color, material, fine identity, state, motion, text, brand, or make/model into shared wording.
- `scene_summary`, `temporal_progression`, and Registry wording are shared-global text. Every descriptive attribute they assert must be independently supported by both active sources; otherwise omit or neutralize it.
- Entity `category` names a physical object class. Use `lighting_effect` for shadows, glare, or reflections and `unknown_object` for an unresolved object.

SOURCE-LOCAL ISOLATION
- For every field under a video analysis, temporarily ignore the opposite source and the wording of `referential_scope`. Use the strongest description independently supported by the current source, but no stronger.
- Never borrow an attribute or interpretation merely because the other source or Registry contains it. Downgrade unsupported adjectives, identities, categories, states, motion, directions, text, or causal links.
- Shared coarse physical facts may legitimately be supported by both modalities. Near-identical facts do not imply that either source is wrong. Differentiate the sources only when the supplied frames genuinely support different evidence; never manufacture modality-exclusive atoms.

ATOM MINIMALITY AND PROVENANCE
- Create atoms before captions. Each atom states one minimal, directly observable physical proposition. There is no upper atom-count limit, but do not create one atom per frame, placeholders, or repeated low-value paraphrases.
- Every atom has a unique correctly prefixed `atom_id` (`v1_atom_` or `v2_atom_`), one or more supplied `frame_keys`, non-empty `entity_refs`, and a non-empty `fact`.
- Every physical participant explicitly named by an atom must be registered and included in `entity_refs`. A multi-entity atom is allowed only for one direct relation, interaction, or joint event; frame co-occurrence alone is not semantic support.
- Make important observable facts explicit as atoms so downstream stages can use them. Prefer new evidence about salient entities, relations, temporal phases, occlusion, motion, or state changes over redundant wording.

CAPTION CLOSURE
- `detailed_caption` is a lossy prose compression of same-source atoms: it may contain less information, never more. Every important caption claim must map to one or more same-source atom IDs.
- If no atom supports a proposed claim, create a minimal atom first or remove the claim. Captions must not add unsupported transitions, identities, actions, directions, observer motion, or causal links.
- Temporal words such as initially, later, or then require a same-source atom establishing that change. Match atom wording closely enough that provenance remains unambiguous.
- Meet existing minimum lengths with supported content only. Aim for 24-40 words in `scene_summary` and 35-60 words in each `detailed_caption`; never add filler.

PHYSICAL-WORLD WORDING
- Describe physical entities, structure, spatial relations, motion, visibility, and temporal change directly. PHYSICAL-WORLD SUBJECT TEST: the subject should be the physical scene, not an image, channel, signal, response, map, or representation.
- MODALITY-HIDDEN TEST: if the modality name were hidden, an atom or caption should still read as a natural physical-world observation. Otherwise rewrite it, move only the raw segment-specific response to `sensor_specific_cues`, or omit it.
- `sensor_specific_cues` is the ONLY place where mechanism-oriented response patterns are permitted, and a raw cue does not justify an unsupported physical interpretation.
- `sensor_limitations` and `why_missing` must be anchored to this segment, a registered entity, or supplied frames. Generic sensor theory that would apply equally to every segment is forbidden.
- Avoid inferential linkage in grounded text. In particular, do not use `corresponding`, `suggesting`, `indicating`, or `implying`; captions also avoid appear/seem and representation-oriented explanations. `scene_summary` and `temporal_progression` must not mention camera, lens, viewpoint, or frame.

UNCERTAINTY CONSISTENCY
- `uncertain_observations` is optional; an empty list is valid and preferred over invented uncertainty.
- Every item uses a unique correctly prefixed uncertainty ID (`v1_unc_` or `v2_unc_`), one known `entity_id`, and at least one same-source `evidence_ref` connected to that entity. `hypotheses` may be empty; when present, include at least two distinct plausible interpretations.
- Hypotheses arise from current-source evidence and MUST NOT contradict a same-source Atom. If an Atom establishes a coarse category, uncertainty may concern only finer compatible categories; do not reopen an incompatible coarse category.
- Do not claim a property is missing when a same-source atom establishes it. For each item, read only its cited atoms: if they resolve the property, either remove the uncertainty or weaken the Atom.

MISSING ATTRIBUTES: PASS 2 TARGETS
- `missing_key_attributes` is optional. Empty missing lists are preferable to unrecoverable targets and are never forced to be non-empty.
- Include a target only when the opposite source has explicit Entity-connected Atom evidence directly supporting the missing property and resolving it matters; modality capability alone is never enough.
- Every item targets one known `entity_id`. `missing_attribute` names an unknown property, not an unknown concrete value; `why_missing` gives a segment-local explanation.
- Do not propose make/model without direct distinguishing opposite-source evidence; coarse shape is not make/model evidence. Do not propose license-plate characters without readable opposite-source characters; an outer outline is not license-plate-text evidence.
- Every item contains exactly these five fields and no others:
  {"entity_id": "entity_004", "attribute_type": "<allowed enum>", "missing_attribute": "<unknown property>", "why_missing": "<segment-local explanation>", "recoverable_evidence_refs": []}
- `recoverable_evidence_refs` remains exactly [] in Pass 1. Nulls, empty strings, placeholders, partial objects, extra fields, and `missing_attribute_type` are forbidden.

EVIDENCE DENSITY AND TEMPORAL COVERAGE
- Maximize distinct reasoning-relevant evidence without exceeding what each source directly supports. Cover salient entities, spatial relations, interactions, temporal phases, occlusion changes, motion, and state changes when observable.
- Do not force equal atom counts across modalities or exclusive observations. Do not spend many atoms repeating one entity while omitting other salient evidence.
- `temporal_progression` must describe observable change across time or explicitly state supported stability; it must not be only a static inventory.

FINAL SELF-CHECK
- Output has exactly `global_scene`, `video1_analysis`, and `video2_analysis`, with no downstream Pass 2 fields.
- IDs are unique and correctly prefixed; all frame keys are supplied; every reference resolves; every named atom participant is registered and referenced.
- Every atom is one grounded proposition; every important caption claim closes to same-source atoms.
- Registry and both source-local analyses pass the source-isolation test; shared coarse facts are allowed without artificial differentiation.
- Uncertainty is genuine and atom-consistent; every missing attribute satisfies the five-field actual-evidence contract and has `recoverable_evidence_refs: []`.
- Physical-world wording is used; raw response wording appears only in `sensor_specific_cues`; limitations are segment-anchored.
- `temporal_progression` expresses change or explicit stability. Required fields, enums, minimum lengths, and the supplied JSON skeleton are preserved.

OUTPUT CONTRACT: Return ONLY valid JSON matching the provided skeleton. No prose and no other top-level fields."""


def build_pass1_user_prompt(task: Any, modality1: str | None = None, modality2: str | None = None) -> str:
    modality1 = normalize_modality_name(modality1 if modality1 is not None else task.modality1)
    modality2 = normalize_modality_name(modality2 if modality2 is not None else task.modality2)
    frame_names = ", ".join(path.name for path in task.composite_frames)
    constraint_block = build_modality_constraint_block(modality1, modality2).split(
        "\n\nCROSS-MODAL DIFFERENTIATION GUIDE", 1
    )[0]
    dynamic_guidance = build_modality_physical_guidance(modality1, modality2)
    schema_example_text = json.dumps(_build_prompt_schema_example(task, modality1, modality2), indent=2, ensure_ascii=False)

    return "\n".join(
        [
            "TASK: Construct PASS 1 evidence for an aligned multimodal video segment. Neither source is ground truth.",
            "SCOPE: Produce ONLY global_scene, video1_analysis, and video2_analysis. NO cross-modal fusion, information-gain, directional contribution, cross-source ambiguity resolution, QA, or recoverability reasoning.",
            "PRIORITIES: source-local correctness, valid provenance, stable entity identity, dense reasoning-relevant coverage, minimal atoms, and caption-to-atom closure.",
            constraint_block,
            f"CURRENT SOURCE MODALITIES\nSource 1 modality: {modality1}\nSource 2 modality: {modality2}",
            dynamic_guidance,
            schema_example_text,
            "ALLOWED ENUMS:",
            "- confidence: low, medium, high",
            _enum_line("attribute_type", ALLOWED_MISSING_ATTRIBUTE_TYPES),
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
                    "uncertainty_id": "v1_unc_001",
                    "entity_id": target_entity,
                    "observed_evidence": "Template placeholder",
                    "missing_evidence": "Placeholder",
                    "evidence_refs": ["v1_atom_001"],
                    "hypotheses": [
                        {"hypothesis": "Placeholder 1", "confidence": "low"},
                        {"hypothesis": "Placeholder 2", "confidence": "low"}
                    ]
                }
            ],
            "missing_key_attributes": [
                {
                    "entity_id": target_entity,
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
                    "uncertainty_id": "v2_unc_001",
                    "entity_id": target_entity,
                    "observed_evidence": "Template placeholder",
                    "missing_evidence": "Placeholder",
                    "evidence_refs": ["v2_atom_001"],
                    "hypotheses": [
                        {"hypothesis": "Placeholder 1", "confidence": "low"},
                        {"hypothesis": "Placeholder 2", "confidence": "low"}
                    ]
                }
            ],
            "missing_key_attributes": [
                {
                    "entity_id": target_entity,
                    "attribute_type": "existence",
                    "missing_attribute": "Template placeholder",
                    "why_missing": "Template placeholder",
                    "recoverable_evidence_refs": []
                }
            ],
        }
    }
