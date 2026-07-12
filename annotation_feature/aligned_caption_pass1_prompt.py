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
    "event": "### ACTIVE MODALITY GUIDANCE: EVENT\n- Supported: Changes in physical boundaries, object position, and motion. Physical outlines under poor illumination.\n- Avoid: Do not use representation-oriented wording ('event activity', 'response clusters', 'dense event clusters surround').\n- Valid: 'The vehicle outline remains distinguishable.'\n- Invalid: 'The event representation resolves the facade.'",
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
            "scene_summary": "A vehicle is situated near a concrete barrier while casting a distinct shadow across the adjacent paved surface under ambient urban lighting conditions.",
            "environment": "urban",
            "temporal_progression": "The spatial relationship between the vehicle and barrier remains consistent.",
            "physical_entities": [
                {
                    "entity_id": "entity_001",
                    "category": "vehicle",
                    "referential_scope": "the vehicle"
                },
                {
                    "entity_id": "entity_002",
                    "category": "barrier",
                    "referential_scope": "concrete barrier"
                },
                {
                    "entity_id": "entity_003",
                    "category": "lighting_effect",
                    "referential_scope": "shadow of the barrier cast on the ground"
                },
                {
                    "entity_id": "entity_004",
                    "category": "roadway",
                    "referential_scope": "cobblestone surface surrounding the barrier"
                }
            ]
        },
        "video1_analysis": {
            "modality": modality1,
            "detailed_caption": "A vehicle rests beside a concrete barrier. The vehicle's surface shows various textures, and a sharp shadow of the barrier falls across the road surface in this urban segment.",
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
                    "entity_refs": ["entity_003"],
                    "fact": "A sharp shadow of the barrier falls across the road surface."
                }
            ],
            "sensor_specific_cues": ["High contrast shadow line marks the barrier edge."],
            "sensor_limitations": ["Low ambient light obscures the surface details of the vehicle."],
            "uncertain_observations": [
                {
                    "uncertainty_id": "v1_unc_001",
                    "entity_id": "entity_001",
                    "observed_evidence": "Partial outline of the vehicle.",
                    "missing_evidence": "Internal structural details.",
                    "evidence_refs": ["v1_atom_001"],
                    "hypotheses": [
                        {"hypothesis": "Small passenger car.", "confidence": "low"},
                        {"hypothesis": "Delivery van.", "confidence": "low"}
                    ]
                }
            ],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": modality2,
            "detailed_caption": "A concrete barrier has a straight upper boundary and a vertical side face. Individual cobblestones remain structurally distinguishable around its base, preserving the visible layout of the paved area surrounding the barrier.",
            "information_atoms": [
                {
                    "atom_id": "v2_atom_001",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_002"],
                    "fact": "A concrete barrier has a straight upper boundary and a vertical side face."
                },
                {
                    "atom_id": "v2_atom_002",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_004"],
                    "fact": "Individual cobblestones remain structurally distinguishable around the base of the barrier."
                }
            ],
            "sensor_specific_cues": [],
            "sensor_limitations": [],
            "uncertain_observations": [],
            "missing_key_attributes": []
        }
    }


def build_pass1_system_prompt() -> str:
    return "\n".join(
        [
            "RULES:",
            "1. SHARED ENTITIES: Reuse `entity_id` for the same physical referent. Never reuse an ID for different referents. Create entities for salient independent physical referents. Preserve minimality; do not hide genuinely salient referents. ENTITY COMPLETENESS SCAN (MANDATORY): Before writing any atom, scan ALL visible physical objects in every supplied frame and enumerate a candidate entity for each. Apply this minimum bar: if an object is (a) large enough to occupy more than ~5% of frame area, AND (b) can be identified to at least a coarse category (e.g. vehicle, building, vegetation), it MUST receive an entity_id. Common omissions to watch for: multiple parked vehicles counted as one, background buildings ignored, trees and utility poles missed. Do NOT merge distinct physical objects into a single entity merely for brevity.",
            "2. REGISTRY METADATA: Global registry metadata is organizational, not source-local evidence. Use conservative but useful identity descriptions. Do not automatically over-neutralize every referent to 'object' or 'unknown', but do not insert source-exclusive details that create leakage risk.",
            "3. ISOLATION & LEAKAGE: Cross-modal leakage occurs ONLY when a claim is not independently supported by the current source. For each source, use the strongest description independently justified by that source: no stronger, but also no weaker merely because another source supports the same interpretation. SOURCE-LOCAL DESCRIPTION DOWNGRADE (MANDATORY): For each source-local caption and atom: 1. Temporarily hide the other source; 2. Treat source-exclusive attributes from the shared entity registry as unavailable; 3. Keep only the most specific description the current source independently supports; 4. Delete or downgrade colors, textures, brands, fine-grained categories, or states that the current source cannot support; 5. Never copy unsupported adjectives just to match registry text. REGISTRY-HIDDEN TEST: Ignore the referential_scope wording and judge the sentence using only the current source. Can every adjective, category, and state still be independently justified? If not, downgrade the description.",
            "4. CROSS-MODAL DIFFERENTIATION — REQUIRED: video1_analysis and video2_analysis MUST each contain at least one atom whose fact is NOT shared by the other modality's atoms. The two modalities are complementary physical instruments; they DO NOT observe the same physical events in the same way. For each modality, look for what is EXCLUSIVELY visible or exclusively absent in this specific source. FORBIDDEN: An atom that is textually near-identical to an atom in the other modality's block. If you find yourself writing the same physical fact for both modalities, one of them is wrong — one modality must contribute a unique exclusive observation that the other modality cannot provide. Refer to the CROSS-MODAL DIFFERENTIATION GUIDE above for a concrete list of what each modality in this segment is exclusively able to capture.",
            "5. ATOMS FIRST — HARD RULE: Build all `information_atoms` before writing `detailed_caption`. There is NO upper limit on atom count; do not reduce atoms for brevity. For every semantic claim you want to include in `detailed_caption`, apply this binary decision: (A) An existing same-source atom directly supports this claim → you may include it in the caption. (B) No atom supports it → you MUST either create a minimal new atom first and then include the claim, OR drop the claim from the caption entirely. A claim present in `detailed_caption` but absent from same-source `information_atoms` is a HARD VIOLATION. Each atom = 1 minimal physical proposition. Unique `atom_id` (v1_atom_/v2_atom_ prefix), valid `frame_keys`, non-empty `entity_refs`. Prefer physical-world wording over mechanism-oriented phrasing.",
            "6. CAPTION CLOSURE — PRE-SUBMISSION CHECK: Before finalizing each `detailed_caption`, perform this self-check sentence by sentence: for each sentence in the caption, identify which atom_id supports it. If no atom supports a sentence, that sentence must be deleted or a new atom must be created first. Temporal transition words (initially, later, then, moving forward, passing, as the sequence progresses) each require an explicit same-source atom that establishes the temporal change they describe. Inferential linkage words (corresponding, indicating, suggesting, implying, appear, seem) are STRICTLY FORBIDDEN. Motion of the observer (e.g. 'as we travel along') must not appear in `detailed_caption` (see Rule 8 for forbidden camera wording in global_scene); if forward motion is observable from physical cues, create a physical-world atom for it first.",
            "7. LENGTH: Meet minimum lengths using supported content only. Target 24-40 words for scene_summary and 35-60 words for each detailed_caption. Never add unsupported facts as filler.",
            "8. PHYSICAL WORDING & ENTITY CATEGORIES: Entity `category` MUST refer to a physical, tangible object class (e.g. vehicle, pedestrian, roadway, sidewalk, vegetation, infrastructure, building, animal, cyclist). For lighting phenomena (shadows, glare, reflections), use `lighting_effect` as category. Do NOT use 'barrier' for non-physical phenomena. For undetermined objects, use `unknown_object`.",
            "9. PHYSICAL-WORLD SUBJECT TEST & MODALITY-HIDDEN TEST (CRITICAL): Describe physical objects, attributes, structures, spatial relations, motion, temporal changes, and visibility directly. Do not describe how an image, channel, map, signal, response, or representation encodes or captures the scene. Raw source-specific response wording belongs only in `sensor_specific_cues`; generic sensor theory and inferential conclusions remain forbidden there. MODALITY-HIDDEN TEST: If a reader does not know the input modality, is this still a natural direct physical fact? If not, rewrite it or move only the raw cue to `sensor_specific_cues`.",
            "10. INTERNAL CONVERSION ORDER (Follow this mentally before writing atoms):\n   1. Identify the source-local visual cue.\n   2. Determine whether it supports a defensible physical-world proposition.\n   3. Write the physical proposition into atoms.\n   4. Put raw modality-response wording only in `sensor_specific_cues`.\n   5. If the cue cannot safely support a physical interpretation, do not invent one. A source-specific response pattern does not automatically justify a specific object identity, color, material, motion cause, or semantic class. When the current source supports only a coarse physical structure, write the coarse structure. When no defensible physical proposition can be formed, keep the raw segment-specific observation only in `sensor_specific_cues` or omit it. Do not borrow the interpretation from the opposite source.",
            "11. SENSORS & UNCERTAINTY: `sensor_specific_cues` is the ONLY place where mechanism-oriented response patterns are permitted; do not use a cue as evidence for an unsupported semantic interpretation. `sensor_limitations` must describe a limitation specifically manifested in this segment's frames — apply the GENERICITY TEST: if the sentence would be equally true for any segment recorded by the same modality, it is forbidden generic theory. FORBIDDEN examples: 'The modality lacks intensity data.', 'The sensor does not record color.', 'Lack of intensity data prevents identifying X.' REQUIRED form — cite what is absent in these specific frames: e.g. 'Vehicle surfaces in the sampled frames show no stable internal detail sufficient to distinguish paint color.' or 'Sunlight glare on windshields in frames 450–870 reduces surface detail.' Uncertain hypotheses must arise from current-source evidence and MUST NOT contradict a same-source Atom. If an Atom establishes a coarse category, uncertainty may concern only finer compatible categories. Do not claim that illumination, color, or object identity is missing when a same-source Atom already asserts it; weaken the Atom instead when the source does not support that assertion. CONSISTENCY TEST: For each uncertainty item, temporarily read only the cited same-source Atoms. Do those Atoms already resolve any property listed as missing or uncertain? If yes, either remove the uncertainty or weaken the Atom.",
            "12. MISSING ATTRIBUTES — PASS 2 RESOLUTION TARGETS: Every item requires exactly one explicit known `entity_id`; never infer the target from prose. `missing_attribute` names the unknown PROPERTY, not an unknown concrete value. `why_missing` must be SEGMENT-LOCAL: cite what is absent IN THESE SPECIFIC FRAMES that prevents observing the attribute. Apply the GENERICITY TEST: if the sentence would be equally true for any segment recorded by the same modality regardless of content, it is forbidden generic theory. Include a target only when the opposite source has explicit Entity-connected Atom evidence that directly supports the missing property and resolving it would matter; modality capability alone is never enough. Do not propose exact make/model unless the opposite source contains direct distinguishing evidence; coarse shape is not make/model evidence. Do not propose license-plate characters unless the opposite source visibly contains readable characters; an outer outline is not license-plate-text evidence. Empty missing lists are preferable to unrecoverable targets, and lists are never forced to be non-empty.\n   JSON CONTRACT — exactly five fields:\n     {\n       \"entity_id\": \"entity_004\",\n       \"attribute_type\": \"<one allowed enum>\",\n       \"missing_attribute\": \"<unknown property, neutrally worded>\",\n       \"why_missing\": \"<non-empty segment-local explanation>\",\n       \"recoverable_evidence_refs\": []\n     }\n   `entity_id` must exist in the shared Registry. `recoverable_evidence_refs` remains exactly [] in Pass 1. Nulls, empty strings, placeholders, partial objects, extra fields, and the key `missing_attribute_type` are forbidden.",
            "13. `recoverable_evidence_refs` MUST remain [] in PASS 1.",
            (
                "PRE-SUBMISSION SELF-CHECK (MANDATORY — complete this check mentally before finalising your JSON output):\n"
                "Before you write your final JSON, go through every item below. If any item fails, fix it before submitting.\n"
                "\n"
                "SCENE & TEXT FIELDS:\n"
                "  [ ] global_scene.scene_summary and temporal_progression: no \"camera\", \"lens\", \"viewpoint\", \"frame\"\n"
                "  [ ] ALL text fields (including atoms and captions): no \"corresponding\", \"suggesting\", \"indicating\", \"implying\"\n"
                "  [ ] detailed captions and atoms contain NO mechanism-oriented wording (e.g. activations, response maps, event activity)\n"
                "\n"
                "ATOMS:\n"
                "  [ ] Every atom under video1_analysis has atom_id starting with 'v1_atom_'.\n"
                "  [ ] Every atom under video2_analysis has atom_id starting with 'v2_atom_'.\n"
                "  [ ] Every atom has a non-empty frame_keys list containing only the frame names provided in this prompt.\n"
                "  [ ] Every atom has a non-empty entity_refs list. An atom with zero entity_refs is a hard schema violation. "
                "If a fact describes a scene-level condition with no specific entity, either assign it to the most relevant "
                "existing entity_id, or create a new entity entry in global_scene.physical_entities first.\n"
                "  [ ] ENTITY LOCKDOWN: For every physical object named by noun phrase in ANY text field — including `fact`, "
                "`missing_attribute`, `observed_evidence`, `missing_evidence`, and `sensor_limitations` — there MUST be a "
                "corresponding entity_id in global_scene.physical_entities. If you referenced a physical object in any of "
                "these fields but have no entity for it, ADD the entity to physical_entities NOW before finalising. "
                "Do not leave unnamed physical referents in any text field.\n"
                "\n"
                "CAPTIONS:\n"
                "  DEFINITION: detailed_caption is a lossy prose compression of information_atoms. It must contain LESS "
                "information than the atoms combined, never more. Any word or phrase that does not map back to a specific "
                "atom_id is a violation.\n"
                "  [ ] Scan each sentence of detailed_caption in video1_analysis. For each sentence, name the atom_id that "
                "supports it. If no atom supports a sentence, delete the sentence or create a new atom first.\n"
                "  [ ] Scan each sentence of detailed_caption in video2_analysis. Same rule as above.\n"
                "  [ ] detailed_caption in BOTH video analyses must NOT contain any of these words (even in non-inferential use): "
                "appear, appears, appeared, seem, seems, seemingly, apparently, suggesting, suggested, indicating, indicated, "
                "implying, implied, corresponding, consequently, therefore, representing, associated.\n"
                "  SUBSTITUTION GUIDE — replace forbidden words as follows:\n"
                "    'X appears in frame N'        → 'X is visible in frame N'\n"
                "    'X appears to be Y'           → 'X is Y' (only if an atom supports it) or drop the claim\n"
                "    'indicating forward motion'   → 'establishing forward motion' or cite the physical atom directly\n"
                "    'suggesting the surface is X' → 'the surface shows X' (only if an atom supports it)\n"
                "  SYNONYM DISCIPLINE — caption word choice must match atom word choice exactly where possible. Common violations to avoid:\n"
                "    'ground'    → use the atom's wording: 'asphalt', 'road surface', 'sidewalk', etc.\n"
                "    'displays'  → 'has', 'shows', 'contains' only if atom uses those words; otherwise rewrite the sentence using the atom's exact phrasing\n"
                "    'lined with'→ rephrase using atom facts: e.g. 'a white sedan, a black van ... are parked on the street' (enumerate from atoms)\n"
                "  OBSERVER MOTION — these phrases are FORBIDDEN in detailed_caption even if an atom supports forward movement:\n"
                "    FORBIDDEN: 'Moving forward', 'As we move', 'moving along', 'as the path advances'\n"
                "    ALLOWED:   Use the physical-world atom wording directly, e.g. 'Further along the street, a white sedan is parked on the right.'\n"
                "\n"
                "SCHEMA:\n"
                "  [ ] Every uncertain_observations item must contain a valid uncertainty_id, one known entity_id, and at least one same-source evidence_ref connected to that entity. hypotheses may be empty; if non-empty, they must contain at least 2 distinct candidate interpretations.\n"
                "  [ ] recoverable_evidence_refs is [] for every missing_key_attributes entry.\n"
                "  [ ] No top-level fields other than global_scene, video1_analysis, video2_analysis.\n"
                "  [ ] global_scene contains no fields other than scene_summary, environment, temporal_progression, "
                "physical_entities."
            ),
            (
                "FINAL HARD CONSTRAINTS (checked after generation):\n"
                "- global_scene.scene_summary / temporal_progression: NO \"camera\", \"lens\", \"viewpoint\", \"frame\"\n"
                "- ALL fields: NO \"corresponding\", \"suggesting\", \"indicating\", \"implying\""
            ),
            "OUTPUT CONTRACT: Return ONLY valid JSON matching the provided skeleton. No other top-level fields.",
        ]
    )


def build_pass1_user_prompt(task: Any, modality1: str | None = None, modality2: str | None = None) -> str:
    modality1 = normalize_modality_name(modality1 if modality1 is not None else task.modality1)
    modality2 = normalize_modality_name(modality2 if modality2 is not None else task.modality2)
    frame_names = ", ".join(path.name for path in task.composite_frames)
    constraint_block = build_modality_constraint_block(modality1, modality2)
    dynamic_guidance = build_modality_physical_guidance(modality1, modality2)
    schema_example_text = json.dumps(_build_prompt_schema_example(task, modality1, modality2), indent=2, ensure_ascii=False)

    return "\n".join(
        [
            "TASK: Construct PASS 1 evidence for an aligned multimodal video segment. Neither source is ground truth.",
            "SCOPE: Produce ONLY global_scene, video1_analysis, and video2_analysis. NO cross-modal fusion, information-gain, directional contribution, cross-source ambiguity resolution, QA, or recoverability reasoning.",
            "PRIORITIES: 1. source-local factual correctness; 2. valid provenance/references; 3. stable entity identity; 4. evidence coverage (do not omit salient observations; every independently distinguishable physical object in any frame that meets the minimum-size threshold MUST have an entity in global_scene.physical_entities); 5. atom proposition integrity (each atom = 1 minimal proposition, but no upper limit on atom count); 6. caption completeness (every caption sentence must be supported by a same-source atom); 7. descriptive richness.",
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
