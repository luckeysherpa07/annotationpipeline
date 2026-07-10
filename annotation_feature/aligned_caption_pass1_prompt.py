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
            "scene_summary": "A stationary vehicle is situated near a concrete barrier, maintaining its position relative to the road markings while casting a distinct, sharp shadow across the adjacent paved surface under the ambient urban lighting conditions.",
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
                },
                {
                    "entity_id": "entity_003",
                    "category": "lighting_effect",
                    "referential_scope": "shadow of the barrier cast on the ground"
                }
            ]
        },
        "video1_analysis": {
            "modality": task.modality1,
            "detailed_caption": "A stationary vehicle rests beside a concrete barrier. The vehicle's surface shows various textures, and a sharp shadow of the barrier falls across the road surface, emphasizing the spatial boundaries in this specific urban segment.",
            "information_atoms": [
                {
                    "atom_id": "v1_atom_001",
                    "frame_keys": [fk],
                    "entity_refs": ["entity_001", "entity_002"],
                    "fact": "A stationary vehicle rests beside a concrete barrier."
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
                    "hypotheses": []
                }
            ],
            "missing_key_attributes": [
                {
                    "attribute_type": "surface_attribute",
                    "missing_attribute": "Material of entity_001.",
                    "why_missing": "Surface texture is not visible in this segment's lighting.",
                    "recoverable_evidence_refs": []
                }
            ]
        },
        "video2_analysis": {
            "modality": task.modality2,
            "detailed_caption": "A concrete barrier is visible. Edge contours and distinct structural outlines define the shape of the barrier and the surrounding cobblestone texture, providing high-frequency details of the static physical boundaries within the scene's view.",
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
            "PRIORITIES: 1. source-local factual correctness; 2. valid provenance/references; 3. stable entity identity; 4. evidence coverage (do not omit salient observations; every independently distinguishable physical object in any frame that meets the minimum-size threshold MUST have an entity in global_scene.physical_entities); 5. atom proposition integrity (each atom = 1 minimal proposition, but no upper limit on atom count); 6. caption completeness (every caption sentence must be supported by a same-source atom); 7. descriptive richness.",
            constraint_block,
            "RULES:",
            "1. SHARED ENTITIES: Reuse `entity_id` for the same physical referent. Never reuse an ID for different referents. Create entities for salient independent physical referents. Preserve minimality; do not hide genuinely salient referents. ENTITY COMPLETENESS SCAN (MANDATORY): Before writing any atom, scan ALL visible physical objects in every supplied frame and enumerate a candidate entity for each. Apply this minimum bar: if an object is (a) large enough to occupy more than ~5% of frame area, AND (b) can be identified to at least a coarse category (e.g. vehicle, building, vegetation), it MUST receive an entity_id. Common omissions to watch for: multiple parked vehicles counted as one, background buildings ignored, trees and utility poles missed. Do NOT merge distinct physical objects into a single entity merely for brevity.",
            "2. REGISTRY METADATA: Global registry metadata is organizational, not source-local evidence. Use conservative but useful identity descriptions. Do not automatically over-neutralize every referent to 'object' or 'unknown', but do not insert source-exclusive details that create leakage risk.",
            "3. ISOLATION & LEAKAGE: Cross-modal leakage occurs ONLY when a claim is not independently supported by the current source. For each source, use the strongest description independently justified by that source: no stronger, but also no weaker merely because another source supports the same interpretation. SOURCE-LOCAL TEST: Judge each claim using only the current source. Hide the opposite source and treat shared registry metadata as non-evidence. If the claim remains reasonably supported, it is allowed.",
            "4. CROSS-MODAL DIFFERENTIATION — REQUIRED: video1_analysis and video2_analysis MUST each contain at least one atom whose fact is NOT shared by the other modality's atoms. The two modalities are complementary physical instruments; they DO NOT observe the same physical events in the same way. For each modality, look for what is EXCLUSIVELY visible or exclusively absent in this specific source. FORBIDDEN: An atom that is textually near-identical to an atom in the other modality's block. If you find yourself writing the same physical fact for both modalities, one of them is wrong — one modality must contribute a unique exclusive observation that the other modality cannot provide. Refer to the CROSS-MODAL DIFFERENTIATION GUIDE above for a concrete list of what each modality in this segment is exclusively able to capture.",
            "5. ATOMS FIRST — HARD RULE: Build all `information_atoms` before writing `detailed_caption`. There is NO upper limit on atom count; do not reduce atoms for brevity. For every semantic claim you want to include in `detailed_caption`, apply this binary decision: (A) An existing same-source atom directly supports this claim → you may include it in the caption. (B) No atom supports it → you MUST either create a minimal new atom first and then include the claim, OR drop the claim from the caption entirely. A claim present in `detailed_caption` but absent from same-source `information_atoms` is a HARD VIOLATION. Each atom = 1 minimal physical proposition. Unique `atom_id` (v1_atom_/v2_atom_ prefix), valid `frame_keys`, non-empty `entity_refs`. Prefer physical-world wording over mechanism-oriented phrasing.",
            "6. CAPTION CLOSURE — PRE-SUBMISSION CHECK: Before finalizing each `detailed_caption`, perform this self-check sentence by sentence: for each sentence in the caption, identify which atom_id supports it. If no atom supports a sentence, that sentence must be deleted or a new atom must be created first. Temporal transition words (initially, later, then, moving forward, passing, as the sequence progresses) each require an explicit same-source atom that establishes the temporal change they describe. Inferential linkage words (corresponding, indicating, suggesting, implying, appear, seem) are STRICTLY FORBIDDEN. Motion of the observer (e.g. 'as we travel along') must not appear in `detailed_caption` (see Rule 8 for forbidden camera wording in global_scene); if forward motion is observable from physical cues, create a physical-world atom for it first.",
            "7. LENGTH: Meet minimum lengths using supported content only. Target 24-40 words for scene_summary and 35-60 words for each detailed_caption. Never add unsupported facts as filler.",
            "8. PHYSICAL WORDING & ENTITY CATEGORIES: Entity `category` MUST refer to a physical, tangible object class (e.g. vehicle, pedestrian, roadway, sidewalk, vegetation, infrastructure, building, animal, cyclist). For lighting phenomena (shadows, glare, reflections), use `lighting_effect` as category. Do NOT use 'barrier' for non-physical phenomena. For undetermined objects, use `unknown_object`. `detailed_caption`, `global_scene`, and `information_atoms` must describe the physical world as observed from within the scene. Forbidden words for ALL fields: modality, rgb, ir, event, depth, edge map, blurry, noisy, pixel, grayscale, sensor, capture, footage, stream, recording. GLOBAL SCENE FIELDS: additionally forbidden: camera, lens, viewpoint, frame. Instead of 'A camera moves forward', write 'The perspective advances along the street' or 'The scene unfolds from a forward-moving vantage point'. Do not use 'the sensor records', 'generates activations', or any modality-mechanism explanation.",
            "9. SENSORS & UNCERTAINTY: `sensor_specific_cues` may describe response patterns but must remain segment-specific; do not use a cue as evidence for an unsupported semantic interpretation. `sensor_limitations` must describe a limitation specifically manifested in this segment's frames — apply the GENERICITY TEST: if the sentence would be equally true for any segment recorded by the same modality, it is forbidden generic theory. FORBIDDEN examples: 'The modality lacks intensity data.', 'The sensor does not record color.', 'Lack of intensity data prevents identifying X.' REQUIRED form — cite what is absent in these specific frames: e.g. 'Vehicle surfaces in the sampled frames show no stable internal detail sufficient to distinguish paint color.' or 'Sunlight glare on windshields in frames 450–870 reduces surface detail.' Uncertain hypotheses must arise from current-source evidence.",
            "10. MISSING ATTRIBUTES: `why_missing` must be SEGMENT-LOCAL: cite what is absent IN THESE SPECIFIC FRAMES that prevents observing the attribute. Apply the GENERICITY TEST: if the sentence would be equally true for any segment recorded by the same modality regardless of content, it is forbidden generic theory.\n   FORBIDDEN form: 'Surface color cannot be observed because this sensing method does not register static spectral differences.' (= generic theory about the modality)\n   FORBIDDEN form: 'The physical instrument does not register static surface paint reflectance.' (= same problem, disguised wording)\n   REQUIRED form: 'The vehicle body in frames 450–480 is partially occluded by shadow, preventing identification of paint color.' or 'Distant vehicles in the background frames show insufficient resolution to distinguish a specific color.' — these cite what is concretely absent in this segment.",
            "11. `recoverable_evidence_refs` MUST remain [] in PASS 1.",
            (
                "PRE-SUBMISSION SELF-CHECK (MANDATORY — complete this check mentally before finalising your JSON output):\n"
                "Before you write your final JSON, go through every item below. If any item fails, fix it before submitting.\n"
                "\n"
                "SCENE & TEXT FIELDS:\n"
                "  [ ] global_scene.scene_summary and temporal_progression: no \"camera\", \"lens\", \"viewpoint\", \"frame\"\n"
                "  [ ] ALL text fields (including atoms and captions): no \"corresponding\", \"suggesting\", \"indicating\", \"implying\"\n"
                "\n"
                "ATOMS:\n"
                "  [ ] Every atom under video1_analysis has atom_id starting with 'v1_atom_'.\n"
                "  [ ] Every atom under video2_analysis has atom_id starting with 'v2_atom_'.\n"
                "  [ ] Every atom has a non-empty frame_keys list containing only the frame names provided in this prompt.\n"
                "  [ ] Every atom has a non-empty entity_refs list. An atom with zero entity_refs is a hard schema violation. "
                "If a fact describes a scene-level condition with no specific entity, either assign it to the most relevant "
                "existing entity_id, or create a new entity entry in global_scene.physical_entities first.\n"
                "\n"
                "CAPTIONS:\n"
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
                    "uncertainty_id": "v1_unc_001",
                    "entity_id": target_entity,
                    "observed_evidence": "Template placeholder",
                    "missing_evidence": "Placeholder",
                    "evidence_refs": ["v1_atom_001"],
                    "hypotheses": []
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
                    "uncertainty_id": "v2_unc_001",
                    "entity_id": target_entity,
                    "observed_evidence": "Template placeholder",
                    "missing_evidence": "Placeholder",
                    "evidence_refs": ["v2_atom_001"],
                    "hypotheses": []
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
