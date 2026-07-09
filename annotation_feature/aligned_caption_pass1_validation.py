"""Semantic validation for Pass 1 (evidence construction) of the aligned multimodal caption pipeline."""

from __future__ import annotations

import re
from typing import Any

from annotation_feature.aligned_caption_schema import (
    ALLOWED_MISSING_ATTRIBUTE_TYPES,
    CaptionValidationError,
    FORBIDDEN_SENSOR_QUALITY_MESSAGE,
    FORBIDDEN_SENSOR_QUALITY_PATTERN,
    MIN_DETAILED_CAPTION_WORDS,
    MIN_FRAME_DETAIL_WORDS,
    MIN_SCENE_SUMMARY_WORDS,
    _normalize_license_plates,
)

from annotation_feature.aligned_caption_validation import (
    _require_object,
    _require_list,
    _require_string,
    _validate_min_words,
    _validate_no_generic_sensor_explanation,
    _validate_uncertain_observations,
    _normalize_referential_scope,
)

PASS1_REQUIRED_TOP_LEVEL_FIELDS = {
    "global_scene",
    "video1_analysis",
    "video2_analysis",
}

PASS1_GLOBAL_SCENE_ALLOWED_FIELDS = {
    "scene_summary",
    "environment",
    "temporal_progression",
    "physical_entities",
}

PASS1_VIDEO_ANALYSIS_ALLOWED_FIELDS = {
    "modality",
    "detailed_caption",
    "information_atoms",
    "sensor_specific_cues",
    "sensor_limitations",
    "uncertain_observations",
    "missing_key_attributes",
}


def _normalize_pass1_hypothesis(text: str) -> str:
    t = text.casefold()
    t = t.rstrip(".!? ")
    t = " ".join(t.split())
    return t


def _validate_pass1_uncertain_observations(values: Any, field: str) -> None:
    _validate_uncertain_observations(values, field)
    for index, item in enumerate(values or [], start=1):
        if not isinstance(item, dict): continue
        hypotheses = item.get("hypotheses") or []
        normalized_hyps: set[str] = set()
        for hyp in hypotheses:
            if not isinstance(hyp, dict): continue
            hyp_text = hyp.get("hypothesis")
            if isinstance(hyp_text, str) and hyp_text.strip():
                norm_meta = hyp_text.strip().casefold()
                # Check for meta-uncertainty
                if re.search(r'\b(cannot be determined|is unknown|more evidence is needed|cannot be definitively determined)\b', norm_meta):
                    raise CaptionValidationError(f"{field}[{index}] contains meta-statement of inability: '{hyp_text}'. Hypotheses must be candidate interpretations, not meta-statements.")
                norm = _normalize_pass1_hypothesis(hyp_text)
                normalized_hyps.add(norm)
        if len(normalized_hyps) < 2:
            raise CaptionValidationError(f"{field}[{index}].hypotheses must contain at least 2 distinct valid candidate hypotheses in Pass 1")


def _validate_pass1_why_missing(text: str, field: str) -> None:
    _validate_no_generic_sensor_explanation(text, field)
    if re.search(r'\b(sensing )?modality does not (record|capture|detect|provide)\b', text, re.I):
        raise CaptionValidationError(f"{field} must be segment-specific, not generic sensor theory (found forbidden wording).")
    
    generic_process_patterns = [
        re.compile(r"\b(the|this)\s+(sensing|imaging)\s+process\b", re.I),
        re.compile(r"\bthe\s+sensor\s+(captures|records)\b", re.I),
        re.compile(r"\bdoes\s+not\s+preserve\s+(static\s+)?color\b", re.I),
        re.compile(r"\bimaging\s+process\s+records\b", re.I),
        re.compile(r"\bsensor\s+captures\s+changes\b", re.I)
    ]
    for pat in generic_process_patterns:
        if pat.search(text):
            raise CaptionValidationError(
                f"{field} must be segment-specific, not generic sensor theory (found generic explanation pattern: '{pat.pattern}')."
            )


def _validate_pass1_schema(
    parsed: dict[str, Any],
    valid_frame_keys: set[str],
    expected_modality1: str,
    expected_modality2: str,
) -> tuple[dict[str, Any], list[str]]:
    if not valid_frame_keys:
        raise CaptionValidationError("valid_frame_keys must not be empty for validation")

    atom_frame_keys: dict[str, set[str]] = {}
    atom_facts: dict[str, str] = {}
    atom_entity_refs: dict[str, set[str]] = {}
    local_warnings: list[str] = []

    missing = [field for field in PASS1_REQUIRED_TOP_LEVEL_FIELDS if field not in parsed]
    if missing:
        raise CaptionValidationError(f"Gemini response missing required Pass 1 field(s): {', '.join(missing)}")
        
    unexpected_fields = set(parsed.keys()) - PASS1_REQUIRED_TOP_LEVEL_FIELDS
    if unexpected_fields:
        raise CaptionValidationError(f"Gemini response contains unknown top-level fields for Pass 1: {', '.join(sorted(unexpected_fields))}. Allowed fields are only: {', '.join(PASS1_REQUIRED_TOP_LEVEL_FIELDS)}")

    evidence_namespace: set[str] = set()

    def _register_evidence_id(eid: str) -> None:
        if eid in evidence_namespace:
            raise CaptionValidationError(f"Duplicate evidence ID found: {eid}. Evidence IDs must be globally unique.")
        evidence_namespace.add(eid)

    global_scene = _require_object(parsed["global_scene"], "global_scene")
    
    global_scene_unexpected = set(global_scene.keys()) - PASS1_GLOBAL_SCENE_ALLOWED_FIELDS
    if global_scene_unexpected:
        raise CaptionValidationError(f"global_scene contains unknown fields for Pass 1: {', '.join(sorted(global_scene_unexpected))}. Fields like reasoning_focus_entities are forbidden in Pass 1.")

    scene_summary = _validate_min_words(global_scene.get("scene_summary"), "global_scene.scene_summary", MIN_SCENE_SUMMARY_WORDS)
    if FORBIDDEN_SENSOR_QUALITY_PATTERN.search(scene_summary):
        raise CaptionValidationError(f"global_scene.scene_summary contains forbidden sensor-quality wording. {FORBIDDEN_SENSOR_QUALITY_MESSAGE}")
    _validate_no_generic_sensor_explanation(scene_summary, "global_scene.scene_summary")
    _require_string(global_scene.get("environment"), "global_scene.environment")
    temporal_progression = _validate_min_words(global_scene.get("temporal_progression"), "global_scene.temporal_progression", MIN_FRAME_DETAIL_WORDS)
    if FORBIDDEN_SENSOR_QUALITY_PATTERN.search(temporal_progression):
        raise CaptionValidationError(f"global_scene.temporal_progression contains forbidden sensor-quality wording. {FORBIDDEN_SENSOR_QUALITY_MESSAGE}")
    _validate_no_generic_sensor_explanation(temporal_progression, "global_scene.temporal_progression")
    
    physical_entities = _require_list(global_scene.get("physical_entities"), "global_scene.physical_entities")
    if not physical_entities:
        raise CaptionValidationError("global_scene.physical_entities must not be empty")
    entity_ids: set[str] = set()
    normalized_entity_scopes: dict[str, str] = {}
    for index, entity in enumerate(physical_entities, start=1):
        if not isinstance(entity, dict):
            raise CaptionValidationError(f"global_scene.physical_entities[{index}] must be an object")
        entity_id = _require_string(entity.get("entity_id"), f"global_scene.physical_entities[{index}].entity_id")
        if entity_id in entity_ids:
            raise CaptionValidationError(f"Duplicate entity_id: {entity_id}")
        entity_ids.add(entity_id)
        _require_string(entity.get("category"), f"global_scene.physical_entities[{index}].category")
        referential_scope = _require_string(entity.get("referential_scope"), f"global_scene.physical_entities[{index}].referential_scope")
        normalized_scope = _normalize_referential_scope(referential_scope)
        existing_entity_id = normalized_entity_scopes.get(normalized_scope)
        if existing_entity_id is not None and existing_entity_id != entity_id:
            raise CaptionValidationError(
                f"Duplicate normalized referential_scope detected for "
                f"{entity_id} and {existing_entity_id}: {referential_scope!r}"
            )
        normalized_entity_scopes[normalized_scope] = entity_id
        if "evidence_profile" in entity:
            prof = _require_object(entity.get("evidence_profile"), f"global_scene.physical_entities[{index}].evidence_profile")
            if not prof:
                raise CaptionValidationError("evidence_profile must not be empty if present.")
            for prof_key in ("identity_evidence", "observable_attributes", "spatial_context"):
                if prof_key in prof:
                    ev_list = _require_list(prof[prof_key], f"global_scene.physical_entities[{index}].evidence_profile.{prof_key}")
                    if not ev_list:
                        raise CaptionValidationError(f"evidence_profile.{prof_key} must not be empty if present.")
                    for j, s in enumerate(ev_list, start=1):
                        _require_string(s, f"evidence_profile.{prof_key}[{j}]")

    video1_analysis = _require_object(parsed.get("video1_analysis"), "video1_analysis")
    video2_analysis = _require_object(parsed.get("video2_analysis"), "video2_analysis")

    modality1 = _require_string(video1_analysis.get("modality"), "video1_analysis.modality")
    if modality1 != expected_modality1:
        raise CaptionValidationError(f"video1_analysis.modality {modality1!r} does not match expected {expected_modality1!r}")
    
    modality2 = _require_string(video2_analysis.get("modality"), "video2_analysis.modality")
    if modality2 != expected_modality2:
        raise CaptionValidationError(f"video2_analysis.modality {modality2!r} does not match expected {expected_modality2!r}")

    def _validate_video_analysis(parsed_obj: dict[str, Any], field: str, atom_prefix: str) -> None:
        analysis = _require_object(parsed_obj.get(field), field)
        
        analysis_unexpected = set(analysis.keys()) - PASS1_VIDEO_ANALYSIS_ALLOWED_FIELDS
        if analysis_unexpected:
            raise CaptionValidationError(f"{field} contains unknown fields for Pass 1: {', '.join(sorted(analysis_unexpected))}.")

        _require_string(analysis.get("modality"), f"{field}.modality")
        detailed_caption = _validate_min_words(
            analysis.get("detailed_caption"),
            f"{field}.detailed_caption",
            MIN_DETAILED_CAPTION_WORDS,
        )
        if FORBIDDEN_SENSOR_QUALITY_PATTERN.search(detailed_caption):
            raise CaptionValidationError(f"{field}.detailed_caption contains forbidden sensor-quality wording. {FORBIDDEN_SENSOR_QUALITY_MESSAGE}")
        _validate_no_generic_sensor_explanation(detailed_caption, f"{field}.detailed_caption")
        
        atoms = _require_list(analysis.get("information_atoms"), f"{field}.information_atoms")
        if not atoms:
            raise CaptionValidationError(f"{field}.information_atoms must not be empty")
            
        for i, atom in enumerate(atoms, start=1):
            if not isinstance(atom, dict):
                raise CaptionValidationError(f"{field}.information_atoms[{i}] must be an object")
            atom_id = _require_string(atom.get("atom_id"), f"{field}.information_atoms[{i}].atom_id")
            if not atom_id.startswith(atom_prefix):
                raise CaptionValidationError(f"atom_id {atom_id} must start with {atom_prefix}")
            _register_evidence_id(atom_id)
            f_keys = _require_list(atom.get("frame_keys"), f"{field}.information_atoms[{i}].frame_keys")
            if not f_keys:
                raise CaptionValidationError(f"{field}.information_atoms[{i}].frame_keys cannot be empty")

            for fk in f_keys:
                if fk not in valid_frame_keys:
                    raise CaptionValidationError(f"Unknown frame_key '{fk}' in {atom_id}")
            atom_frame_keys[atom_id] = set(f_keys)
            entity_refs = _require_list(atom.get("entity_refs"), f"{field}.information_atoms[{i}].entity_refs")
            if not entity_refs:
                raise CaptionValidationError(f"{field}.information_atoms[{i}].entity_refs cannot be empty")
            seen_atom_entities: set[str] = set()
            for entity_ref_index, entity_ref in enumerate(entity_refs, start=1):
                ref_value = _require_string(entity_ref, f"{field}.information_atoms[{i}].entity_refs[{entity_ref_index}]")
                if ref_value in seen_atom_entities:
                    raise CaptionValidationError(f"{field}.information_atoms[{i}].entity_refs contains duplicate entity_id: {ref_value}")
                if ref_value not in entity_ids:
                    raise CaptionValidationError(f"{field}.information_atoms[{i}].entity_refs references unknown entity: {ref_value}")
                seen_atom_entities.add(ref_value)
            atom_entity_refs[atom_id] = seen_atom_entities
            fact = _require_string(atom.get("fact"), f"{field}.information_atoms[{i}].fact")
            atom_facts[atom_id] = fact

        for key in ("sensor_specific_cues", "sensor_limitations"):
            values = _require_list(analysis.get(key), f"{field}.{key}")
            for value_index, value in enumerate(values, start=1):
                item_field = f"{field}.{key}[{value_index}]"
                text = _require_string(value, item_field)
                if key == "sensor_limitations":
                    _validate_no_generic_sensor_explanation(
                        text,
                        item_field,
                        hard_fail=False,
                        warnings=local_warnings,
                    )
                else:
                    _validate_no_generic_sensor_explanation(text, item_field)
        _validate_pass1_uncertain_observations(analysis.get("uncertain_observations"), f"{field}.uncertain_observations")
        
        missing_attrs = _require_list(analysis.get("missing_key_attributes"), f"{field}.missing_key_attributes")
        for i, attr in enumerate(missing_attrs, start=1):
            if not isinstance(attr, dict):
                raise CaptionValidationError(f"{field}.missing_key_attributes[{i}] must be an object")
            attr_type = attr.get("attribute_type")
            if attr_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
                raise CaptionValidationError(f"{field}.missing_key_attributes[{i}].attribute_type invalid: {attr_type}")
            _require_string(attr.get("missing_attribute"), f"{field}.missing_key_attributes[{i}].missing_attribute")
            _require_string(attr.get("why_missing"), f"{field}.missing_key_attributes[{i}].why_missing")
            _validate_pass1_why_missing(attr.get("why_missing"), f"{field}.missing_key_attributes[{i}].why_missing")
            
            recoverable_refs = _require_list(attr.get("recoverable_evidence_refs"), f"{field}.missing_key_attributes[{i}].recoverable_evidence_refs")
            if recoverable_refs:
                raise CaptionValidationError(f"{field}.missing_key_attributes[{i}].recoverable_evidence_refs MUST be empty in Pass 1. Found: {recoverable_refs}")

        # Validator C: unimodal uncertainty consistency
        for i, obs in enumerate(analysis.get("uncertain_observations") or [], start=1):
            if not isinstance(obs, dict): continue
            for hyp_dict in obs.get("hypotheses") or []:
                if not isinstance(hyp_dict, dict): continue
                hyp_text = str(hyp_dict.get("hypothesis", "")).strip()
                if len(hyp_text.split()) >= 2 and hyp_text.lower() in detailed_caption.lower():
                    raise CaptionValidationError(
                        f"{field}.detailed_caption presents uncertain hypothesis '{hyp_text}' as fact. "
                        "Rewrite the caption to use neutral perceptual language."
                    )
                    
        # Validator D: caption-to-atom grounding (soft warning)
        caption_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', detailed_caption.lower()))
        atom_text = " ".join([str(a.get("fact", "")) for a in atoms if isinstance(a, dict)]).lower()
        atom_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', atom_text))
        ungrounded = caption_words - atom_words
        if len(ungrounded) > 10:
             local_warnings.append(f"{field}.detailed_caption may contain ungrounded claims. Words not in atoms: {', '.join(sorted(list(ungrounded))[:5])}...")

    _validate_video_analysis(parsed, "video1_analysis", "v1_atom_")
    _validate_video_analysis(parsed, "video2_analysis", "v2_atom_")
    
    return parsed, local_warnings
