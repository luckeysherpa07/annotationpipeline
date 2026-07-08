"""Semantic validation for aligned multimodal caption JSON."""

from __future__ import annotations

import re
from typing import Any

from annotation_feature.aligned_caption_schema import (
    ALLOWED_AMBIGUITY_DIRECTIONS,
    ALLOWED_CROSS_MODAL_CONTRIBUTION_DIRECTIONS,
    ALLOWED_GAIN_RATINGS,
    ALLOWED_GAIN_TYPES,
    ALLOWED_MISSING_ATTRIBUTE_TYPES,
    ALLOWED_QA_REASONING_PATTERNS,
    ALLOWED_REASONING_EVENT_TYPES,
    CAPTION_REQUIRED_TOP_LEVEL_FIELDS,
    CaptionValidationError,
    FORBIDDEN_SENSOR_QUALITY_MESSAGE,
    FORBIDDEN_SENSOR_QUALITY_PATTERN,
    GENERIC_SENSOR_EXPLANATION_PATTERNS,
    MIN_AMBIGUITY_EVENT_HYPOTHESES,
    MIN_DETAILED_CAPTION_WORDS,
    MIN_FRAME_DETAIL_WORDS,
    MIN_SCENE_SUMMARY_WORDS,
    MIN_UNCERTAIN_OBSERVATION_HYPOTHESES,
    MODALITY_CAPABILITIES,
    _normalize_license_plates,
)

def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaptionValidationError(f"Gemini response field {field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CaptionValidationError(f"Gemini response field {field} must be a list")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptionValidationError(f"Gemini response field {field} must be a non-empty string")
    return value


def _validate_string_list(value: Any, field: str, *, allow_empty: bool = True) -> list[str]:
    items = _require_list(value, field)
    if not allow_empty and not items:
        raise CaptionValidationError(f"{field} must not be empty")
    for i, item in enumerate(items, start=1):
        _require_string(item, f"{field}[{i}]")
    return items


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _validate_min_words(text: Any, field: str, minimum: int) -> str:
    value = _require_string(text, field)
    if _word_count(value) < minimum:
        raise CaptionValidationError(f"{field} is too short; expected at least {minimum} words")
    return value


def _validate_no_generic_sensor_explanation(
    text: str,
    field: str,
    *,
    hard_fail: bool = True,
    warnings: list[str] | None = None,
) -> None:
    for pattern in GENERIC_SENSOR_EXPLANATION_PATTERNS:
        if not pattern.search(text):
            continue

        if hard_fail:
            raise CaptionValidationError(
                f"{field} contains generic sensor-theory wording instead of segment-specific evidence"
            )
        if warnings is not None:
            warnings.append(
                f"{field} may contain generic sensor-theory wording; "
                "check whether the statement is sufficiently segment-specific"
            )
        return


def _validate_uncertain_observations(values: Any, field: str) -> None:
    for index, item in enumerate(_require_list(values, field), start=1):
        if not isinstance(item, dict):
            raise CaptionValidationError(f"{field}[{index}] must be an object")
        _require_string(item.get("observed_evidence"), f"{field}[{index}].observed_evidence")
        _require_string(item.get("missing_evidence"), f"{field}[{index}].missing_evidence")
        hypotheses = _require_list(item.get("hypotheses"), f"{field}[{index}].hypotheses")
        if len(hypotheses) < MIN_UNCERTAIN_OBSERVATION_HYPOTHESES:
            raise CaptionValidationError(
                f"{field}[{index}].hypotheses must contain at least "
                f"{MIN_UNCERTAIN_OBSERVATION_HYPOTHESES} hypothesis"
            )
        normalized_hyps: set[str] = set()
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            hyp_text = hypothesis.get("hypothesis")
            if isinstance(hyp_text, str) and hyp_text.strip():
                normalized_hyps.add(hyp_text.strip().casefold())
        if len(normalized_hyps) < MIN_UNCERTAIN_OBSERVATION_HYPOTHESES:
            raise CaptionValidationError(
                f"{field}[{index}].hypotheses must contain at least "
                f"{MIN_UNCERTAIN_OBSERVATION_HYPOTHESES} valid hypothesis"
            )
        for hyp_index, hyp in enumerate(hypotheses, start=1):
            if not isinstance(hyp, dict):
                raise CaptionValidationError(f"{field}[{index}].hypotheses[{hyp_index}] must be an object")
            _require_string(hyp.get("hypothesis"), f"{field}[{index}].hypotheses[{hyp_index}].hypothesis")
            conf = hyp.get("confidence")
            if conf not in ALLOWED_GAIN_RATINGS:
                raise CaptionValidationError(f"{field}[{index}].hypotheses[{hyp_index}].confidence must be high, medium, or low")

def _require_atom_entity_connection(
    ref: str,
    entity_id: str,
    atom_entity_refs: dict[str, set[str]],
    field: str,
) -> None:
    if entity_id not in atom_entity_refs.get(ref, set()):
        raise CaptionValidationError(
            f"{field} references atom {ref} that is not explicitly connected to entity {entity_id}"
        )


def _validate_source_local_atom_refs(
    refs: list[Any],
    field: str,
    prefix: str,
    evidence_namespace: set[str],
    atom_entity_refs: dict[str, set[str]],
    entity_id: str,
) -> list[str]:
    validated_refs: list[str] = []
    for ref_index, ref_value in enumerate(refs, start=1):
        ref = _require_string(ref_value, f"{field}[{ref_index}]")
        if not ref.startswith(prefix):
            raise CaptionValidationError(f"{field} must only contain {prefix} IDs")
        if ref not in evidence_namespace:
            raise CaptionValidationError(f"{field} references unknown atom: {ref}")
        _require_atom_entity_connection(ref, entity_id, atom_entity_refs, field)
        validated_refs.append(ref)
    return validated_refs


def _validate_cross_modal_evidence_links(
    values: Any,
    entity_ids: set[str],
    evidence_namespace: set[str],
    atom_entity_refs: dict[str, set[str]],
    field: str,
) -> None:
    seen_entities = set()
    links = _require_list(values, field)
    for index, item in enumerate(links, start=1):
        if not isinstance(item, dict):
            raise CaptionValidationError(f"{field}[{index}] must be an object")
        entity_id = _require_string(item.get("entity_id"), f"{field}[{index}].entity_id")
        if entity_id in seen_entities:
            raise CaptionValidationError(f"{field} contains duplicate entry for entity_id: {entity_id}")
        seen_entities.add(entity_id)
        if entity_id not in entity_ids:
            raise CaptionValidationError(f"{field}[{index}].entity_id must match a global_scene entity_id")
        
        refs_by_side: dict[str, list[str]] = {}
        for v_field, prefix in [("video1_evidence_refs", "v1_atom_"), ("video2_evidence_refs", "v2_atom_")]:
            ref_field = f"{field}[{index}].{v_field}"
            refs = _require_list(item.get(v_field), ref_field)
            refs_by_side[v_field] = _validate_source_local_atom_refs(
                refs,
                ref_field,
                prefix,
                evidence_namespace,
                atom_entity_refs,
                entity_id,
            )
        if not (refs_by_side["video1_evidence_refs"] or refs_by_side["video2_evidence_refs"]):
            raise CaptionValidationError(
                f"{field}[{index}] must cite at least one source-local evidence atom"
            )

        for key in ("shared_evidence", "unique_to_video1", "unique_to_video2"):
            _validate_string_list(item.get(key), f"{field}[{index}].{key}", allow_empty=True)
        
        directional = _require_list(item.get("directional_contributions", []), f"{field}[{index}].directional_contributions")
        for j, dc in enumerate(directional, start=1):
            if not isinstance(dc, dict):
                raise CaptionValidationError(f"{field}[{index}].directional_contributions[{j}] must be an object")
            direction = _require_string(dc.get("direction"), f"{field}[{index}].directional_contributions[{j}].direction")
            if direction not in ALLOWED_CROSS_MODAL_CONTRIBUTION_DIRECTIONS:
                raise CaptionValidationError(f"{field}[{index}].directional_contributions[{j}].direction must be one of {ALLOWED_CROSS_MODAL_CONTRIBUTION_DIRECTIONS}")
            _require_string(dc.get("contribution"), f"{field}[{index}].directional_contributions[{j}].contribution")
            
            has_v1 = bool(refs_by_side["video1_evidence_refs"])
            has_v2 = bool(refs_by_side["video2_evidence_refs"])
            if direction == "video1_improves_video2" and not has_v1:
                raise CaptionValidationError(f"{field}[{index}] direction='video1_improves_video2' requires Video 1 evidence")
            elif direction == "video2_improves_video1" and not has_v2:
                raise CaptionValidationError(f"{field}[{index}] direction='video2_improves_video1' requires Video 2 evidence")
            elif direction in ("confirmation_only", "mutual_complementarity") and not (has_v1 and has_v2):
                raise CaptionValidationError(f"{field}[{index}] direction='{direction}' requires evidence from both videos")

def _validate_information_gain(
    values: Any,
    entity_ids: set[str],
    evidence_namespace: set[str],
    atom_entity_refs: dict[str, set[str]],
    field: str,
) -> None:
    seen_entities = set()
    gains = _require_list(values, field)
    for index, item in enumerate(gains, start=1):
        if not isinstance(item, dict):
            raise CaptionValidationError(f"{field}[{index}] must be an object")
        entity_id = _require_string(item.get("entity_id"), f"{field}[{index}].entity_id")
        if entity_id in seen_entities:
            raise CaptionValidationError(f"{field} contains duplicate entry for entity_id: {entity_id}")
        seen_entities.add(entity_id)
        if entity_id not in entity_ids:
            raise CaptionValidationError(f"{field}[{index}].entity_id must match a global_scene entity_id")
            
        refs_by_side: dict[str, list[str]] = {}
        for v_field, prefix in [("video1_evidence_refs", "v1_atom_"), ("video2_evidence_refs", "v2_atom_")]:
            ref_field = f"{field}[{index}].{v_field}"
            refs = _require_list(item.get(v_field), ref_field)
            refs_by_side[v_field] = _validate_source_local_atom_refs(
                refs,
                ref_field,
                prefix,
                evidence_namespace,
                atom_entity_refs,
                entity_id,
            )

        for key in ("video1_can_determine", "video1_cannot_determine", "video2_can_determine", "video2_cannot_determine", "fusion_additionally_reveals"):
            _validate_string_list(item.get(key), f"{field}[{index}].{key}", allow_empty=True)
            
        rating = _require_string(item.get("gain_rating"), f"{field}[{index}].gain_rating")
        gain_type = _require_string(item.get("gain_type"), f"{field}[{index}].gain_type")
        if gain_type not in ALLOWED_GAIN_TYPES:
            raise CaptionValidationError(f"{field}[{index}].gain_type must be one of {', '.join(ALLOWED_GAIN_TYPES)}")
        if rating not in ALLOWED_GAIN_RATINGS:
            raise CaptionValidationError(f"{field}[{index}].gain_rating must be high, medium, or low")
        has_v1_evidence = bool(refs_by_side["video1_evidence_refs"])
        has_v2_evidence = bool(refs_by_side["video2_evidence_refs"])
        if not (has_v1_evidence or has_v2_evidence):
            raise CaptionValidationError(
                f"{field}[{index}] must cite at least one source-local evidence atom"
            )
        if gain_type in ("confirmation", "complementarity") and not (has_v1_evidence and has_v2_evidence):
            raise CaptionValidationError(
                f"{field}[{index}] gain_type='{gain_type}' requires evidence from both videos"
            )

def _derive_reasoning_focus_entities(parsed: dict[str, Any], entity_ids: set[str]) -> list[dict[str, Any]]:
    entity_reasons: dict[str, set[str]] = {eid: set() for eid in entity_ids}
    
    for link in parsed.get("cross_modal_evidence_links", []):
        if isinstance(link, dict) and link.get("entity_id") in entity_reasons:
            eid = link["entity_id"]
            for dc in link.get("directional_contributions", []):
                direction = dc.get("direction")
                if direction == "confirmation_only":
                    entity_reasons[eid].add("confirmation")
                elif direction == "mutual_complementarity":
                    entity_reasons[eid].add("cross_modal_complementarity")
                elif direction in ("video1_improves_video2", "video2_improves_video1"):
                    entity_reasons[eid].add("directional_gain")
            
    for gain in parsed.get("information_gain", []):
        if isinstance(gain, dict) and gain.get("entity_id") in entity_reasons:
            eid = gain["entity_id"]
            gain_type = gain.get("gain_type")
            if gain_type in ALLOWED_GAIN_TYPES:
                entity_reasons[eid].add(f"gain_{gain_type}")
            
    for event in parsed.get("reasoning_events", []):
        if not isinstance(event, dict): continue
        evt_type = event.get("event_type")
        if evt_type in ("temporal_change", "interaction", "occlusion_change", "spatial_transition", "joint_fusion"):
            for ent in event.get("participating_entities", []):
                if ent in entity_reasons:
                    entity_reasons[ent].add(evt_type)
                    
    for amb in parsed.get("ambiguity_events", []):
        if not isinstance(amb, dict): continue
        ent = amb.get("target_entity")
        if ent in entity_reasons:
            entity_reasons[ent].add("ambiguity_resolution")
            
    derived = []
    for eid, reasons in entity_reasons.items():
        if reasons:
            derived.append({"entity_id": eid, "focus_reasons": sorted(list(reasons))})
            
    derived.sort(key=lambda x: x["entity_id"])
    return derived

def _infer_required_capability(attribute_type: str, missing_attribute: str) -> str | None:
    text = missing_attribute.casefold()
    if _contains_any_term(text, ["color", "paint"]):
        return "color"
    if _contains_any_term(text, ["depth", "distance", "range"]):
        return "depth"
    if _contains_any_term(text, ["thermal", "temperature", "heat"]):
        return "thermal"
    if _contains_any_term(text, ["vehicle type", "vehicle category", "object category"]) or attribute_type in {"semantic_identity", "fine_grained_category"}:
        return "visual_category"
    return None

def _contains_term(text: str, term: str) -> bool:
    # Use \b for word boundaries. We want to match exact phrases
    # optionally containing hyphens.
    return re.search(
        rf"\b{re.escape(term)}\b",
        text,
        flags=re.I,
    ) is not None

def _contains_any_term(text: str, terms: list[str] | tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)

def _visual_category_support_status(missing_attribute: str, recovering_facts: str) -> str:
    target = missing_attribute.casefold()
    facts = recovering_facts.casefold()
    
    # Vehicle-oriented target
    if _contains_any_term(target, ["vehicle", "car", "truck", "van", "bus", "sedan", "suv", "automobile", "motor vehicle"]):
        vehicle_support_terms = ["vehicle", "car", "truck", "van", "bus", "sedan", "suv", "automobile", "hatchback", "pickup", "minivan", "box-shaped vehicle", "tall rear body", "four-wheeled vehicle"]
        if _contains_any_term(facts, vehicle_support_terms):
            return "accept"
        person_support_terms = ["person", "human", "pedestrian", "worker", "walking figure", "cyclist", "rider"]
        if _contains_any_term(facts, person_support_terms) and not _contains_any_term(facts, vehicle_support_terms + ["box-shaped", "road-going", "proportions", "wheel", "window", "door"]):
            return "reject" # purely person evidence for vehicle target
        return "warn"
        
    # Person/pedestrian-oriented target
    if _contains_any_term(target, ["person", "pedestrian", "human", "worker", "cyclist"]):
        person_support_terms = ["person", "pedestrian", "human", "worker", "walking figure", "cyclist", "rider"]
        if _contains_any_term(facts, person_support_terms):
            return "accept"
        vehicle_support_terms = ["vehicle", "car", "truck", "van", "bus", "sedan", "suv", "automobile"]
        if _contains_any_term(facts, vehicle_support_terms) and not _contains_any_term(facts, person_support_terms + ["arm", "leg", "head", "body", "walking"]):
            return "reject" # purely vehicle evidence for person target
        return "warn"
        
    # Bicycle/motorcycle-oriented target
    if _contains_any_term(target, ["bicycle", "bike", "motorcycle", "motorbike", "two-wheeler"]):
        bike_support_terms = ["bicycle", "bike", "motorcycle", "motorbike", "two-wheeler", "rider"]
        if _contains_any_term(facts, bike_support_terms):
            return "accept"
        return "warn"
        
    # Generic object-category target
    return "warn"

def _conditional_recovery_support_status(capability_name: str, missing_attribute: str, recovering_facts: str) -> str:
    facts = recovering_facts.casefold()
    if capability_name == "visual_category":
        return _visual_category_support_status(missing_attribute, recovering_facts)
    elif capability_name == "color":
        if _contains_any_term(facts, ["red", "blue", "green", "yellow", "black", "white", "silver", "gray", "grey", "color", "paint"]):
            return "accept"
        if _contains_any_term(facts, ["depth", "distance"]):
            return "reject"
    elif capability_name == "depth":
        if _contains_any_term(facts, ["depth", "distance", "range", "meters", "close", "far", "closer", "further", "metric"]):
            return "accept"
        if _contains_any_term(facts, ["color", "red"]):
            return "reject"
    elif capability_name == "thermal":
        if _contains_any_term(facts, ["heat", "thermal", "temperature", "hot", "cold", "warm", "cool", "signature"]):
            return "accept"
        if _contains_any_term(facts, ["color", "red"]):
            return "reject"
    elif capability_name == "structure_edge":
        if _contains_any_term(facts, ["edge", "boundary", "structure", "shape", "contour"]):
            return "accept"
            
    return "warn"


def _normalize_referential_scope(value: str) -> str:
    return " ".join(value.casefold().split())


def _validate_caption_schema(parsed: dict[str, Any], valid_frame_keys: set[str], expected_modality1: str, expected_modality2: str) -> tuple[dict[str, Any], list[str]]:
    if not valid_frame_keys:
        raise CaptionValidationError("valid_frame_keys must not be empty for validation")

    atom_frame_keys: dict[str, set[str]] = {}
    atom_facts: dict[str, str] = {}
    atom_entity_refs: dict[str, set[str]] = {}
    local_warnings: list[str] = []

    missing = [field for field in CAPTION_REQUIRED_TOP_LEVEL_FIELDS if field not in parsed]
    if missing:
        raise CaptionValidationError(f"Gemini response missing required caption field(s): {', '.join(missing)}")
        
    unexpected_fields = set(parsed.keys()) - CAPTION_REQUIRED_TOP_LEVEL_FIELDS
    if unexpected_fields:
        raise CaptionValidationError(f"Gemini response contains unknown top-level fields: {', '.join(sorted(unexpected_fields))}. Allowed fields are only: {', '.join(CAPTION_REQUIRED_TOP_LEVEL_FIELDS)}")

    evidence_namespace: set[str] = set()

    def _register_evidence_id(eid: str) -> None:
        if eid in evidence_namespace:
            raise CaptionValidationError(f"Duplicate evidence ID found: {eid}. Evidence IDs must be globally unique.")
        evidence_namespace.add(eid)

    global_scene = _require_object(parsed["global_scene"], "global_scene")
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

    modality1 = parsed.get("video1_analysis", {}).get("modality", "")
    if modality1 != expected_modality1:
        raise CaptionValidationError(f"video1_analysis.modality {modality1!r} does not match expected {expected_modality1!r}")
    modality2 = parsed.get("video2_analysis", {}).get("modality", "")
    if modality2 != expected_modality2:
        raise CaptionValidationError(f"video2_analysis.modality {modality2!r} does not match expected {expected_modality2!r}")

    def _validate_video_analysis(parsed: dict[str, Any], field: str, atom_prefix: str) -> None:
        analysis = _require_object(parsed.get(field), field)
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
        _validate_uncertain_observations(analysis.get("uncertain_observations"), f"{field}.uncertain_observations")
        
        missing_attrs = _require_list(analysis.get("missing_key_attributes"), f"{field}.missing_key_attributes")
        for i, attr in enumerate(missing_attrs, start=1):
            if not isinstance(attr, dict):
                raise CaptionValidationError(f"{field}.missing_key_attributes[{i}] must be an object")
            attr_type = attr.get("attribute_type")
            if attr_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
                raise CaptionValidationError(f"{field}.missing_key_attributes[{i}].attribute_type invalid: {attr_type}")
            _require_string(attr.get("missing_attribute"), f"{field}.missing_key_attributes[{i}].missing_attribute")
            _require_string(attr.get("why_missing"), f"{field}.missing_key_attributes[{i}].why_missing")
            _require_list(attr.get("recoverable_evidence_refs"), f"{field}.missing_key_attributes[{i}].recoverable_evidence_refs")
            # Defer cross-modal rule check until all atoms are registered

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
    
    _validate_cross_modal_evidence_links(parsed.get("cross_modal_evidence_links"), entity_ids, evidence_namespace, atom_entity_refs, "cross_modal_evidence_links")
    _validate_information_gain(parsed.get("information_gain"), entity_ids, evidence_namespace, atom_entity_refs, "information_gain")

    events = _require_list(parsed["reasoning_events"], "reasoning_events")
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise CaptionValidationError(f"reasoning_events[{index}] must be an object")
        evt_id = _require_string(event.get("event_id"), f"reasoning_events[{index}].event_id")
        if not evt_id.startswith("evt_"):
            raise CaptionValidationError(f"reasoning_events[{index}].event_id must start with evt_")
        _register_evidence_id(evt_id)
        evt_type = _require_string(event.get("event_type"), f"reasoning_events[{index}].event_type")
        if evt_type not in ALLOWED_REASONING_EVENT_TYPES:
            raise CaptionValidationError(f"reasoning_events[{index}].event_type {evt_type} is not a valid reasoning event type")
        part_ents = _require_list(event.get("participating_entities"), f"reasoning_events[{index}].participating_entities")
        if not part_ents:
            raise CaptionValidationError(f"reasoning_events[{index}].participating_entities must not be empty")
        participating_entity_set: set[str] = set()
        for pe in part_ents:
            pe = _require_string(pe, f"reasoning_events[{index}].participating_entities[]")
            if pe in participating_entity_set:
                raise CaptionValidationError(f"reasoning_events[{index}].participating_entities contains duplicate entity: {pe}")
            if pe not in entity_ids:
                raise CaptionValidationError(f"reasoning_events[{index}] entity {pe} not found in physical_entities")
            participating_entity_set.add(pe)
        
        atom_refs = _require_list(event.get("supporting_atom_refs"), f"reasoning_events[{index}].supporting_atom_refs")
        if not atom_refs:
            raise CaptionValidationError(f"reasoning_events[{index}] must have supporting_atom_refs")
        covered_participating_entities: set[str] = set()
        for ref in atom_refs:
            if not (ref.startswith("v1_atom_") or ref.startswith("v2_atom_")):
                raise CaptionValidationError(f"reasoning_events[{index}].supporting_atom_refs must only point to atoms. Invalid: {ref}")
            if ref not in evidence_namespace:
                raise CaptionValidationError(f"reasoning_events[{index}] references unknown atom: {ref}")
            connected_entities = atom_entity_refs.get(ref, set()) & participating_entity_set
            if not connected_entities:
                raise CaptionValidationError(
                    f"reasoning_events[{index}] supporting atom {ref} is not connected to any participating entity"
                )
            covered_participating_entities.update(connected_entities)
        missing_entity_coverage = participating_entity_set - covered_participating_entities
        if missing_entity_coverage:
            raise CaptionValidationError(
                f"reasoning_events[{index}] participating_entities not covered by supporting atoms: "
                f"{', '.join(sorted(missing_entity_coverage))}"
            )
                
        dynamic_event_types = {"temporal_change", "occlusion_change", "spatial_transition"}
        if evt_type in dynamic_event_types:
            supporting_frames = set()
            for ref in atom_refs:
                supporting_frames.update(atom_frame_keys.get(ref, set()))
            if len(supporting_frames) < 2:
                raise CaptionValidationError(f"reasoning_events[{index}] {evt_type} requires evidence spanning at least 2 distinct frames")
                
        if evt_type == "joint_fusion":
            has_v1 = any(r.startswith("v1_atom_") for r in atom_refs)
            has_v2 = any(r.startswith("v2_atom_") for r in atom_refs)
            if not (has_v1 and has_v2):
                raise CaptionValidationError(f"reasoning_events[{index}] joint_fusion requires at least one V1 atom and one V2 atom")
            
            # Validator B: Reasoning consistency
            res_dir = event.get("resolution_direction")
            if res_dir in ("video1_resolves_video2", "video2_resolves_video1"):
                raise CaptionValidationError(
                    f"reasoning_events[{index}] has event_type='joint_fusion' but resolution_direction='{res_dir}'. "
                    "Use unidirectional_disambiguation instead, or remove the unidirectional direction if it is genuinely joint fusion."
                )

        _require_string(event.get("description"), f"reasoning_events[{index}].description")

    ambiguities = _require_list(parsed["ambiguity_events"], "ambiguity_events")
    for index, event in enumerate(ambiguities, start=1):
        if not isinstance(event, dict):
            raise CaptionValidationError(f"ambiguity_events[{index}] must be an object")
        amb_id = _require_string(event.get("ambiguity_id"), f"ambiguity_events[{index}].ambiguity_id")
        if not amb_id.startswith("amb_"):
            raise CaptionValidationError(f"ambiguity_events[{index}].ambiguity_id must start with amb_")
        _register_evidence_id(amb_id)
        target = _require_string(event.get("target_entity"), f"ambiguity_events[{index}].target_entity")
        if target not in entity_ids:
            raise CaptionValidationError(f"ambiguity_events[{index}] target_entity {target} not found in physical_entities")
        
        direction = event.get("direction")
        if direction not in ALLOWED_AMBIGUITY_DIRECTIONS:
            raise CaptionValidationError(f"ambiguity_events[{index}].direction must be video1_resolves_video2 or video2_resolves_video1")
        
        amb_video = _require_string(event.get("ambiguous_video"), f"ambiguity_events[{index}].ambiguous_video")
        res_video = _require_string(event.get("resolving_video"), f"ambiguity_events[{index}].resolving_video")
        if direction == "video1_resolves_video2":
            if amb_video != "video2" or res_video != "video1":
                raise CaptionValidationError(f"ambiguity_events[{index}] direction {direction} contradicts video fields")
        elif direction == "video2_resolves_video1":
            if amb_video != "video1" or res_video != "video2":
                raise CaptionValidationError(f"ambiguity_events[{index}] direction {direction} contradicts video fields")

        for key in (
            "low_confidence_observation", "why_ambiguous_video_cannot_resolve", 
            "resolving_discriminative_evidence", "fusion_conclusion",
        ):
            _require_string(event.get(key), f"ambiguity_events[{index}].{key}")
            
        missing_type = event.get("missing_attribute_type")
        if missing_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
            raise CaptionValidationError(f"ambiguity_events[{index}].missing_attribute_type invalid: {missing_type}")

        hypotheses = _require_list(event.get("candidate_hypotheses"), f"ambiguity_events[{index}].candidate_hypotheses")
        if len(hypotheses) < MIN_AMBIGUITY_EVENT_HYPOTHESES:
            raise CaptionValidationError(f"ambiguity_events[{index}].candidate_hypotheses must include at least two hypotheses")
            
        normalized_hyps: set[str] = set()
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            hyp_text = hypothesis.get("hypothesis")
            if isinstance(hyp_text, str) and hyp_text.strip():
                normalized_hyps.add(hyp_text.strip().casefold())
        if len(normalized_hyps) < MIN_AMBIGUITY_EVENT_HYPOTHESES:
            raise CaptionValidationError(f"ambiguity_events[{index}].candidate_hypotheses must contain at least two distinct hypotheses")
            
        for hyp_index, hypothesis in enumerate(hypotheses, start=1):
            if not isinstance(hypothesis, dict):
                raise CaptionValidationError(f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}] must be an object")
            _require_string(hypothesis.get("hypothesis"), f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}].hypothesis")
            _require_string(hypothesis.get("why_compatible_with_ambiguous"), f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}].why_compatible_with_ambiguous")
            _require_string(hypothesis.get("support_from_resolving"), f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}].support_from_resolving")
            
        eliminated = _require_list(event.get("eliminated_hypotheses"), f"ambiguity_events[{index}].eliminated_hypotheses")
        if not eliminated:
            raise CaptionValidationError(f"ambiguity_events[{index}].eliminated_hypotheses must not be empty")
            
        candidate_names = {h["hypothesis"].strip().casefold() for h in hypotheses if isinstance(h, dict) and "hypothesis" in h}
            
        for elim_index, hypothesis in enumerate(eliminated, start=1):
            if not isinstance(hypothesis, dict):
                raise CaptionValidationError(f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}] must be an object")
            elim_name = _require_string(hypothesis.get("hypothesis"), f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}].hypothesis")
            if elim_name.strip().casefold() not in candidate_names:
                raise CaptionValidationError(f"ambiguity_events[{index}] eliminated hypothesis must appear in candidate_hypotheses")
            _require_string(hypothesis.get("why_eliminated"), f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}].why_eliminated")
        
        amb_refs = _require_list(event.get("ambiguous_evidence_refs"), f"ambiguity_events[{index}].ambiguous_evidence_refs")
        if not amb_refs:
            raise CaptionValidationError(f"ambiguity_events[{index}].ambiguous_evidence_refs must not be empty")
        res_refs = _require_list(event.get("resolving_evidence_refs"), f"ambiguity_events[{index}].resolving_evidence_refs")
        if not res_refs:
            raise CaptionValidationError(f"ambiguity_events[{index}].resolving_evidence_refs must not be empty")
            
        if direction == "video1_resolves_video2":
            if not all(r.startswith("v2_atom_") for r in amb_refs):
                raise CaptionValidationError(f"ambiguity_events[{index}] direction {direction} requires ambiguous_evidence_refs to be v2_atom_")
            if not all(r.startswith("v1_atom_") for r in res_refs):
                raise CaptionValidationError(f"ambiguity_events[{index}] direction {direction} requires resolving_evidence_refs to be v1_atom_")
        elif direction == "video2_resolves_video1":
            if not all(r.startswith("v1_atom_") for r in amb_refs):
                raise CaptionValidationError(f"ambiguity_events[{index}] direction {direction} requires ambiguous_evidence_refs to be v1_atom_")
            if not all(r.startswith("v2_atom_") for r in res_refs):
                raise CaptionValidationError(f"ambiguity_events[{index}] direction {direction} requires resolving_evidence_refs to be v2_atom_")

        for ref in amb_refs + res_refs:
            if not (ref.startswith("v1_atom_") or ref.startswith("v2_atom_")):
                raise CaptionValidationError(f"ambiguity_events[{index}] evidence_refs must only point to atoms. Invalid: {ref}")
            if ref not in evidence_namespace:
                raise CaptionValidationError(f"ambiguity_events[{index}] references unknown atom: {ref}")
            _require_atom_entity_connection(ref, target, atom_entity_refs, f"ambiguity_events[{index}]")

    QA_PATTERN_EVENT_TYPE_MAP = {
        "temporal_integration": {"temporal_change"},
        "occlusion_reasoning": {"occlusion_change"},
        "interaction_reasoning": {"interaction"},
        "spatial_transition": {"spatial_transition"},
        "joint_fusion": {"joint_fusion"},
    }

    qa_details = _require_list(parsed["qa_relevant_details"], "qa_relevant_details")
    for index, qa in enumerate(qa_details, start=1):
        if not isinstance(qa, dict):
            raise CaptionValidationError(f"qa_relevant_details[{index}] must be an object")
        qa_id = _require_string(qa.get("detail_id"), f"qa_relevant_details[{index}].detail_id")
        if not qa_id.startswith("qa_detail_"):
            raise CaptionValidationError(f"qa_relevant_details[{index}].detail_id must start with qa_detail_")
        _register_evidence_id(qa_id)
        pat = _require_string(qa.get("reasoning_pattern"), f"qa_relevant_details[{index}].reasoning_pattern")
        if pat not in ALLOWED_QA_REASONING_PATTERNS:
            raise CaptionValidationError(f"qa_relevant_details[{index}].reasoning_pattern invalid: {pat}")
        refs = _require_list(qa.get("supporting_refs"), f"qa_relevant_details[{index}].supporting_refs")
        if not refs:
            raise CaptionValidationError(f"qa_relevant_details[{index}].supporting_refs must not be empty")
        for ref in refs:
            if ref.startswith("qa_detail_"):
                raise CaptionValidationError(f"qa_relevant_details[{index}] illegally references another qa_detail: {ref}")
            if ref not in evidence_namespace:
                raise CaptionValidationError(f"qa_relevant_details[{index}] references unknown ID: {ref}")
                
        expected_types = QA_PATTERN_EVENT_TYPE_MAP.get(pat)
        if expected_types:
            referenced_event_types = {
                evt["event_type"] for evt in events if evt["event_id"] in refs
            }
            if not (referenced_event_types & expected_types):
                raise CaptionValidationError(f"qa_relevant_details[{index}] pattern '{pat}' requires at least one supporting event of type: {', '.join(expected_types)}")
            
        def _resolve_atoms(ref_id: str) -> set[str]:
            if ref_id.startswith("v1_atom_") or ref_id.startswith("v2_atom_"):
                return {ref_id}
            atoms = set()
            if ref_id.startswith("evt_"):
                evt = next(e for e in events if e.get("event_id") == ref_id)
                atoms.update(evt.get("supporting_atom_refs", []))
            elif ref_id.startswith("amb_"):
                amb = next(a for a in ambiguities if a.get("ambiguity_id") == ref_id)
                atoms.update(amb.get("ambiguous_evidence_refs", []))
                atoms.update(amb.get("resolving_evidence_refs", []))
            return atoms

        resolved_atoms = set()
        for ref in refs:
            resolved_atoms.update(_resolve_atoms(ref))
            
        if pat == "multi_hop_composition" and len(resolved_atoms) < 2:
            raise CaptionValidationError(f"qa_relevant_details[{index}] multi_hop_composition requires at least 2 underlying atoms")
        if pat in ("cross_modal_disambiguation", "hypothesis_elimination"):
            if not any(r.startswith("amb_") for r in refs):
                raise CaptionValidationError(f"qa_relevant_details[{index}] {pat} MUST reference at least one amb_ event directly")
        if pat == "joint_fusion":
            has_v1 = any(a.startswith("v1_atom_") for a in resolved_atoms)
            has_v2 = any(a.startswith("v2_atom_") for a in resolved_atoms)
            if not (has_v1 and has_v2):
                raise CaptionValidationError(f"qa_relevant_details[{index}] joint_fusion requires at least one V1 atom and one V2 atom in its resolved tree")
        
        _require_string(qa.get("why_question_worthy"), f"qa_relevant_details[{index}].why_question_worthy")

    rejected = _require_list(parsed["rejected_observations"], "rejected_observations")
    for index, item in enumerate(rejected, start=1):
        if not isinstance(item, dict):
            raise CaptionValidationError(f"rejected_observations[{index}] must be an object")
        _require_string(item.get("observation"), f"rejected_observations[{index}].observation")
        _require_string(item.get("reason"), f"rejected_observations[{index}].reason")

    def _check_missing_attrs(analysis_key: str, required_prefix: str, ref_modality: str):
        analysis = parsed.get(analysis_key, {})
        for i, attr in enumerate(analysis.get("missing_key_attributes", []), start=1):
            refs = attr.get("recoverable_evidence_refs", [])
            if refs:
                for ref in refs:
                    if not (ref.startswith("v1_atom_") or ref.startswith("v2_atom_")):
                        raise CaptionValidationError(
                            f"{analysis_key}.missing_key_attributes[{i}] "
                            f"recoverable_evidence_refs MUST only reference atoms. "
                            f"Invalid: {ref}"
                        )

                    if ref not in evidence_namespace:
                        raise CaptionValidationError(
                            f"{analysis_key}.missing_key_attributes[{i}] "
                            f"references unknown atom: {ref}"
                        )

                    if not ref.startswith(required_prefix):
                        raise CaptionValidationError(
                            f"{analysis_key}.missing_key_attributes[{i}] "
                            f"recoverable_evidence_refs must only contain "
                            f"{required_prefix} atom IDs. Invalid cross-side ref: {ref}"
                        )
                
                attr_type = attr.get("attribute_type", "")
                missing_attr = attr.get("missing_attribute", "")
                required_cap = _infer_required_capability(attr_type, missing_attr)
                
                if required_cap:
                    cap_state = MODALITY_CAPABILITIES.get(ref_modality, {}).get(required_cap, "conditional")
                    if cap_state == "not_direct":
                        raise CaptionValidationError(
                            f"{analysis_key}.missing_key_attributes[{i}]: "
                            f"'{missing_attr}' marked recoverable from {ref_modality}, "
                            f"but {ref_modality} has 'not_direct' capability for {required_cap}."
                        )
                    elif cap_state == "conditional":
                        recovering_facts = " ".join(atom_facts.get(r, "") for r in refs).casefold()
                        support_status = _conditional_recovery_support_status(required_cap, missing_attr, recovering_facts)
                                
                        if support_status == "reject":
                            raise CaptionValidationError(
                                f"{analysis_key}.missing_key_attributes[{i}]: "
                                f"'{missing_attr}' marked recoverable via {ref_modality} ({required_cap}), "
                                f"but supporting atoms clearly do not contain relevant information."
                            )
                        elif support_status == "warn":
                            local_warnings.append(
                                f"{analysis_key}.missing_key_attributes[{i}]: "
                                f"'{missing_attr}' recovered via conditional capability {required_cap}. "
                                f"Check if referenced atoms actually support this."
                            )

    _check_missing_attrs("video1_analysis", "v2_atom_", modality2)
    _check_missing_attrs("video2_analysis", "v1_atom_", modality1)
    
    # Honest empty reasoning graphs are allowed; no 'has_reasoning_content' check is enforced.
        
    # Validator E: Precision language
    precision_warnings = [
        "razor-sharp", "exact coordinates", "perfect boundaries", 
        "extreme edge definition", "impossible to distinguish", "exact model"
    ]

    def _check_precision(obj: Any, path: str = "") -> None:
        if isinstance(obj, str):
            for w in precision_warnings:
                if w in obj.lower():
                    local_warnings.append(f"{path} contains unsupported precision language: '{w}'")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _check_precision(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check_precision(v, f"{path}[{i}]")
    _check_precision(parsed)
    parsed["global_scene"]["reasoning_focus_entities"] = _derive_reasoning_focus_entities(parsed, entity_ids)
    parsed = _normalize_license_plates(parsed)
    return parsed, local_warnings


