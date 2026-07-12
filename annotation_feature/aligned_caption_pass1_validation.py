"""Semantic validation for Pass 1 (evidence construction) of the aligned multimodal caption pipeline."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Mapping

from annotation_feature.aligned_caption_schema import (
    ALLOWED_MISSING_ATTRIBUTE_TYPES,
    CaptionValidationError,
    FORBIDDEN_GLOBAL_SCENE_MESSAGE,
    FORBIDDEN_GLOBAL_SCENE_PATTERN,
    FORBIDDEN_SENSOR_QUALITY_MESSAGE,
    FORBIDDEN_SENSOR_QUALITY_PATTERN,
    MIN_DETAILED_CAPTION_WORDS,
    MIN_FRAME_DETAIL_WORDS,
    MIN_SCENE_SUMMARY_WORDS,
    _normalize_license_plates,
)

from annotation_feature.aligned_caption_schema import (
    MODALITY_CAPABILITIES,
    FORBIDDEN_MECHANISM_PATTERNS,
    FORBIDDEN_MECHANISM_MESSAGE,
    FORBIDDEN_INFERENTIAL_MESSAGE,
    FORBIDDEN_INFERENTIAL_PATTERN,
    GENERIC_SENSOR_EXPLANATION_PATTERNS,
    normalize_modality_name,
)
from annotation_feature.aligned_caption_validation import (
    _require_object,
    _require_list,
    _require_string,
    _validate_min_words,
    _validate_physical_world_wording,
    _validate_uncertain_observations,
    _normalize_referential_scope,
    _infer_required_capability,
    _conditional_recovery_support_status,
    GENERIC_SENSOR_EXPLANATION_PATTERNS,
    FORBIDDEN_INFERENTIAL_PATTERN,
)

REPRESENTATION_SUBJECT_PATTERN = re.compile(
    r"\b(pattern|map|signal|response|activation|data|representation|event boundary|motion boundary|contour pattern)s?\b",
    re.I
)
REPRESENTATION_PREDICATE_PATTERN = re.compile(
    r"\b(capture|resolve|define|encode|represent|delineate|mark|trace|map|show)(s|d|ed|ing|n)?\b",
    re.I
)

EVENT_REPRESENTATION_CLAIM_PATTERN = re.compile(
    r"\b(?:motion\s+boundar(?:y|ies)|architectural\s+edges?|(?:intricate\s+)?silhouettes?|"
    r"(?:blocky|rectangular)\s+(?:boundary|profile))\b.{0,45}\b"
    r"(?:captur(?:e|ed)|defin(?:e|ed)|resolv(?:e|ed)|delineat(?:e|ed)|mark(?:ed)?)\b",
    re.I,
)
DEPTH_EXACT_DISTANCE_PATTERN = re.compile(r"\b(?:exactly\s+)?\d+(?:\.\d+)?\s*(?:m|meter|meters|metre|metres|cm|centimeters?|feet|ft)\b", re.I)
IR_EXACT_TEMPERATURE_PATTERN = re.compile(r"\b-?\d+(?:\.\d+)?\s*(?:°\s*)?(?:c|f|celsius|fahrenheit|kelvin)\b", re.I)


def _validate_modality_specific_physical_claims(
    modality: str,
    text: str,
    path: str,
    issues: list[Pass1ValidationIssue],
    warnings: list[str],
) -> None:
    """Validate source-local physical claims using the trusted expected modality."""
    canonical = normalize_modality_name(modality)
    if canonical == "event" and EVENT_REPRESENTATION_CLAIM_PATTERN.search(text):
        issues.append(Pass1ValidationIssue(
            path,
            "physical_world_wording",
            "Event-local physical text describes a representation as resolving, defining, or capturing the scene; rewrite it as a direct physical observation.",
            scope="video_analysis",
        ))
    elif canonical == "depth":
        if DEPTH_EXACT_DISTANCE_PATTERN.search(text):
            warnings.append(f"{path} unsupported_modality_claim: exact metric distance requires explicit calibration support.")
        if re.search(r"\b(?:red|blue|green|yellow|white|black|silver|paint(?:ed)?|wooden|metallic)\b", text, re.I):
            warnings.append(f"{path} unsupported_modality_claim: depth-local color or material certainty requires exceptional direct support.")
    elif canonical == "ir":
        if IR_EXACT_TEMPERATURE_PATTERN.search(text) or re.search(r"\b(?:because|caused by)\b.{0,30}\b(?:heat|hot|warm|cold|thermal)\b", text, re.I):
            warnings.append(f"{path} unsupported_modality_claim: exact temperature or thermal cause is not directly established by generic IR metadata.")

from dataclasses import dataclass
@dataclass
class Pass1ValidationIssue:
    path: str
    category: str
    message: str
    scope: str = ""

class Pass1StructuralValidationError(CaptionValidationError):
    def __init__(self, message: str, errors: list[Pass1ValidationIssue]):
        details = "; ".join(error.message for error in errors)
        super().__init__(f"{message}: {details}" if details else message)
        self.errors = errors

class Pass1SemanticValidationError(CaptionValidationError):
    def __init__(self, message: str, errors: list[Pass1ValidationIssue]):
        details = "; ".join(error.message for error in errors)
        super().__init__(f"{message}: {details}" if details else message)
        self.errors = errors

class Pass1ValidationContext:
    def __init__(self):
        self.entity_ids: set[str] = set()
        self.evidence_namespace: set[str] = set()
        self.atom_entity_refs: dict[str, set[str]] = {}
        self.atom_sources: dict[str, str] = {}
        self.expected_source_modalities: Mapping[str, str] = MappingProxyType({})
        self.atom_facts: dict[str, Any] = {}
        self.atom_frame_keys: dict[str, set[str]] = {}
        self.atom_paths: dict[str, str] = {}
        self.entity_scopes: dict[str, str] = {}
        self.entity_categories: dict[str, str] = {}


def _validate_pass1_structure(
    parsed: Any,
    valid_frame_keys: set[str],
    expected_source_modalities: Mapping[str, str],
) -> Pass1ValidationContext:
    """Aggregate Pass 1 shape/reference errors before semantic validation."""
    issues: list[Pass1ValidationIssue] = []
    context = Pass1ValidationContext()

    if not isinstance(parsed, dict):
        raise Pass1StructuralValidationError("Structural Validation Errors", [Pass1ValidationIssue("root", "invalid_type", "Pass 1 response must be an object", "root")])
    if not valid_frame_keys:
        issues.append(Pass1ValidationIssue("valid_frame_keys", "invalid_type", "valid_frame_keys must be non-empty", "root"))

    normalized_expected: dict[str, str] = {}
    for key, value in expected_source_modalities.items():
        normalized = normalize_modality_name(value) if isinstance(value, str) else ""
        if normalized not in MODALITY_CAPABILITIES:
            issues.append(Pass1ValidationIssue(key, "unsupported_modality", f"Unsupported expected modality: {value!r}", "root"))
        normalized_expected[key] = normalized
    context.expected_source_modalities = MappingProxyType(normalized_expected)

    for key in PASS1_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in parsed:
            issues.append(Pass1ValidationIssue(key, "missing_field", f"Missing required top-level field: {key}", "root"))
        elif not isinstance(parsed[key], dict):
            issues.append(Pass1ValidationIssue(key, "invalid_type", f"{key} must be an object", "root"))
    for key in set(parsed) - PASS1_REQUIRED_TOP_LEVEL_FIELDS:
        issues.append(Pass1ValidationIssue(key, "unexpected_field", f"Unexpected top-level field: {key}", "root"))

    # Stop here so absent containers do not generate dozens of child errors.
    if any(i.category in {"missing_field", "invalid_type"} and i.path in PASS1_REQUIRED_TOP_LEVEL_FIELDS for i in issues):
        raise Pass1StructuralValidationError("Structural Validation Errors", issues)

    global_scene = parsed["global_scene"]
    for key in set(global_scene) - PASS1_GLOBAL_SCENE_ALLOWED_FIELDS:
        issues.append(Pass1ValidationIssue(f"global_scene.{key}", "unexpected_field", f"Unexpected global_scene field: {key}", "global_scene"))
    for key in ("scene_summary", "environment", "temporal_progression"):
        if not isinstance(global_scene.get(key), str) or not global_scene.get(key, "").strip():
            issues.append(Pass1ValidationIssue(f"global_scene.{key}", "missing_field", f"{key} must be a non-empty string", "global_scene"))

    entities = global_scene.get("physical_entities")
    if not isinstance(entities, list) or not entities:
        issues.append(Pass1ValidationIssue("global_scene.physical_entities", "invalid_type", "physical_entities must be a non-empty list", "global_scene"))
        entities = []
    normalized_scopes: dict[str, str] = {}
    for index, entity in enumerate(entities):
        path = f"global_scene.physical_entities[{index}]"
        if not isinstance(entity, dict):
            issues.append(Pass1ValidationIssue(path, "invalid_type", "entity must be an object", "entity"))
            continue
        for key in set(entity) - {"entity_id", "category", "referential_scope", "evidence_profile"}:
            issues.append(Pass1ValidationIssue(f"{path}.{key}", "unexpected_field", f"Unexpected entity field: {key}", "entity"))
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id.strip():
            issues.append(Pass1ValidationIssue(f"{path}.entity_id", "missing_field", "entity_id must be a non-empty string", "entity"))
        else:
            if entity_id in context.entity_ids:
                issues.append(Pass1ValidationIssue(f"{path}.entity_id", "duplicate_id", f"Duplicate entity_id: {entity_id}", "entity"))
            context.entity_ids.add(entity_id)
        for key in ("category", "referential_scope"):
            if not isinstance(entity.get(key), str) or not entity.get(key, "").strip():
                issues.append(Pass1ValidationIssue(f"{path}.{key}", "missing_field", f"{key} must be a non-empty string", "entity"))
        scope = entity.get("referential_scope")
        if isinstance(scope, str) and scope.strip() and isinstance(entity_id, str):
            normalized = _normalize_referential_scope(scope)
            if normalized in normalized_scopes and normalized_scopes[normalized] != entity_id:
                issues.append(Pass1ValidationIssue(f"{path}.referential_scope", "invalid_reference", "Duplicate normalized referential_scope", "entity"))
            normalized_scopes[normalized] = entity_id
            context.entity_scopes[entity_id] = scope
        category = entity.get("category")
        if isinstance(category, str) and category.strip() and isinstance(entity_id, str):
            context.entity_categories[entity_id] = category
        profile = entity.get("evidence_profile")
        if profile is not None:
            if not isinstance(profile, dict) or not profile:
                issues.append(Pass1ValidationIssue(f"{path}.evidence_profile", "invalid_type", "evidence_profile must be a non-empty object when present", "evidence_profile"))
            else:
                for key in set(profile) - {"identity_evidence", "observable_attributes", "spatial_context"}:
                    issues.append(Pass1ValidationIssue(f"{path}.evidence_profile.{key}", "unexpected_field", f"Unexpected evidence_profile field: {key}", "evidence_profile"))
                for key, value in profile.items():
                    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value):
                        issues.append(Pass1ValidationIssue(f"{path}.evidence_profile.{key}", "invalid_type", f"{key} must be a non-empty string list", "evidence_profile"))

    for source_key, prefix in (("video1_analysis", "v1_atom_"), ("video2_analysis", "v2_atom_")):
        analysis = parsed[source_key]
        for key in set(analysis) - PASS1_VIDEO_ANALYSIS_ALLOWED_FIELDS:
            issues.append(Pass1ValidationIssue(f"{source_key}.{key}", "unexpected_field", f"Unexpected analysis field: {key}", "video_analysis"))
        expected = normalized_expected.get(source_key)
        generated = analysis.get("modality")
        if not isinstance(generated, str) or not generated.strip():
            issues.append(Pass1ValidationIssue(f"{source_key}.modality", "missing_field", "modality must be a non-empty string", "video_analysis"))
        elif expected and normalize_modality_name(generated) != expected:
            issues.append(Pass1ValidationIssue(f"{source_key}.modality", "modality_mismatch", f"Generated modality {generated!r} does not match expected {expected!r}", "video_analysis"))
        if not isinstance(analysis.get("detailed_caption"), str) or not analysis.get("detailed_caption", "").strip():
            issues.append(Pass1ValidationIssue(f"{source_key}.detailed_caption", "missing_field", "detailed_caption must be a non-empty string", "video_analysis"))
        for key in ("sensor_specific_cues", "sensor_limitations"):
            value = analysis.get(key)
            if not isinstance(value, list):
                issues.append(Pass1ValidationIssue(f"{source_key}.{key}", "invalid_type", f"{key} must be a list", "video_analysis"))
            elif any(not isinstance(item, str) or not item.strip() for item in value):
                issues.append(Pass1ValidationIssue(f"{source_key}.{key}", "invalid_type", f"{key} items must be non-empty strings", "video_analysis"))
        for key in ("uncertain_observations", "missing_key_attributes"):
            if not isinstance(analysis.get(key), list):
                issues.append(Pass1ValidationIssue(f"{source_key}.{key}", "invalid_type", f"{key} must be a list", "video_analysis"))

        atoms = analysis.get("information_atoms")
        if not isinstance(atoms, list) or not atoms:
            issues.append(Pass1ValidationIssue(f"{source_key}.information_atoms", "invalid_type", "information_atoms must be a non-empty list", "video_analysis"))
            continue
        for index, atom in enumerate(atoms):
            path = f"{source_key}.information_atoms[{index}]"
            if not isinstance(atom, dict):
                issues.append(Pass1ValidationIssue(path, "invalid_type", "atom must be an object", "atom"))
                continue
            atom_id = atom.get("atom_id")
            valid_id = isinstance(atom_id, str) and atom_id.startswith(prefix)
            if not isinstance(atom_id, str) or not atom_id.strip():
                issues.append(Pass1ValidationIssue(f"{path}.atom_id", "missing_field", "atom_id must be non-empty", "atom"))
            elif not valid_id:
                issues.append(Pass1ValidationIssue(f"{path}.atom_id", "invalid_reference", f"atom_id must start with {prefix}", "atom"))
            elif atom_id in context.evidence_namespace:
                issues.append(Pass1ValidationIssue(f"{path}.atom_id", "duplicate_id", f"Duplicate evidence ID: {atom_id}", "atom"))
            else:
                context.evidence_namespace.add(atom_id)
            frames = atom.get("frame_keys")
            valid_frames: set[str] = set()
            if not isinstance(frames, list) or not frames:
                issues.append(Pass1ValidationIssue(f"{path}.frame_keys", "invalid_type", "frame_keys must be a non-empty list", "atom"))
            else:
                for idx, frame in enumerate(frames):
                    if not isinstance(frame, str):
                        issues.append(Pass1ValidationIssue(f"{path}.frame_keys[{idx}]", "invalid_type", "frame key must be a string", "atom"))
                    elif frame not in valid_frame_keys:
                        issues.append(Pass1ValidationIssue(f"{path}.frame_keys[{idx}]", "invalid_frame_reference", f"Unknown frame_key {frame!r}", "atom"))
                    else:
                        valid_frames.add(frame)
            refs = atom.get("entity_refs")
            valid_refs: set[str] = set()
            if not isinstance(refs, list) or not refs:
                issues.append(Pass1ValidationIssue(f"{path}.entity_refs", "invalid_type", "entity_refs must be a non-empty list", "atom"))
            else:
                for idx, ref in enumerate(refs):
                    if not isinstance(ref, str):
                        issues.append(Pass1ValidationIssue(f"{path}.entity_refs[{idx}]", "invalid_type", "entity_ref must be a string", "atom"))
                    elif ref not in context.entity_ids:
                        issues.append(Pass1ValidationIssue(f"{path}.entity_refs[{idx}]", "invalid_entity_reference", f"Unknown entity_ref: {ref}", "atom"))
                    else:
                        valid_refs.add(ref)
            fact = atom.get("fact")
            if not isinstance(fact, str) or not fact.strip():
                issues.append(Pass1ValidationIssue(f"{path}.fact", "missing_field", "fact must be a non-empty string", "atom"))
            if valid_id and valid_frames and valid_refs and isinstance(fact, str) and fact.strip():
                context.atom_frame_keys[atom_id] = valid_frames
                context.atom_entity_refs[atom_id] = valid_refs
                context.atom_facts[atom_id] = fact
                context.atom_sources[atom_id] = source_key
                context.atom_paths[atom_id] = path

    # Cross-references are checked only against fully valid atoms/entities.
    for source_key, prefix in (("video1_analysis", "v1_atom_"), ("video2_analysis", "v2_atom_")):
        analysis = parsed[source_key]
        uncertainties = analysis.get("uncertain_observations") if isinstance(analysis.get("uncertain_observations"), list) else []
        for index, obs in enumerate(uncertainties):
            path = f"{source_key}.uncertain_observations[{index}]"
            if not isinstance(obs, dict):
                issues.append(Pass1ValidationIssue(path, "invalid_type", "uncertain observation must be an object", "uncertainty"))
                continue
            entity_id = obs.get("entity_id")
            valid_entity = isinstance(entity_id, str) and entity_id in context.entity_ids
            if not isinstance(entity_id, str) or not entity_id.strip():
                issues.append(Pass1ValidationIssue(f"{path}.entity_id", "missing_field", "entity_id must be a non-empty string", "uncertainty"))
            elif not valid_entity:
                issues.append(Pass1ValidationIssue(f"{path}.entity_id", "invalid_entity_reference", f"Unknown entity_id: {entity_id}", "uncertainty"))
            refs = obs.get("evidence_refs")
            if not isinstance(refs, list) or not refs:
                issues.append(Pass1ValidationIssue(f"{path}.evidence_refs", "invalid_type", "evidence_refs must be a non-empty list", "uncertainty"))
            else:
                for idx, ref in enumerate(refs):
                    rpath = f"{path}.evidence_refs[{idx}]"
                    if not isinstance(ref, str):
                        issues.append(Pass1ValidationIssue(rpath, "invalid_type", "evidence_ref must be a string", "uncertainty"))
                    elif not ref.startswith(prefix):
                        issues.append(Pass1ValidationIssue(rpath, "invalid_reference", f"evidence_ref {ref} must start with {prefix} and belongs to a different source", "uncertainty"))
                    elif ref not in context.atom_entity_refs:
                        issues.append(Pass1ValidationIssue(rpath, "invalid_reference", f"Unknown evidence_ref: {ref}", "uncertainty"))
                    elif valid_entity and entity_id not in context.atom_entity_refs[ref]:
                        issues.append(Pass1ValidationIssue(rpath, "invalid_atom_entity_connection", f"Atom {ref} is not connected to entity {entity_id}", "uncertainty"))

        missing_attrs = analysis.get("missing_key_attributes") if isinstance(analysis.get("missing_key_attributes"), list) else []
        for index, attr in enumerate(missing_attrs):
            path = f"{source_key}.missing_key_attributes[{index}]"
            if not isinstance(attr, dict):
                issues.append(Pass1ValidationIssue(path, "invalid_type", "missing attribute must be an object", "missing_attribute"))
                continue
            required = {"entity_id", "attribute_type", "missing_attribute", "why_missing", "recoverable_evidence_refs"}
            for key in required - set(attr):
                issues.append(Pass1ValidationIssue(f"{path}.{key}", "missing_field", f"Missing required field: {key}", "missing_attribute"))
            for key in set(attr) - required:
                issues.append(Pass1ValidationIssue(f"{path}.{key}", "unexpected_field", f"Unexpected missing-attribute field: {key}", "missing_attribute"))
            entity_id = attr.get("entity_id")
            if "entity_id" in attr:
                if not isinstance(entity_id, str) or not entity_id.strip():
                    issues.append(Pass1ValidationIssue(f"{path}.entity_id", "invalid_type", "entity_id must be a non-empty string", "missing_attribute"))
                elif entity_id not in context.entity_ids:
                    issues.append(Pass1ValidationIssue(f"{path}.entity_id", "invalid_entity_reference", f"Unknown entity_id: {entity_id}", "missing_attribute"))
            if attr.get("attribute_type") not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
                issues.append(Pass1ValidationIssue(f"{path}.attribute_type", "invalid_type", "Invalid attribute_type", "missing_attribute"))
            for key in ("missing_attribute", "why_missing"):
                if key in attr and (not isinstance(attr[key], str) or not attr[key].strip()):
                    issues.append(Pass1ValidationIssue(f"{path}.{key}", "invalid_type", f"{key} must be a non-empty string", "missing_attribute"))
            refs = attr.get("recoverable_evidence_refs")
            if "recoverable_evidence_refs" in attr and (not isinstance(refs, list) or refs):
                issues.append(Pass1ValidationIssue(f"{path}.recoverable_evidence_refs", "invalid_type", "recoverable_evidence_refs must be exactly []", "missing_attribute"))

    if issues:
        # Stable de-duplication preserves the earliest, most useful path.
        unique = list({(i.path, i.category, i.message): i for i in issues}.values())
        raise Pass1StructuralValidationError("Structural Validation Errors", unique)
    return context


_COLOR_WORDS = (
    "red", "blue", "green", "yellow", "black", "white", "silver", "gray", "grey",
    "orange", "brown", "purple", "pink", "dark",
)
_MATERIAL_WORDS = ("wooden", "metal", "metallic", "concrete", "brick", "glass", "plastic")
_MOTION_WORDS = (
    "moving", "parked", "stationary", "stopped", "driving", "walking", "running",
    "approaching", "receding", "turning",
)
_IDENTITY_WORDS = (
    "pedestrian", "cyclist", "motorcyclist", "sedan", "suv", "truck", "van", "bus",
    "bicycle", "motorcycle", "motorhome", "campervan", "minivan", "hatchback",
)


def _extract_high_confidence_attributes(text: str) -> list[tuple[str, str]]:
    """Extract only the conservative shared-field categories named by the policy."""
    lowered = text.casefold()
    found: list[tuple[str, str]] = []
    for kind, terms in (
        ("color", _COLOR_WORDS),
        ("material", _MATERIAL_WORDS),
        ("motion", _MOTION_WORDS),
        ("identity", _IDENTITY_WORDS),
    ):
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                found.append((kind, term))
    for kind, pattern, value in (
        ("temperature", r"\b(?:hot|warm|cold|cool|heated|thermal)\b", "temperature"),
        ("headlights", r"\b(?:headlight|headlights|headlamp|headlamps)\b", "headlights"),
        ("reflection", r"\b(?:reflection|reflections|reflective|reflected)\b", "reflection"),
    ):
        if re.search(pattern, lowered):
            found.append((kind, value))
    return found


def _entity_aliases(context: Pass1ValidationContext, entity_id: str) -> set[str]:
    scope = context.entity_scopes.get(entity_id, "").casefold()
    aliases = {
        word for word in re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", scope)
        if word not in {"the", "this", "that", "left", "right", "near", "far", "front", "rear"}
    }
    category = context.entity_categories.get(entity_id, "").casefold().strip()
    if category:
        aliases.add(category)
    aliases.update(word[:-1] for word in list(aliases) if word.endswith("s") and len(word) > 4)
    aliases.add(entity_id.casefold())
    return aliases


def _entity_bound_source_facts(
    parsed: dict[str, Any],
    context: Pass1ValidationContext,
    source_key: str,
    entity_id: str | None,
) -> list[str]:
    """Return conservative Atom fragments; multi-Entity adjectives are locally bound."""
    facts: list[str] = []
    aliases = _entity_aliases(context, entity_id) if entity_id else set()
    for atom_id, source in context.atom_sources.items():
        if source != source_key:
            continue
        refs = context.atom_entity_refs.get(atom_id, set())
        if entity_id is not None and entity_id not in refs:
            continue
        fact = str(context.atom_facts.get(atom_id, ""))
        if entity_id is None or len(refs) == 1:
            facts.append(fact)
            continue
        # A multi-Entity Atom is usable only through a local clause naming the
        # target.  This prevents "a white car beside a blue bike" from assigning
        # both colors to both entities.
        for clause in re.split(r"(?<=[.;])\s+|\b(?:and|while|whereas|but)\b", fact, flags=re.I):
            clause_lower = clause.casefold()
            matches = [
                match
                for alias in sorted(aliases, key=len, reverse=True)
                if alias != entity_id.casefold()
                for match in [re.search(rf"\b{re.escape(alias)}\b", clause_lower)]
                if match
            ]
            if matches:
                match = matches[0]
                facts.append(clause[max(0, match.start() - 24):match.end() + 24])
    return facts


def _facts_support_attribute(kind: str, value: str, facts: list[str]) -> bool:
    for fact in facts:
        lowered = fact.casefold()
        if kind == "color":
            if not re.search(rf"\b{re.escape(value)}\b", lowered):
                continue
            if re.search(r"\b(?:headlights?|headlamps?|lights?|lamps?|sky|illumination)\b", lowered) and not re.search(
                r"\b(?:paint|painted|body|surface|facade|wall|clothing|coat|shirt)\b",
                lowered,
            ):
                continue
            return True
        if kind == "material" and re.search(rf"\b{re.escape(value)}\b", lowered):
            return True
        if kind == "temperature" and re.search(r"\b(?:hot|warm|cold|cool|heated|thermal|temperature)\b", lowered):
            return True
        if kind == "motion" and re.search(rf"\b{re.escape(value)}\b", lowered):
            return True
        if kind == "identity" and _conditional_recovery_support_status("visual_category", value, lowered) == "accept":
            return True
        if kind == "identity" and re.search(rf"\b{re.escape(value)}\b", lowered):
            return True
        if kind == "headlights" and re.search(r"\b(?:headlight|headlights|headlamp|headlamps)\b", lowered):
            return True
        if kind == "reflection" and re.search(r"\b(?:reflection|reflections|reflective|reflected)\b", lowered):
            return True
    return False


_VEHICLE_IDENTITY_TERMS = {
    "vehicle", "car", "automobile", "sedan", "suv", "truck", "van", "minivan",
    "motorhome", "campervan", "bus", "hatchback", "pickup", "utility vehicle",
    "delivery van",
}
_NON_VEHICLE_IDENTITY_TERMS = {
    "pallet", "pallets", "cargo", "crate", "crates", "container", "barrier",
    "installation", "cabinet", "stack",
}


def _text_contains_any(text: str, terms: set[str] | tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text, re.I) for term in terms)


def _validate_source_local_uncertainty_consistency(
    parsed: Mapping[str, Any],
    context: Pass1ValidationContext,
    issues: list[Pass1ValidationIssue],
    warnings: list[str],
) -> None:
    """Reject uncertainty that reopens a property resolved by same-source Atoms."""
    del warnings  # Reserved for deliberately ambiguous future checks.
    for source_key in context.expected_source_modalities:
        analysis = parsed.get(source_key, {})
        for index, uncertainty in enumerate(analysis.get("uncertain_observations", [])):
            if not isinstance(uncertainty, Mapping):
                continue
            entity_id = uncertainty.get("entity_id")
            if not isinstance(entity_id, str) or entity_id not in context.entity_ids:
                continue
            facts = _entity_bound_source_facts(dict(parsed), context, source_key, entity_id)
            if not facts:
                continue
            joined_facts = " ".join(facts)
            base_path = f"{source_key}.uncertain_observations[{index}]"
            observed = str(uncertainty.get("observed_evidence", ""))
            missing = str(uncertainty.get("missing_evidence", ""))

            fact_colors = [
                value for kind, value in _extract_high_confidence_attributes(joined_facts)
                if kind == "color" and _facts_support_attribute("color", value, facts)
            ]
            if fact_colors and re.search(r"\b(?:colou?r|paint|body colou?r|surface appearance)\b", missing, re.I):
                issues.append(Pass1ValidationIssue(
                    f"{base_path}.missing_evidence",
                    "source_uncertainty_contradiction",
                    f"Same-source Entity-bound Atom already asserts visible color ({fact_colors[0]}).",
                    "uncertainty",
                ))

            atom_asserts_illumination = re.search(
                r"\b(?:under|beneath|illuminated by|lit by|in the light of)\b.{0,35}\b"
                r"(?:streetlight|street light|lamp|light)\b|\b(?:illuminated|well-lit|lit)\b",
                joined_facts,
                re.I,
            )
            uncertainty_denies_illumination = re.search(
                r"\b(?:direct )?(?:illumination|lighting)\b|"
                r"\b(?:missing|no|without|lacks?|lack of|not visible)\b.{0,25}\blight\b",
                missing,
                re.I,
            )
            if atom_asserts_illumination and uncertainty_denies_illumination:
                issues.append(Pass1ValidationIssue(
                    f"{base_path}.missing_evidence",
                    "source_uncertainty_contradiction",
                    "Same-source Entity-bound Atom already asserts illumination of the Entity.",
                    "uncertainty",
                ))

            atom_establishes_vehicle = _text_contains_any(joined_facts, _VEHICLE_IDENTITY_TERMS)
            for hyp_index, hypothesis in enumerate(uncertainty.get("hypotheses", [])):
                if not isinstance(hypothesis, Mapping):
                    continue
                hypothesis_text = str(hypothesis.get("hypothesis", ""))
                if atom_establishes_vehicle and _text_contains_any(hypothesis_text, _NON_VEHICLE_IDENTITY_TERMS) \
                        and not _text_contains_any(hypothesis_text, _VEHICLE_IDENTITY_TERMS):
                    issues.append(Pass1ValidationIssue(
                        f"{base_path}.hypotheses[{hyp_index}].hypothesis",
                        "source_uncertainty_contradiction",
                        "Hypothesis reopens a non-vehicle coarse category after a same-source Atom establishes a vehicle.",
                        "uncertainty",
                    ))

            if fact_colors and re.search(r"\bdark\b.{0,20}\b(?:object|box|shape|body)\b", observed, re.I):
                # A dark silhouette can coexist with a visible paint assertion in another
                # sampled time, so this alone is not a hard contradiction.
                continue


def _validate_temporal_progression(text: str, warnings: list[str]) -> None:
    """Warn when temporal_progression is only a static scene inventory."""
    temporal_cues = (
        r"\b(?:initially|later|then|finally|subsequently|over time|across (?:the )?sampled times?)\b",
        r"\b(?:continues?|moves?|moving|approaches?|recedes?|turns?|enters?|exits?|appears?|disappears?)\b",
        r"\b(?:changes?|shifts?|transitions?|progress(?:ion|es)?|unfolds?)\b",
        r"\b(?:start|beginning)\b.{0,35}\b(?:finish|end)\b",
        r"\b(?:remains?|retains?|stays?)\b.{0,55}\b(?:across|throughout|over|sampled times?|sequence)\b",
        r"\b(?:same|stable|unchanged|consistent)\b.{0,35}\b(?:across|throughout|over|sampled times?|sequence)\b",
    )
    if not any(re.search(pattern, text, re.I) for pattern in temporal_cues):
        warnings.append(
            "weak_temporal_progression: global_scene.temporal_progression: "
            "text does not describe change, continued movement, or explicit stability across sampled times."
        )


def _validate_entity_reference_closure(
    context: Pass1ValidationContext,
    issues: list[Pass1ValidationIssue],
) -> None:
    """Require registered Entities that directly participate in an Atom relation."""
    relation_prefix = re.compile(
        r"(?:\bunder\b|\bbeneath\b|\bbeside\b|\bnext to\b|\bin front of\b|\bbehind\b|"
        r"\bacross from\b|\bagainst\b|\btoward\b|\bon (?:the )?(?:left|right) side of\b|"
        r"\balong\b|\bon\b|\bnear\b)\s+(?:a|an|the)?\s*$",
        re.I,
    )
    for atom_id, fact_value in context.atom_facts.items():
        fact = str(fact_value)
        refs = context.atom_entity_refs.get(atom_id, set())
        for entity_id in context.entity_ids - refs:
            aliases = sorted(_entity_aliases(context, entity_id), key=len, reverse=True)
            for alias in aliases:
                if alias == entity_id.casefold():
                    continue
                for match in re.finditer(rf"\b{re.escape(alias)}\b", fact, re.I):
                    prefix = fact[max(0, match.start() - 45):match.start()]
                    if relation_prefix.search(prefix):
                        issues.append(Pass1ValidationIssue(
                            f"{context.atom_paths.get(atom_id, atom_id)}.entity_refs",
                            "missing_entity_reference",
                            f"Atom relation names registered Entity {entity_id} ({alias!r}) but omits it from entity_refs.",
                            "atom",
                        ))
                        break
                else:
                    continue
                break


def _check_premature_registry_resolution(
    parsed: dict[str, Any],
    entity_id: str,
    registry_text: str,
    path: str,
    issues: list[Pass1ValidationIssue],
) -> None:
    registry_terms = set(re.findall(r"\b[a-z]{3,}\b", registry_text.casefold()))
    concept_groups = (
        {"vehicle", "car", "sedan", "suv", "truck", "van", "bus", "automobile"},
        {"person", "human", "pedestrian", "worker"},
        {"bicycle", "bike", "cyclist"},
        {"reflection", "reflective", "installation", "marker"},
    )

    def concepts(terms: set[str]) -> set[int]:
        return {index for index, group in enumerate(concept_groups) if terms & group}

    registry_concepts = concepts(registry_terms)
    for source_key in ("video1_analysis", "video2_analysis"):
        for uncertainty in parsed.get(source_key, {}).get("uncertain_observations", []):
            if not isinstance(uncertainty, dict) or uncertainty.get("entity_id") != entity_id:
                continue
            hypotheses = [
                str(item.get("hypothesis", "")).casefold()
                for item in uncertainty.get("hypotheses", []) if isinstance(item, dict)
            ]
            if len(hypotheses) < 2:
                continue
            hypothesis_terms = [set(re.findall(r"\b[a-z]{3,}\b", text)) for text in hypotheses]
            selected = [
                terms for terms in hypothesis_terms
                if registry_terms & terms or registry_concepts & concepts(terms)
            ]
            observed_terms = set(re.findall(r"\b[a-z]{3,}\b", str(uncertainty.get("observed_evidence", "")).casefold()))
            # Hard-fail only when the registry selects one candidate, competing
            # candidates are lexically incompatible, and shared observed evidence
            # does not already establish that selection.
            if len(selected) == 1 and not (registry_terms & observed_terms):
                issues.append(Pass1ValidationIssue(
                    path,
                    "premature_entity_resolution",
                    "Registry text selects one incompatible uncertainty hypothesis without shared evidence.",
                    "shared_neutrality",
                ))


def _validate_shared_neutrality(
    parsed: dict[str, Any],
    context: Pass1ValidationContext,
    issues: list[Pass1ValidationIssue],
    warnings: list[str],
) -> None:
    """Keep shared text neutral without treating ordinary nouns as source leakage."""
    global_scene = parsed["global_scene"]
    def _check_neutrality(text: str, path: str) -> None:
        if FORBIDDEN_SENSOR_QUALITY_PATTERN.search(text):
            issues.append(Pass1ValidationIssue(path, "sensor_quality_wording", "Shared field contains forbidden sensor-quality wording.", "shared_neutrality"))
        if FORBIDDEN_GLOBAL_SCENE_PATTERN.search(text):
            issues.append(Pass1ValidationIssue(path, "scene_level_wording", "Shared field contains forbidden scene-level terms.", "shared_neutrality"))

    _check_neutrality(global_scene.get("scene_summary", ""), "global_scene.scene_summary")
    _check_neutrality(global_scene.get("temporal_progression", ""), "global_scene.temporal_progression")
    for index, entity in enumerate(global_scene.get("physical_entities", [])):
        _check_neutrality(entity.get("referential_scope", ""), f"global_scene.physical_entities[{index}].referential_scope")

    def _locally_bound_entity(text: str, attribute_kind: str, attribute_value: str) -> str | None:
        candidates: set[str] = set()
        for candidate_id in context.entity_ids:
            aliases = _entity_aliases(context, candidate_id)
            for alias in aliases:
                if alias == candidate_id.casefold():
                    continue
                if attribute_kind == "identity":
                    patterns = (rf"\b{re.escape(attribute_value)}\b",) if alias == attribute_value else ()
                elif attribute_kind in {"color", "material", "temperature", "motion"}:
                    patterns = (
                        rf"\b{re.escape(attribute_value)}\b(?:\s+[a-z-]+){{0,2}}\s+\b{re.escape(alias)}\b",
                        rf"\b{re.escape(alias)}\b(?:\s+[a-z-]+){{0,3}}\s+\b{re.escape(attribute_value)}\b",
                    )
                else:
                    patterns = (
                        rf"\b{re.escape(alias)}\b.{{0,30}}\b{re.escape(attribute_value)}\b",
                        rf"\b{re.escape(attribute_value)}\b.{{0,30}}\b{re.escape(alias)}\b",
                    )
                if any(re.search(pattern, text, re.I) for pattern in patterns):
                    candidates.add(candidate_id)
                    break
        return next(iter(candidates)) if len(candidates) == 1 else None

    def _check_shared_leakage(text: str, path: str, entity_id: str | None = None) -> None:
        attributes = _extract_high_confidence_attributes(text)
        for attribute_kind, value in attributes:
            bound_entity_id = entity_id or _locally_bound_entity(text, attribute_kind, value)
            unsupported_sources: list[str] = []
            for source_key in context.expected_source_modalities:
                facts = _entity_bound_source_facts(parsed, context, source_key, bound_entity_id)
                if not _facts_support_attribute(attribute_kind, value, facts):
                    unsupported_sources.append(source_key)
            if not unsupported_sources:
                continue
            category = "registry_source_attribute_leakage" if entity_id else "shared_global_source_attribute_leakage"
            message = (
                f"Shared {attribute_kind} attribute {value!r} lacks Entity-bound Atom support "
                f"in {', '.join(unsupported_sources)}."
            )
            # A registry row identifies the entity unambiguously.  Free global prose
            # often does not, so ambiguous global binding remains a warning.
            if entity_id or bound_entity_id:
                issues.append(Pass1ValidationIssue(path, category, message, "shared_neutrality"))
            else:
                warnings.append(f"{category}: {path}: {message}")

        if entity_id:
            _check_premature_registry_resolution(parsed, entity_id, text, path, issues)

    _check_shared_leakage(global_scene.get("scene_summary", ""), "global_scene.scene_summary")
    _check_shared_leakage(global_scene.get("temporal_progression", ""), "global_scene.temporal_progression")
    for index, entity in enumerate(global_scene.get("physical_entities", [])):
        _check_shared_leakage(entity.get("referential_scope", ""), f"global_scene.physical_entities[{index}].referential_scope", entity.get("entity_id"))

def _validate_recoverability(
    parsed: dict[str, Any],
    context: Pass1ValidationContext,
    issues: list[Pass1ValidationIssue],
    warnings: list[str],
) -> None:
    """Validate curated targets against actual, Entity-bound opposite evidence."""

    def recovery_status(required_capability: str | None, target: str, facts: list[str]) -> str:
        joined = " ".join(facts)
        if not facts:
            return "reject"
        target_lower = target.casefold()
        if re.search(r"\b(?:make and model|make/model|manufacturer and model|specific model)\b", target_lower):
            return "accept" if re.search(
                r"\b(?:make|model|manufacturer|brand|badge|logo|emblem)\b.{0,45}\b"
                r"(?:reads?|states?|shows?|visible|legible|identif(?:y|ies|ied))\b|"
                r"\b(?:readable|legible|visible)\b.{0,30}\b(?:badge|logo|emblem|model name)\b",
                joined,
                re.I,
            ) else "reject"
        if re.search(r"\b(?:licen[cs]e plate|number plate|plate characters?|registration plate)\b", target_lower):
            return "accept" if re.search(
                r"\b(?:licen[cs]e|number|registration) plate\b.{0,55}\b"
                r"(?:reads?|characters?|text|number|readable|legible|visible)\b|"
                r"\b(?:readable|legible|visible)\b.{0,30}\b(?:plate characters?|plate text|registration)\b",
                joined,
                re.I,
            ) else "reject"
        if required_capability == "color":
            for _, color in _extract_high_confidence_attributes(joined):
                if color in _COLOR_WORDS and _facts_support_attribute("color", color, facts):
                    return "accept"
            if re.search(r"\b(?:headlights?|headlamps?|illumination|sky)\b", joined, re.I):
                return "reject"
            return "warn"
        if required_capability == "visual_category":
            return _conditional_recovery_support_status("visual_category", target, joined)
        if required_capability:
            return _conditional_recovery_support_status(required_capability, target, joined)
        if "motion" in target_lower:
            return "accept" if re.search(
                r"\b(?:moving|moves?|motion|parked|stationary|stopped|driving|changes? position|approaching|receding|turning)\b",
                joined,
                re.I,
            ) else "warn"
        if "exist" in target_lower:
            return "accept"
        return "warn"

    for source_key, _modality in context.expected_source_modalities.items():
        analysis = parsed.get(source_key, {})
        missing_attrs = analysis.get("missing_key_attributes", [])
        opposing_sources = [s for s in context.expected_source_modalities if s != source_key]

        for index, attr in enumerate(missing_attrs):
            path = f"{source_key}.missing_key_attributes[{index}]"
            entity_id = attr.get("entity_id")
            if not entity_id:
                continue
            attr_type = attr.get("attribute_type", "")
            target = str(attr.get("missing_attribute", ""))
            if not target:
                continue

            # Type-aware contradiction checks use word boundaries and include the
            # same-source Entity-bound facts plus why_missing.
            target_lower = target.casefold()
            why_missing = str(attr.get("why_missing", ""))
            same_source_facts = _entity_bound_source_facts(parsed, context, source_key, entity_id)
            local_evidence = " ".join([target, why_missing, *same_source_facts])
            if attr_type == "surface_attribute" and any(
                re.search(rf"\b{re.escape(color)}\b", local_evidence, re.I) for color in _COLOR_WORDS
            ):
                issues.append(Pass1ValidationIssue(f"{path}.missing_attribute", "missing_attribute_contradiction", f"Missing attribute '{target}' contains concrete value.", "semantic"))
            elif attr_type == "motion_state" and re.search(
                r"\b(?:parked|moving|stopped|driving|stationary|approaching|receding)\b",
                local_evidence,
                re.I,
            ):
                issues.append(Pass1ValidationIssue(f"{path}.missing_attribute", "missing_attribute_contradiction", f"Missing attribute '{target}' contains concrete value.", "semantic"))

            opposite_evidence = {
                opposite_source_key: _entity_bound_source_facts(
                    parsed, context, opposite_source_key, entity_id
                )
                for opposite_source_key in opposing_sources
            }
            if not any(opposite_evidence.values()):
                issues.append(Pass1ValidationIssue(path, "unrecoverable_missing_attribute", f"No opposite-source Atom references target entity {entity_id}.", "semantic"))
                continue

            req_cap = _infer_required_capability(attr_type, target)
            usable = False
            weak_sources: list[str] = []
            for opposite_source_key, facts in opposite_evidence.items():
                if not facts:
                    continue
                opposite_modality = context.expected_source_modalities[opposite_source_key]
                capability_state = MODALITY_CAPABILITIES.get(opposite_modality, {}).get(
                    req_cap, "conditional"
                ) if req_cap else "conditional"
                if capability_state == "not_direct":
                    continue
                status = recovery_status(req_cap, target, facts)
                if status == "accept":
                    usable = True
                elif status == "warn":
                    weak_sources.append(opposite_source_key)

            if usable:
                continue
            if weak_sources:
                warnings.append(
                    f"weak_cross_source_recoverability at {path}: target {target!r} for {entity_id} "
                    f"lacks an explicit compatible proposition in {weak_sources[0]}."
                )
            else:
                capability_note = ""
                if req_cap and all(
                    MODALITY_CAPABILITIES.get(context.expected_source_modalities[key], {}).get(req_cap, "conditional") == "not_direct"
                    for key, facts in opposite_evidence.items() if facts
                ):
                    capability_note = f" Opposite modality capability for {req_cap} is not_direct."
                issues.append(Pass1ValidationIssue(
                    path,
                    "unrecoverable_missing_attribute",
                    f"No opposite-source Entity-bound Atom contains compatible evidence for {target!r}.{capability_note}",
                    "semantic",
                ))

    _warn_for_missing_color_targets(parsed, context, warnings)


def _warn_for_missing_color_targets(
    parsed: dict[str, Any],
    context: Pass1ValidationContext,
    warnings: list[str],
) -> None:
    """Conservative recall warning for clear RGB paint/color evidence only."""
    source_keys = list(context.expected_source_modalities)
    for source_key in source_keys:
        modality = context.expected_source_modalities[source_key]
        if MODALITY_CAPABILITIES.get(modality, {}).get("color") != "direct":
            continue
        for opposite_source_key in source_keys:
            if opposite_source_key == source_key:
                continue
            existing_targets = parsed[opposite_source_key].get("missing_key_attributes", [])
            for entity_id in context.entity_ids:
                source_facts = _entity_bound_source_facts(parsed, context, source_key, entity_id)
                opposite_facts = _entity_bound_source_facts(parsed, context, opposite_source_key, entity_id)
                source_colors = [
                    value for kind, value in _extract_high_confidence_attributes(" ".join(source_facts))
                    if kind == "color" and _facts_support_attribute("color", value, source_facts)
                ]
                if not source_colors or not opposite_facts:
                    continue
                if any(_facts_support_attribute("color", color, opposite_facts) for color in source_colors):
                    continue
                has_target = any(
                    isinstance(item, dict)
                    and item.get("entity_id") == entity_id
                    and _infer_required_capability(
                        str(item.get("attribute_type", "")),
                        str(item.get("missing_attribute", "")),
                    ) == "color"
                    for item in existing_targets
                )
                if not has_target:
                    warnings.append(
                        f"possible_missing_recoverability_target in {opposite_source_key}: "
                        f"Entity {entity_id} has explicit paint/color evidence in {source_key}."
                    )

def _validate_pass1_semantics(parsed: dict[str, Any], context: Pass1ValidationContext) -> list[str]:
    """Run semantic-only checks after a successful structural stage."""
    issues: list[Pass1ValidationIssue] = []
    warnings: list[str] = []

    def check(text: str, path: str, modality: str | None = None) -> None:
        if not text: return
        for pattern in FORBIDDEN_MECHANISM_PATTERNS:
            match = pattern.search(text)
            if match:
                issues.append(Pass1ValidationIssue(path, "physical_world_wording", f"Forbidden mechanism-oriented wording {match.group(0)!r}", "semantic"))
        if REPRESENTATION_SUBJECT_PATTERN.search(text) and REPRESENTATION_PREDICATE_PATTERN.search(text):
            issues.append(Pass1ValidationIssue(path, "physical_world_wording", "Forbidden representation-oriented construction", "semantic"))
        if FORBIDDEN_INFERENTIAL_PATTERN.search(text):
            issues.append(Pass1ValidationIssue(path, "forbidden_inferential_terms", "Forbidden inferential wording", "semantic"))
        if any(pattern.search(text) for pattern in GENERIC_SENSOR_EXPLANATION_PATTERNS):
            issues.append(Pass1ValidationIssue(path, "generic_sensor_theory", "Generic sensor-theory wording", "semantic"))
        if modality is not None:
            _validate_modality_specific_physical_claims(modality, text, path, issues, warnings)

    _validate_shared_neutrality(parsed, context, issues, warnings)
    _validate_source_local_uncertainty_consistency(parsed, context, issues, warnings)
    _validate_entity_reference_closure(context, issues)

    global_scene = parsed["global_scene"]
    _validate_temporal_progression(str(global_scene.get("temporal_progression", "")), warnings)
    for key in ("scene_summary", "environment", "temporal_progression"):
        check(str(global_scene.get(key, "")), f"global_scene.{key}")
    for index, entity in enumerate(global_scene.get("physical_entities", [])):
        check(str(entity.get("category", "")), f"global_scene.physical_entities[{index}].category")
        check(str(entity.get("referential_scope", "")), f"global_scene.physical_entities[{index}].referential_scope")

    for source_key, modality in context.expected_source_modalities.items():
        analysis = parsed[source_key]
        check(str(analysis.get("detailed_caption", "")), f"{source_key}.detailed_caption", modality)

        # Validator C: unimodal uncertainty consistency
        det_cap_lower = str(analysis.get("detailed_caption", "")).lower()
        for i, obs in enumerate(analysis.get("uncertain_observations", [])):
            if not isinstance(obs, dict): continue
            check(str(obs.get("observed_evidence", "")), f"{source_key}.uncertain_observations[{i}].observed_evidence", modality)
            check(str(obs.get("missing_evidence", "")), f"{source_key}.uncertain_observations[{i}].missing_evidence", modality)
            for j, hyp_dict in enumerate(obs.get("hypotheses", [])):
                if not isinstance(hyp_dict, dict): continue
                hyp_text = str(hyp_dict.get("hypothesis", "")).strip()
                check(hyp_text, f"{source_key}.uncertain_observations[{i}].hypotheses[{j}].hypothesis", modality)
                if len(hyp_text.split()) >= 2 and hyp_text.lower() in det_cap_lower:
                    issues.append(Pass1ValidationIssue(
                        f"{source_key}.detailed_caption", "uncertainty_presented_as_fact",
                        f"presents uncertain hypothesis '{hyp_text}' as fact.", "semantic"
                    ))

        # Validator D: caption-to-atom grounding (soft warning)
        caption_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", det_cap_lower))
        atoms = analysis.get("information_atoms", [])
        atom_text = " ".join([str(a.get("fact", "")) for a in atoms if isinstance(a, dict)]).lower()
        atom_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", atom_text))
        ungrounded = caption_words - atom_words - GROUNDING_EXEMPT_WORDS
        if len(ungrounded) > 10:
             warnings.append(f"{source_key}.detailed_caption may contain ungrounded claims. Words not in atoms: {', '.join(sorted(list(ungrounded))[:5])}...")
             warnings.append("caption_paraphrase_grounding")

        for index, atom in enumerate(analysis.get("information_atoms", [])):
            check(str(atom.get("fact", "")), f"{source_key}.information_atoms[{index}].fact", modality)
        for index, cue in enumerate(analysis.get("sensor_specific_cues", [])):
            if any(pattern.search(str(cue)) for pattern in GENERIC_SENSOR_EXPLANATION_PATTERNS):
                issues.append(Pass1ValidationIssue(f"{source_key}.sensor_specific_cues[{index}]", "generic_sensor_theory", "Generic sensor-theory wording", "sensor_specific_cue"))
        for index, limitation in enumerate(analysis.get("sensor_limitations", [])):
            limitation_text = str(limitation)
            limitation_path = f"{source_key}.sensor_limitations[{index}]"
            check(limitation_text, limitation_path)
            entity_categories = {
                str(entity.get("category", ""))
                for entity in global_scene.get("physical_entities", []) if isinstance(entity, dict)
            }
            anchors = set(context.entity_ids) | set(context.entity_scopes.values()) | entity_categories | context.atom_frame_keys.keys()
            has_anchor = any(str(anchor).casefold() in limitation_text.casefold() for anchor in anchors)
            has_frame_anchor = any(frame.casefold() in limitation_text.casefold() for frame in {
                frame for frames in context.atom_frame_keys.values() for frame in frames
            })
            if not has_anchor and not has_frame_anchor:
                issues.append(Pass1ValidationIssue(
                    limitation_path,
                    "segment_specificity",
                    "sensor limitation must anchor to a known Entity, frame, or segment region.",
                    "sensor_limitation",
                ))
            if re.search(r"\bstatic objects?\b.{0,40}\b(?:without|no)\b.{0,30}\b(?:motion|movement)\b.{0,20}\b(?:invisible|not visible)\b", limitation_text, re.I):
                issues.append(Pass1ValidationIssue(
                    limitation_path,
                    "generic_sensor_theory",
                    "Generic sensor-theory wording is not a segment-specific limitation.",
                    "sensor_limitation",
                ))
        for index, attr in enumerate(analysis.get("missing_key_attributes", [])):
            check(str(attr.get("missing_attribute", "")), f"{source_key}.missing_key_attributes[{index}].missing_attribute")
            check(str(attr.get("why_missing", "")), f"{source_key}.missing_key_attributes[{index}].why_missing")

    _validate_recoverability(parsed, context, issues, warnings)

    if issues:
        unique = list({(i.path, i.category, i.message): i for i in issues}.values())
        raise Pass1SemanticValidationError("Semantic Validation Errors", unique)
    return warnings

GROUNDING_EXEMPT_WORDS = {
    "initially", "later", "finally", "previously", "subsequently",
    "along", "across", "through", "around", "between", "beyond",
    "within", "toward", "beside", "against",
    "bright", "clear", "broad", "large", "small", "short",
    "various", "several", "multiple", "additional", "further",
    "including", "passing", "following",
    "progresses", "advances", "continues", "remains",
}

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


def _validate_pass1_uncertain_observations(
    values: Any,
    field: str,
    raw_issues: list[Pass1ValidationIssue],
    *,
    entity_ids: set[str],
    evidence_namespace: set[str],
    atom_entity_refs: dict[str, set[str]],
    atom_prefix: str,
    register_evidence_id: Any,
) -> None:
    _validate_uncertain_observations(values, field, min_hypotheses=0)
    for index, item in enumerate(values or [], start=1):
        if not isinstance(item, dict): continue
        unc_id = _require_string(item.get("uncertainty_id"), f"{field}[{index}].uncertainty_id")
        unc_prefix = atom_prefix.replace("atom", "unc")
        if not unc_id.startswith(unc_prefix):
            raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].uncertainty_id", "invalid_id_format", f"must start with {unc_prefix}", scope=field))
        register_evidence_id(unc_id)
        
        ent_id = _require_string(item.get("entity_id"), f"{field}[{index}].entity_id")
        if ent_id not in entity_ids:
            raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].entity_id", "invalid_entity_reference", f"references unknown entity: {ent_id}", scope=field))
            
        ev_refs = _require_list(item.get("evidence_refs"), f"{field}[{index}].evidence_refs")
        if not ev_refs:
            raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].evidence_refs", "empty_list", "evidence_refs must be a non-empty list", scope=field))
            
        seen_refs = set()
        for ref_idx, ref in enumerate(ev_refs, start=1):
            ref_val = _require_string(ref, f"{field}[{index}].evidence_refs[{ref_idx}]")
            if ref_val in seen_refs:
                raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].evidence_refs", "duplicate_reference", f"contains duplicate: {ref_val}", scope=field))
            seen_refs.add(ref_val)
            if not ref_val.startswith(atom_prefix):
                raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].evidence_refs", "invalid_id_format", f"evidence_refs {ref_val} must start with {atom_prefix}", scope=field))
            if ref_val not in atom_entity_refs:
                raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].evidence_refs", "invalid_atom_reference", f"references unknown atom: {ref_val}", scope=field))
                continue
            if ent_id not in atom_entity_refs[ref_val]:
                raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].evidence_refs", "missing_entity_connection", f"not connected to entity {ent_id}", scope=field))

        hypotheses = item.get("hypotheses") or []
        normalized_hyps: set[str] = set()
        for hyp in hypotheses:
            if not isinstance(hyp, dict): continue
            hyp_text = hyp.get("hypothesis")
            if isinstance(hyp_text, str) and hyp_text.strip():
                norm_meta = hyp_text.strip().casefold()
                # Check for meta-uncertainty
                if re.search(r'\b(cannot be determined|is unknown|more evidence is needed|cannot be definitively determined)\b', norm_meta):
                    raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].hypotheses", "meta_statement", "contains meta-statement", scope=field))
                norm = _normalize_pass1_hypothesis(hyp_text)
                normalized_hyps.add(norm)
        if hypotheses and len(normalized_hyps) < 2:
            raw_issues.append(Pass1ValidationIssue(f"{field}[{index}].hypotheses", "insufficient_hypotheses", "must contain at least 2", scope=field))


def _validate_pass1_why_missing(text: str, field: str, raw_issues: list[Pass1ValidationIssue]) -> None:
    for pattern in GENERIC_SENSOR_EXPLANATION_PATTERNS:
        match = pattern.search(text)
        if match:
            raw_issues.append(Pass1ValidationIssue(field, "sensor_theory_wording", f"Generic sensor-theory wording in {field}", scope=field))
            break
    if re.search(r'\b(sensing )?modality does not (record|capture|detect|provide)\b', text, re.I):
        raw_issues.append(Pass1ValidationIssue(field, "generic_theory", "must be segment-specific", scope=field))
    
    generic_process_patterns = [
        re.compile(r"\b(the|this)\s+(sensing|imaging)\s+process\b", re.I),
        re.compile(r"\bthe\s+sensor\s+(captures|records)\b", re.I),
        re.compile(r"\bdoes\s+not\s+preserve\s+(static\s+)?color\b", re.I),
        re.compile(r"\bimaging\s+process\s+records\b", re.I),
        re.compile(r"\bsensor\s+captures\s+changes\b", re.I),
        re.compile(r"\b(does|do)\s+not\s+(register|detect|record)\s+(static|absolute|surface)\b", re.I),
        re.compile(r"\bphysical\s+instrument\s+(does|do)\s+not\b", re.I),
        re.compile(r"\bnot\s+possible\s+to\s+(distinguish|identify|determine)\s+.{0,30}\b(without|from)\s+(color|intensity|additional)\b", re.I)
    ]
    for pat in generic_process_patterns:
        if pat.search(text):
            raw_issues.append(Pass1ValidationIssue(field, "generic_theory", f"found generic explanation pattern: '{pat.pattern}'", scope=field))


def _validate_pass1_schema(
    parsed: dict[str, Any],
    valid_frame_keys: set[str],
    expected_source_modalities: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate Pass 1 structural schema, cross-references, and physical wording."""
    context = _validate_pass1_structure(
        parsed,
        valid_frame_keys,
        expected_source_modalities,
    )
    warnings = _validate_pass1_semantics(parsed, context)
    return parsed, warnings
