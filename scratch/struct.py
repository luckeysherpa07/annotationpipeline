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

    normalized_expected = {
        key: normalize_modality_name(value) if isinstance(value, str) else ""
        for key, value in expected_source_modalities.items()
    }
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
