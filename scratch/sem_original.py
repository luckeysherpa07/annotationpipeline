def _validate_pass1_semantics(parsed: dict[str, Any], context: Pass1ValidationContext) -> list[str]:
    """Run semantic-only checks after a successful structural stage."""
    issues: list[Pass1ValidationIssue] = []
    warnings: list[str] = []

    def check(text: str, path: str, modality: str | None = None) -> None:
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

    global_scene = parsed["global_scene"]
    for key in ("scene_summary", "environment", "temporal_progression"):
        check(global_scene[key], f"global_scene.{key}")
    for index, entity in enumerate(global_scene["physical_entities"]):
        check(entity["category"], f"global_scene.physical_entities[{index}].category")
        check(entity["referential_scope"], f"global_scene.physical_entities[{index}].referential_scope")

    for source_key, modality in context.expected_source_modalities.items():
        analysis = parsed[source_key]
        check(analysis["detailed_caption"], f"{source_key}.detailed_caption", modality)
        for index, atom in enumerate(analysis["information_atoms"]):
            check(atom["fact"], f"{source_key}.information_atoms[{index}].fact", modality)
        for index, cue in enumerate(analysis["sensor_specific_cues"]):
            if any(pattern.search(cue) for pattern in GENERIC_SENSOR_EXPLANATION_PATTERNS):
                issues.append(Pass1ValidationIssue(f"{source_key}.sensor_specific_cues[{index}]", "generic_sensor_theory", "Generic sensor-theory wording", "sensor_specific_cue"))
        for index, limitation in enumerate(analysis["sensor_limitations"]):
            check(limitation, f"{source_key}.sensor_limitations[{index}]")
        for index, attr in enumerate(analysis["missing_key_attributes"]):
            check(attr["missing_attribute"], f"{source_key}.missing_key_attributes[{index}].missing_attribute")
            check(attr["why_missing"], f"{source_key}.missing_key_attributes[{index}].why_missing")

    if issues:
        unique = list({(i.path, i.category, i.message): i for i in issues}.values())
        raise Pass1SemanticValidationError("Semantic Validation Errors", unique)
    return warnings
