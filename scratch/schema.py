def _validate_pass1_schema(
    parsed: dict[str, Any],
    valid_frame_keys: set[str],
    expected_source_modalities: Mapping[str, str] | str,
    expected_modality2: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if isinstance(expected_source_modalities, str):
        if expected_modality2 is None:
            raise TypeError("expected_modality2 is required with the legacy string call form")
        expected_source_modalities = {
            "video1_analysis": expected_source_modalities,
            "video2_analysis": expected_modality2,
        }
    # Structural staging is deliberately completed before any semantic work so
    # malformed containers cannot cascade into misleading semantic failures.
    _validate_pass1_structure(parsed, valid_frame_keys, expected_source_modalities)
    policy_mechanism = "hard_fail"
    policy_inferential = "hard_fail"

    raw_issues: list[Pass1ValidationIssue] = []
    if not valid_frame_keys:
        raw_issues.append(Pass1ValidationIssue("valid_frame_keys", "empty_list", "valid_frame_keys must not be empty", scope="global"))

    normalized_expected: dict[str, str] = {}
    for source_key in ("video1_analysis", "video2_analysis"):
        raw_modality = expected_source_modalities.get(source_key)
        normalized = normalize_modality_name(raw_modality) if isinstance(raw_modality, str) else ""
        if normalized not in SUPPORTED_MODALITIES:
            raw_issues.append(Pass1ValidationIssue(source_key, "unsupported_modality", f"Unsupported expected modality: {raw_modality!r}", scope=source_key))
        else:
            normalized_expected[source_key] = normalized
    expected_source_modalities = MappingProxyType(normalized_expected)
    atom_frame_keys: dict[str, set[str]] = {}
    atom_facts: dict[str, str] = {}
    atom_sources: dict[str, str] = {}
    atom_entity_refs: dict[str, set[str]] = {}
    local_warnings: list[str] = []

    missing = [field for field in PASS1_REQUIRED_TOP_LEVEL_FIELDS if field not in parsed]
    if missing:
        for _m in missing:
            raw_issues.append(Pass1ValidationIssue(f"{_m}", "missing_field", "Missing required Pass 1 field.", scope="global"))
        
    unexpected_fields = set(parsed.keys()) - PASS1_REQUIRED_TOP_LEVEL_FIELDS
    if unexpected_fields:
        for uf in unexpected_fields:
            raw_issues.append(Pass1ValidationIssue(uf, "unknown_field", "unknown top-level field", scope="global"))

    evidence_namespace: set[str] = set()

    