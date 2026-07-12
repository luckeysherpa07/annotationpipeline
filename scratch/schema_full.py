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

    def _register_evidence_id(eid: str, path: str = "evidence") -> None:
        if eid in evidence_namespace:
            raw_issues.append(Pass1ValidationIssue(path, "duplicate_evidence_id", f"Duplicate evidence ID: {eid}", scope="global"))
        evidence_namespace.add(eid)

    global_scene = _require_object(parsed["global_scene"], "global_scene")
    
    global_scene_unexpected = set(global_scene.keys()) - PASS1_GLOBAL_SCENE_ALLOWED_FIELDS
    if global_scene_unexpected:
        for uf in global_scene_unexpected:
            raw_issues.append(Pass1ValidationIssue(f"global_scene.{uf}", "unknown_field", "unknown field", scope="global_scene"))

    scene_summary = _validate_min_words(global_scene.get("scene_summary"), "global_scene.scene_summary", MIN_SCENE_SUMMARY_WORDS)
    if FORBIDDEN_SENSOR_QUALITY_PATTERN.search(scene_summary):
        raw_issues.append(Pass1ValidationIssue("global_scene.scene_summary", "sensor_quality_wording", "forbidden sensor-quality wording", scope="global_scene"))
    if FORBIDDEN_GLOBAL_SCENE_PATTERN.search(scene_summary):
        raw_issues.append(Pass1ValidationIssue("global_scene.scene_summary", "scene_level_wording", "forbidden scene-level terms", scope="global_scene"))
    _require_string(global_scene.get("environment"), "global_scene.environment")
    temporal_progression = _validate_min_words(global_scene.get("temporal_progression"), "global_scene.temporal_progression", MIN_FRAME_DETAIL_WORDS)
    if FORBIDDEN_SENSOR_QUALITY_PATTERN.search(temporal_progression):
        raw_issues.append(Pass1ValidationIssue("global_scene.temporal_progression", "sensor_quality_wording", "forbidden sensor-quality wording", scope="global_scene"))
    if FORBIDDEN_GLOBAL_SCENE_PATTERN.search(temporal_progression):
        raw_issues.append(Pass1ValidationIssue("global_scene.temporal_progression", "scene_level_wording", "forbidden scene-level terms", scope="global_scene"))
    
    physical_entities = _require_list(global_scene.get("physical_entities"), "global_scene.physical_entities")
    if not physical_entities:
        raw_issues.append(Pass1ValidationIssue("global_scene.physical_entities", "empty_list", "must not be empty", scope="global_scene"))
    entity_ids: set[str] = set()
    normalized_entity_scopes: dict[str, str] = {}
    for index, entity in enumerate(physical_entities, start=1):
        if not isinstance(entity, dict):
            raw_issues.append(Pass1ValidationIssue(f"global_scene.physical_entities[{index}]", "invalid_type", "must be an object", scope="global_scene"))
        entity_id = _require_string(entity.get("entity_id"), f"global_scene.physical_entities[{index}].entity_id")
        if entity_id in entity_ids:
            raw_issues.append(Pass1ValidationIssue(f"global_scene.physical_entities[{index}].entity_id", "duplicate_entity_id", f"Duplicate entity_id: {entity_id}", scope="global_scene"))
        entity_ids.add(entity_id)
        _require_string(entity.get("category"), f"global_scene.physical_entities[{index}].category")
        referential_scope = _require_string(entity.get("referential_scope"), f"global_scene.physical_entities[{index}].referential_scope")
        normalized_scope = _normalize_referential_scope(referential_scope)
        existing_entity_id = normalized_entity_scopes.get(normalized_scope)
        if existing_entity_id is not None and existing_entity_id != entity_id:
            raw_issues.append(Pass1ValidationIssue(f"global_scene.physical_entities[{index}].referential_scope", "duplicate_normalized_scope", f"Duplicate normalized referential_scope detected for {entity_id} and {existing_entity_id}: {referential_scope!r}", scope="global_scene"))
        normalized_entity_scopes[normalized_scope] = entity_id
        if "evidence_profile" in entity:
            prof = _require_object(entity.get("evidence_profile"), f"global_scene.physical_entities[{index}].evidence_profile")
            if not prof:
                raw_issues.append(Pass1ValidationIssue(f"global_scene.physical_entities[{index}].evidence_profile", "empty_evidence_profile", "evidence_profile must not be empty if present.", scope="global_scene"))
            for prof_key in ("identity_evidence", "observable_attributes", "spatial_context"):
                if prof_key in prof:
                    ev_list = _require_list(prof[prof_key], f"global_scene.physical_entities[{index}].evidence_profile.{prof_key}")
                    if not ev_list:
                        raw_issues.append(Pass1ValidationIssue(f"global_scene.physical_entities[{index}].evidence_profile.{prof_key}", "empty_list", f"evidence_profile.{prof_key} must not be empty if present.", scope="global_scene"))
                    for j, s in enumerate(ev_list, start=1):
                        _require_string(s, f"evidence_profile.{prof_key}[{j}]")

    for source_key, expected_mod in expected_source_modalities.items():
        analysis = _require_object(parsed.get(source_key), source_key)
        modality = _require_string(analysis.get("modality"), f"{source_key}.modality")
        if normalize_modality_name(modality) != expected_mod:
            raw_issues.append(Pass1ValidationIssue(f"{source_key}.modality", "modality_mismatch", f"modality {modality!r} does not match expected {expected_mod!r}", scope=source_key))

    def _validate_video_analysis(parsed_obj: dict[str, Any], field: str, atom_prefix: str) -> None:
        analysis = _require_object(parsed_obj.get(field), field)
        
        analysis_unexpected = set(analysis.keys()) - PASS1_VIDEO_ANALYSIS_ALLOWED_FIELDS
        if analysis_unexpected:
            for uf in analysis_unexpected:
                raw_issues.append(Pass1ValidationIssue(f"{field}.{uf}", "unknown_field", "unknown field", scope=field))

        _require_string(analysis.get("modality"), f"{field}.modality")
        detailed_caption = _validate_min_words(
            analysis.get("detailed_caption"),
            f"{field}.detailed_caption",
            MIN_DETAILED_CAPTION_WORDS,
        )
        if FORBIDDEN_SENSOR_QUALITY_PATTERN.search(detailed_caption):
            raw_issues.append(Pass1ValidationIssue(f"{field}.detailed_caption", "sensor_quality_wording", "forbidden sensor-quality wording", scope=field))
        
        atoms = _require_list(analysis.get("information_atoms"), f"{field}.information_atoms")
        if not atoms:
            raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms", "empty_list", "information_atoms must not be empty.", scope=field))
            
        for i, atom in enumerate(atoms, start=1):
            if not isinstance(atom, dict):
                raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms[{i}]", "invalid_type", "must be an object", scope=field))
            atom_id = _require_string(atom.get("atom_id"), f"{field}.information_atoms[{i}].atom_id")
            if not atom_id.startswith(atom_prefix):
                raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms[{i}].atom_id", "invalid_id_format", f"must start with {atom_prefix}", scope=field))
            _register_evidence_id(atom_id, f"{field}.information_atoms[{i}].atom_id")
            f_keys = _require_list(atom.get("frame_keys"), f"{field}.information_atoms[{i}].frame_keys")
            if not f_keys:
                raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms[{i}].frame_keys", "empty_list", "frame_keys must be a non-empty list", scope=field))

            for fk in f_keys:
                if fk not in valid_frame_keys:
                    raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms[{i}].frame_keys", "invalid_frame_reference", f"Unknown frame_key '{fk}'", scope=field))
            atom_frame_keys[atom_id] = set(f_keys)
            entity_refs = _require_list(atom.get("entity_refs"), f"{field}.information_atoms[{i}].entity_refs")
            if not entity_refs:
                raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms[{i}].entity_refs", "empty_list", "entity_refs must be a non-empty list", scope=field))
            seen_atom_entities: set[str] = set()
            for entity_ref_index, entity_ref in enumerate(entity_refs, start=1):
                ref_value = _require_string(entity_ref, f"{field}.information_atoms[{i}].entity_refs[{entity_ref_index}]")
                if ref_value in seen_atom_entities:
                    raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms[{i}].entity_refs", "duplicate_entity_id", f"Duplicate entity_id: {ref_value}", scope=field))
                if ref_value not in entity_ids:
                    raw_issues.append(Pass1ValidationIssue(f"{field}.information_atoms[{i}].entity_refs", "invalid_entity_reference", f"References unknown entity: {ref_value}", scope=field))
                seen_atom_entities.add(ref_value)
            atom_entity_refs[atom_id] = seen_atom_entities
            fact = _require_string(atom.get("fact"), f"{field}.information_atoms[{i}].fact")
            atom_facts[atom_id] = fact
            atom_sources[atom_id] = field

        for key in ("sensor_specific_cues", "sensor_limitations"):
            values = _require_list(analysis.get(key), f"{field}.{key}")
            for value_index, value in enumerate(values, start=1):
                item_field = f"{field}.{key}[{value_index}]"
                text = _require_string(value, item_field)
                pass
        _validate_pass1_uncertain_observations(
            analysis.get("uncertain_observations"),
            f"{field}.uncertain_observations",
            raw_issues,
            entity_ids=entity_ids,
            evidence_namespace=evidence_namespace,
            atom_entity_refs=atom_entity_refs,
            atom_prefix=atom_prefix,
            register_evidence_id=_register_evidence_id,
        )
        
        missing_attrs = _require_list(analysis.get("missing_key_attributes"), f"{field}.missing_key_attributes")
        for i, attr in enumerate(missing_attrs, start=1):
            if not isinstance(attr, dict):
                raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i}]", "invalid_type", "must be an object", scope=field))
                
            expected_fields = {"entity_id", "attribute_type", "missing_attribute", "why_missing", "recoverable_evidence_refs"}
            actual_fields = set(attr.keys())
            missing_fields = expected_fields - actual_fields
            unexpected_fields = actual_fields - expected_fields
            
            if missing_fields or unexpected_fields:
                err_parts = []
                if missing_fields:
                    err_parts.append(f"Missing required fields: {sorted(missing_fields)}")
                if unexpected_fields:
                    err_parts.append(f"Unexpected fields: {sorted(unexpected_fields)}")
                raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i}]", "invalid_structure", f"{'; '.join(err_parts)}", scope=field))

            if "entity_id" not in attr:
                raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i}].entity_id", "missing_field", "Missing required field: entity_id", scope=field))

            entity_id = attr.get("entity_id")
            if entity_id is not None and (not isinstance(entity_id, str) or not entity_id.strip()):
                raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i}].entity_id", "invalid_type", "entity_id must be a non-empty string", scope=field))
            elif entity_id not in entity_ids:
                raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i}].entity_id", "invalid_entity_reference", f"Entity ID '{entity_id}' not found in global_scene", scope=field))

            attr_type = attr.get("attribute_type")
            if attr_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
                raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i}].attribute_type", "invalid_value", f"invalid attribute type: {attr_type}", scope=field))
            _require_string(attr.get("missing_attribute"), f"{field}.missing_key_attributes[{i}].missing_attribute")
            _require_string(attr.get("why_missing"), f"{field}.missing_key_attributes[{i}].why_missing")
            _validate_pass1_why_missing(attr.get("why_missing"), f"{field}.missing_key_attributes[{i}].why_missing", raw_issues)
            
            recoverable_refs = _require_list(attr.get("recoverable_evidence_refs"), f"{field}.missing_key_attributes[{i}].recoverable_evidence_refs")
            if recoverable_refs:
                raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i}].recoverable_evidence_refs", "invalid_value", "MUST be empty in Pass 1", scope=field))

        # Validator C: unimodal uncertainty consistency
        for i, obs in enumerate(analysis.get("uncertain_observations") or [], start=1):
            if not isinstance(obs, dict): continue
            for hyp_dict in obs.get("hypotheses") or []:
                if not isinstance(hyp_dict, dict): continue
                hyp_text = str(hyp_dict.get("hypothesis", "")).strip()
                if len(hyp_text.split()) >= 2 and hyp_text.lower() in detailed_caption.lower():
                    raw_issues.append(Pass1ValidationIssue(f"{field}.detailed_caption", "uncertainty_leakage", f"presents uncertain hypothesis '{hyp_text}' as fact", scope=field))
                    

        # Validator C2: cross-source recoverability check
        for field in expected_source_modalities:
            analysis = parsed.get(field, {})
            missing_attrs = analysis.get("missing_key_attributes")
            if not isinstance(missing_attrs, list): continue
            
            opposing_sources = [s for s in expected_source_modalities if s != field]
            if not opposing_sources: continue
            
            for i, attr in enumerate(missing_attrs, start=1):
                if not isinstance(attr, dict): continue
                entity_id = attr.get("entity_id")
                if not isinstance(entity_id, str): continue
                
                attr_type = str(attr.get("attribute_type", ""))
                missing_attr = str(attr.get("missing_attribute", "")).lower()
                
                # Check missing_attribute_contradiction
                contradiction_words = ["white", "black", "red", "blue", "green", "yellow", "silver", "gray", "grey", "parked", "moving", "stopped", "driving"]
                if any(term in missing_attr for term in contradiction_words):
                    raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i-1}].missing_attribute", "missing_attribute_contradiction", f"Missing attribute '{missing_attr}' contains concrete value.", scope="missing_attribute"))
                
                # Check if target entity exists in any opposing source
                entity_has_opposing_atoms = False
                for opp_src in opposing_sources:
                    for atom_id, entity_refs in atom_entity_refs.items():
                        if atom_sources.get(atom_id) == opp_src and entity_id in entity_refs:
                            entity_has_opposing_atoms = True
                            break
                    if entity_has_opposing_atoms:
                        break
                        
                if not entity_has_opposing_atoms:
                    raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i-1}]", "unrecoverable_missing_attribute", f"No opposite-source Atom references target entity {entity_id}.", scope="missing_attribute"))
                    continue
                    
                req_cap = _infer_required_capability(attr_type, missing_attr)
                
                if req_cap:
                    all_opposing_not_direct = True
                    any_opposing_conditional = False
                    conditional_support_status = None
                    
                    for opp_src in opposing_sources:
                        opp_mod = expected_source_modalities[opp_src]
                        cap_state = MODALITY_CAPABILITIES.get(opp_mod, {}).get(req_cap, "conditional")
                        opp_facts = " ".join(str(atom_facts.get(a, "")) for a, e_refs in atom_entity_refs.items() if atom_sources.get(a) == opp_src and entity_id in e_refs).casefold()
                        if cap_state != "not_direct":
                            all_opposing_not_direct = False
                        if cap_state == "conditional":
                            any_opposing_conditional = True
                            conditional_support_status = _conditional_recovery_support_status(req_cap, missing_attr, opp_facts)
                        elif cap_state == "direct" and req_cap == "visual_category":
                            direct_status = _conditional_recovery_support_status(req_cap, missing_attr, opp_facts)
                            if direct_status == "warn":
                                local_warnings.append(f"{field}.missing_key_attributes[{i-1}] weak_cross_source_recoverability: '{missing_attr}' has only generic opposite-source category evidence.")
                            
                    if all_opposing_not_direct:
                        raw_issues.append(Pass1ValidationIssue(f"{field}.missing_key_attributes[{i-1}]", "unrecoverable_missing_attribute", f"'{missing_attr}' marked recoverable, but all opposite sources have 'not_direct' capability for {req_cap}.", scope="missing_attribute"))
                    elif any_opposing_conditional and conditional_support_status == "reject":
                        local_warnings.append(f"{field}.missing_key_attributes[{i-1}] weak_cross_source_recoverability: '{missing_attr}' recovered via conditional capability {req_cap}, but supporting atoms lack evidence.")
                    elif any_opposing_conditional and conditional_support_status == "warn":
                        local_warnings.append(f"{field}.missing_key_attributes[{i-1}] weak_cross_source_recoverability: '{missing_attr}' recovered via conditional capability {req_cap}. Check if referenced atoms actually support this.")
                else:
                    if attr_type == "motion_state" or "motion" in missing_attr:
                        opp_facts = " ".join(str(atom_facts.get(a, "")) for opp_src in opposing_sources for a, e_refs in atom_entity_refs.items() if atom_sources.get(a) == opp_src and entity_id in e_refs).casefold()
                        if not any(term in opp_facts for term in ["mov", "stop", "park", "driv", "speed", "position", "motion", "stationary", "fast", "slow"]):
                            local_warnings.append(f"{field}.missing_key_attributes[{i-1}] weak_cross_source_recoverability: '{missing_attr}' lacks clear supporting evidence.")

        # Validator D: caption-to-atom grounding and inferential word check
        # Phase 1: Hard-fail on inferential linkage words
        INFERENTIAL_LINKAGE_WORDS = {
            "corresponding", "indicating", "suggesting", "implying",
            "therefore", "consequently", "representing", "associated",
            "appear", "seem", "apparently"
        }
        caption_lower = detailed_caption.lower()
        inferential_hits = set(re.findall(r'\b[a-z]+\b', caption_lower)) & INFERENTIAL_LINKAGE_WORDS
        if inferential_hits:
            raw_issues.append(Pass1ValidationIssue(f"{field}.detailed_caption", "inferential_wording", f"contains inferential linkage words: {', '.join(sorted(inferential_hits))}", scope=field))

        # Phase 2: Soft warning on ungrounded words with tighter threshold
        
        caption_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', caption_lower)) - GROUNDING_EXEMPT_WORDS
        atom_text = " ".join([str(a.get("fact", "")) for a in atoms if isinstance(a, dict)]).lower()
        atom_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', atom_text))
        ungrounded = caption_words - atom_words
        if len(ungrounded) > 5:
             local_warnings.append(f"{field}.detailed_caption may contain ungrounded claims. Words not in atoms: {', '.join(sorted(list(ungrounded))[:5])}...")

    _validate_video_analysis(parsed, "video1_analysis", "v1_atom_")
    _validate_video_analysis(parsed, "video2_analysis", "v2_atom_")
    
    def _check_physical(text: str, path: str, scope: str):
        if policy_mechanism == "hard_fail":
            for pattern in FORBIDDEN_MECHANISM_PATTERNS:
                match = pattern.search(text)
                if match:
                    raw_issues.append(Pass1ValidationIssue(path, "physical_world_wording", f"Forbidden mechanism-oriented wording {match.group(0)!r}. {FORBIDDEN_MECHANISM_MESSAGE}", scope=scope))
                    
            subj_match = REPRESENTATION_SUBJECT_PATTERN.search(text)
            pred_match = REPRESENTATION_PREDICATE_PATTERN.search(text)
            if subj_match and pred_match:
                raw_issues.append(Pass1ValidationIssue(path, "physical_world_wording", f"Forbidden representation-oriented construction ({subj_match.group(0)} + {pred_match.group(0)}).", scope=scope))
            elif pred_match:
                local_warnings.append(f"{path} physical_world_wording: May contain representation-oriented predicate '{pred_match.group(0)}'. Verify physical subject.")

    def _check_inferential(text: str, path: str, scope: str):
        if policy_inferential == "hard_fail":
            match = FORBIDDEN_INFERENTIAL_PATTERN.search(text)
            if match:
                raw_issues.append(Pass1ValidationIssue(path, "forbidden_inferential_terms", f"Forbidden inferential wording {match.group(0)!r}. {FORBIDDEN_INFERENTIAL_MESSAGE}", scope=scope))

    def _check_generic_sensor(text: str, path: str, scope: str):
        if policy_mechanism == "hard_fail":
            for pattern in GENERIC_SENSOR_EXPLANATION_PATTERNS:
                match = pattern.search(text)
                if match:
                    raw_issues.append(Pass1ValidationIssue(path, "generic_sensor_theory", f"Generic sensor-theory wording in {path}", scope=scope))

    def _run_semantic_checks(text: str, path: str, scope: str, entity_id: str | None = None):
        if not text: return
        _check_physical(text, path, scope)
        _check_inferential(text, path, scope)
        _check_generic_sensor(text, path, scope)
        if scope == "registry":
            _check_shared_leakage(text, path, scope, entity_id)

    def _check_shared_leakage(text: str, path: str, scope: str, entity_id: str | None = None):
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())) - GROUNDING_EXEMPT_WORDS
        
        for word in words:
            req_cap = _infer_required_capability("semantic_identity", word)
            if not req_cap:
                continue
                
            # 1. Check registry_source_attribute_leakage
            for source_key, mod in expected_source_modalities.items():
                cap_state = MODALITY_CAPABILITIES.get(mod, {}).get(req_cap, "conditional")
                if cap_state == "direct":
                    continue
                    
                if entity_id:
                    parts = []
                    for a, e_refs in atom_entity_refs.items():
                        if atom_sources.get(a) == source_key and entity_id in e_refs:
                            parts.append(str(atom_facts.get(a, "")))
                    source_facts = " ".join(parts).casefold()
                else:
                    parts = []
                    for a in atom_facts:
                        if atom_sources.get(a) == source_key:
                            parts.append(str(atom_facts.get(a, "")))
                    source_facts = " ".join(parts).casefold()
                support_status = _conditional_recovery_support_status(req_cap, word, source_facts)

                if cap_state == "not_direct" and support_status != "accept":
                    raw_issues.append(Pass1ValidationIssue(path, "registry_source_attribute_leakage", f"Shared attribute '{word}' requires '{req_cap}', which is {cap_state} in {mod} and not explicitly supported in {source_key} atoms.", scope=scope))
                elif cap_state == "conditional" and support_status == "reject":
                    raw_issues.append(Pass1ValidationIssue(path, "registry_source_attribute_leakage", f"Shared attribute '{word}' requires '{req_cap}', which is rejected by {source_key} atoms.", scope=scope))

            # 2. Check premature_entity_resolution
            if entity_id:
                for source_key in expected_source_modalities:
                    uncertainties = parsed.get(source_key, {}).get("uncertain_observations", [])
                    for unc in uncertainties:
                        if not isinstance(unc, dict): continue
                        if unc.get("entity_id") == entity_id:
                            unc_is_about_cap = False
                            for hyp in unc.get("hypotheses", []):
                                if not isinstance(hyp, dict): continue
                                hyp_text = str(hyp.get("hypothesis", "")).lower()
                                for h_word in re.findall(r'\b[a-zA-Z]{3,}\b', hyp_text):
                                    if _infer_required_capability("semantic_identity", h_word) == req_cap:
                                        if word not in str(unc.get("observed_evidence", "")).casefold():
                                            unc_is_about_cap = True
                                            break
                                if unc_is_about_cap: break
                            
                            if unc_is_about_cap:
                                raw_issues.append(Pass1ValidationIssue(path, "premature_entity_resolution", f"Shared attribute '{word}' makes a definitive claim about {req_cap}, but {source_key} has an unresolved uncertainty for entity {entity_id}.", scope=scope))



    semantic_categories = {"inferential_wording", "sensor_theory_wording", "generic_theory", "generic_sensor_theory", "registry_source_attribute_leakage", "premature_entity_resolution", "physical_world_wording", "forbidden_inferential_terms", "sensor_limitations", "unrecoverable_missing_attribute", "missing_attribute_contradiction"}
    structural_issues = []
    early_semantic_issues = []
    seen_struct = set()
    for issue in raw_issues:
        key = (issue.path, issue.category, issue.message)
        if key not in seen_struct:
            seen_struct.add(key)
            if issue.category in semantic_categories:
                early_semantic_issues.append(issue)
            else:
                structural_issues.append(issue)

    if structural_issues:
        raise Pass1StructuralValidationError("Structural Validation Errors", structural_issues)
        
    raw_issues.clear()
    raw_issues.extend(early_semantic_issues)

    # --- Semantic Check Loop ---
    # Global Scene
    _run_semantic_checks(global_scene.get("scene_summary", ""), "global_scene.scene_summary", "global_scene")
    _run_semantic_checks(global_scene.get("environment", ""), "global_scene.environment", "global_scene")
    _run_semantic_checks(global_scene.get("temporal_progression", ""), "global_scene.temporal_progression", "global_scene")
    
    for i, ent in enumerate(global_scene.get("physical_entities", [])):
        ent_id = ent.get("entity_id") if isinstance(ent.get("entity_id"), str) else None
        _run_semantic_checks(ent.get("category", ""), f"global_scene.physical_entities[{i}].category", "global_scene")
        _run_semantic_checks(ent.get("referential_scope", ""), f"global_scene.physical_entities[{i}].referential_scope", "registry", ent_id)
        prof = ent.get("evidence_profile", {})
        for j, ev in enumerate(prof.get("identity_evidence", [])):
            _run_semantic_checks(ev, f"global_scene.physical_entities[{i}].evidence_profile.identity_evidence[{j}]", "global_scene")
        for j, ev in enumerate(prof.get("observable_attributes", [])):
            _run_semantic_checks(ev, f"global_scene.physical_entities[{i}].evidence_profile.observable_attributes[{j}]", "global_scene")
        for j, ev in enumerate(prof.get("spatial_context", [])):
            _run_semantic_checks(ev, f"global_scene.physical_entities[{i}].evidence_profile.spatial_context[{j}]", "global_scene")
    
    # Shared Registry
    for i, ent in enumerate(global_scene.get("shared_registry", [])):
        ent_id = ent.get("entity_id")
        _run_semantic_checks(ent.get("semantic_identity", ""), f"global_scene.shared_registry[{i}].semantic_identity", "registry", ent_id)
        for j, attr in enumerate(ent.get("attributes", [])):
            _run_semantic_checks(attr.get("attribute", ""), f"global_scene.shared_registry[{i}].attributes[{j}].attribute", "registry", ent_id)

    # Video Analysis
    for source_key in expected_source_modalities:
        analysis = parsed.get(source_key, {})
        _run_semantic_checks(analysis.get("detailed_caption", ""), f"{source_key}.detailed_caption", "video_analysis")
        _validate_modality_specific_physical_claims(
            expected_source_modalities[source_key],
            analysis.get("detailed_caption", ""),
            f"{source_key}.detailed_caption",
            raw_issues,
            local_warnings,
        )
        
        for i, atom in enumerate(analysis.get("information_atoms", [])):
            _run_semantic_checks(atom.get("fact", ""), f"{source_key}.information_atoms[{i}].fact", "atom")
            _validate_modality_specific_physical_claims(
                expected_source_modalities[source_key],
                atom.get("fact", ""),
                f"{source_key}.information_atoms[{i}].fact",
                raw_issues,
                local_warnings,
            )
            
        for i, unc in enumerate(analysis.get("uncertain_observations", [])):
            _run_semantic_checks(unc.get("observed_evidence", ""), f"{source_key}.uncertain_observations[{i}].observed_evidence", "uncertainty")
            _run_semantic_checks(unc.get("missing_evidence", ""), f"{source_key}.uncertain_observations[{i}].missing_evidence", "uncertainty")
            for j, hyp in enumerate(unc.get("hypotheses", [])):
                _run_semantic_checks(hyp.get("hypothesis", ""), f"{source_key}.uncertain_observations[{i}].hypotheses[{j}].hypothesis", "uncertainty")
                _run_semantic_checks(hyp.get("why_missing", ""), f"{source_key}.uncertain_observations[{i}].hypotheses[{j}].why_missing", "uncertainty")

        for i, cue in enumerate(analysis.get("sensor_specific_cues", [])):
            _check_generic_sensor(cue, f"{source_key}.sensor_specific_cues[{i}]", "sensor_specific_cue")
        
        for i, sl in enumerate(analysis.get("sensor_limitations", [])):
            _run_semantic_checks(sl, f"{source_key}.sensor_limitations[{i}]", "sensor_limitation")
            
        for i, attr in enumerate(analysis.get("missing_key_attributes", [])):
            if not isinstance(attr, dict): continue
            ent_id = attr.get("entity_id")
            ent_id = ent_id if isinstance(ent_id, str) else None
            _run_semantic_checks(str(attr.get("missing_attribute", "")), f"{source_key}.missing_key_attributes[{i}].missing_attribute", "missing_attribute", ent_id)
            _run_semantic_checks(str(attr.get("why_missing", "")), f"{source_key}.missing_key_attributes[{i}].why_missing", "missing_attribute", ent_id)

    # Phase C: warning-only recall for a clear color target in the active pair.
    # Capability lookup, not a hard-coded modality name, selects the supporting source.
    active_sources = tuple(expected_source_modalities)
    if len(active_sources) == 2:
        for supporting_source in active_sources:
            supporting_modality = expected_source_modalities[supporting_source]
            if MODALITY_CAPABILITIES.get(supporting_modality, {}).get("color") != "direct":
                continue
            opposite_source = next(source for source in active_sources if source != supporting_source)
            for entity_id in entity_ids:
                supporting_facts = " ".join(
                    atom_facts[atom_id]
                    for atom_id, refs in atom_entity_refs.items()
                    if atom_sources.get(atom_id) == supporting_source and entity_id in refs
                )
                opposite_facts = " ".join(
                    atom_facts[atom_id]
                    for atom_id, refs in atom_entity_refs.items()
                    if atom_sources.get(atom_id) == opposite_source and entity_id in refs
                )
                if not supporting_facts or not opposite_facts:
                    continue
                if _conditional_recovery_support_status("color", "surface paint color", supporting_facts) != "accept":
                    continue
                if _conditional_recovery_support_status("color", "surface paint color", opposite_facts) == "accept":
                    continue
                existing_target = any(
                    isinstance(attr, dict)
                    and attr.get("entity_id") == entity_id
                    and (
                        attr.get("attribute_type") == "surface_attribute"
                        or _infer_required_capability(str(attr.get("attribute_type", "")), str(attr.get("missing_attribute", ""))) == "color"
                    )
                    for attr in parsed[opposite_source].get("missing_key_attributes", [])
                )
                if not existing_target:
                    local_warnings.append(
                        f"{opposite_source}.missing_key_attributes possible_missing_recoverability_target: "
                        f"entity {entity_id} has direct color evidence in {supporting_source} but only non-color evidence in the opposite active source."
                    )



    semantic_issues = []
    seen_sem = set()
    for issue in raw_issues:
        key = (issue.path, issue.category, issue.message)
        if key not in seen_sem:
            seen_sem.add(key)
            semantic_issues.append(issue)
            
    if semantic_issues:
        raise Pass1SemanticValidationError("Semantic Validation Errors", semantic_issues)

    return parsed, local_warnings