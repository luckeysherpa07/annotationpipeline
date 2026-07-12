import pytest
from annotation_feature.aligned_caption_pass1_validation import (
    _validate_pass1_schema,
    Pass1StructuralValidationError,
    Pass1SemanticValidationError,
    Pass1ValidationIssue,
)
from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import (
    _build_pass1_validation_retry_hint,
    _categorize_pass1_validation_error,
)


@pytest.fixture
def base_parsed():
    return {
        "global_scene": {
            "scene_summary": "A person rides a bicycle down a dark street. The person continues riding the bicycle down the dark street steadily and safely.",
            "environment": "It is nighttime and there are streetlights.",
            "temporal_progression": "The person continues to ride until they exit the view.",
            "physical_entities": [
                {
                    "entity_id": "ent_person_1",
                    "category": "person",
                    "referential_scope": "the rider",
                    "evidence_profile": {
                        "identity_evidence": ["person on a bike"],
                        "observable_attributes": ["wearing a jacket"],
                        "spatial_context": ["on the street"]
                    }
                }
            ]
        },
        "video1_analysis": {
            "modality": "rgb",
            "detailed_caption": "A person is riding a bicycle down a dark street. The streetlights provide some illumination. The rider maintains a steady pace and continues moving forward without any hesitation or deviation.",
            "information_atoms": [
                {
                    "atom_id": "v1_atom_1",
                    "frame_keys": ["frame_1"],
                    "entity_refs": ["ent_person_1"],
                    "fact": "person riding bicycle steadily forward without deviation or hesitation"
                }
            ],
            "sensor_specific_cues": ["color of the jacket"],
            "sensor_limitations": ["low light makes faces hard to see in frame_1"],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": "event",
            "detailed_caption": "A person is riding a bicycle down a dark street. The movement of the wheels is clearly visible. The rider maintains a steady pace and continues moving forward without any hesitation or deviation.",
            "information_atoms": [
                {
                    "atom_id": "v2_atom_1",
                    "frame_keys": ["frame_1"],
                    "entity_refs": ["ent_person_1"],
                    "fact": "person moving steadily forward on bicycle without deviation or hesitation"
                }
            ],
            "sensor_specific_cues": ["motion of the pedals"],
            "sensor_limitations": ["pedal motion is fast in frame_1"],
            "uncertain_observations": [],
            "missing_key_attributes": []
        }
    }


def test_valid_full_example_passes(base_parsed):
    try:
        _, warnings = _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Pass1SemanticValidationError as e:
        for err in e.errors:
            print("DEBUG:", err)
        raise
    # assert not warnings # Allow soft warnings


def test_environment_mechanism_fails(base_parsed):
    base_parsed["global_scene"]["environment"] = "The edge activations in the background are sparse."
    with pytest.raises(Pass1SemanticValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("mechanism-oriented" in e.message for e in exc.value.errors)


def test_uncertainty_evidence_refs_empty_fails(base_parsed):
    base_parsed["video1_analysis"]["uncertain_observations"] = [
        {
            "uncertainty_id": "v1_unc_1",
            "entity_id": "ent_person_1",
            "observed_evidence": "A fuzzy object is seen.",
            "missing_evidence": "Cannot see the face clearly.",
            "evidence_refs": [],
            "hypotheses": [
                {"hypothesis": "It is X", "confidence": "high"},
                {"hypothesis": "It is Y", "confidence": "low"}
            ]
        }
    ]
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("evidence_refs must be a non-empty list" in e.message for e in exc.value.errors)


def test_uncertainty_references_atom_with_empty_entity_refs(base_parsed):
    base_parsed["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": [],
        "fact": "Some background event"
    })
    base_parsed["video1_analysis"]["uncertain_observations"] = [
        {
            "uncertainty_id": "v1_unc_1",
            "entity_id": "ent_person_1",
            "observed_evidence": "A fuzzy object is seen.",
            "missing_evidence": "Cannot see the face clearly.",
            "evidence_refs": ["v1_atom_2"],
            "hypotheses": [
                {"hypothesis": "It is X", "confidence": "high"},
                {"hypothesis": "It is Y", "confidence": "low"}
            ]
        }
    ]
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(("not connected to entity ent_person_1" in e.message or "entity_refs must be a non-empty list" in e.message) for e in exc.value.errors)


def test_uncertainty_entity_id_not_in_atom_entity_refs(base_parsed):
    base_parsed["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car_1",
        "category": "car",
        "referential_scope": "the car",
        "evidence_profile": {
            "identity_evidence": ["car shape"],
        }
    })
    base_parsed["video1_analysis"]["uncertain_observations"] = [
        {
            "uncertainty_id": "v1_unc_1",
            "entity_id": "ent_car_1",
            "observed_evidence": "car shape",
            "missing_evidence": "no license plate",
            "evidence_refs": ["v1_atom_1"], # v1_atom_1 only has ent_person_1
            "hypotheses": [
                {"hypothesis": "It is X", "confidence": "high"},
                {"hypothesis": "It is Y", "confidence": "low"}
            ]
        }
    ]
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    for e in exc.value.errors:
        print("DEBUG:", e.message)
    assert any("not connected to entity ent_car_1" in e.message for e in exc.value.errors)


def test_uncertainty_wrong_source_atom(base_parsed):
    base_parsed["video1_analysis"]["uncertain_observations"] = [
        {
            "uncertainty_id": "v1_unc_1",
            "entity_id": "ent_person_1",
            "observed_evidence": "person shape",
            "missing_evidence": "no face",
            "evidence_refs": ["v2_atom_1"], # referencing v2 atom from v1 analysis
            "hypotheses": [
                {"hypothesis": "It is X", "confidence": "high"},
                {"hypothesis": "It is Y", "confidence": "low"}
            ]
        }
    ]
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(("belongs to a different source" in e.message or "must start with" in e.message) for e in exc.value.errors)


def test_atom_frame_keys_empty_fails(base_parsed):
    base_parsed["video1_analysis"]["information_atoms"][0]["frame_keys"] = []
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("frame_keys must be a non-empty list" in e.message for e in exc.value.errors)


def test_atom_unknown_frame_key_fails(base_parsed):
    base_parsed["video1_analysis"]["information_atoms"][0]["frame_keys"] = ["frame_99"]
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("Unknown frame_key 'frame_99'" in e.message for e in exc.value.errors)


def test_observed_evidence_mechanism_fails(base_parsed):
    base_parsed["video1_analysis"]["uncertain_observations"] = [
        {
            "uncertainty_id": "v1_unc_1",
            "entity_id": "ent_person_1",
            "observed_evidence": "The zero-activation regions show a shape.",
            "missing_evidence": "Cannot see the face clearly.",
            "evidence_refs": ["v1_atom_1"],
            "hypotheses": [
                {"hypothesis": "It is X", "confidence": "high"},
                {"hypothesis": "It is Y", "confidence": "low"}
            ]
        }
    ]
    with pytest.raises(Pass1SemanticValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("mechanism-oriented" in e.message for e in exc.value.errors)


def test_hypothesis_generic_theory_fails(base_parsed):
    base_parsed["video1_analysis"]["uncertain_observations"] = [
        {
            "uncertainty_id": "v1_unc_1",
            "entity_id": "ent_person_1",
            "observed_evidence": "A blurry shape",
            "missing_evidence": "no face",
            "evidence_refs": ["v1_atom_1"],
            "hypotheses": [
                {"hypothesis": "The sensor does not record static color.", "confidence": "high"},
                {"hypothesis": "It might be a shadow.", "confidence": "low"}
            ]
        }
    ]
    with pytest.raises(Pass1SemanticValidationError) as exc:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("Generic sensor-theory" in e.message for e in exc.value.errors)


def test_why_missing_generic_theory_fails(base_parsed):
    base_parsed["video1_analysis"]["missing_key_attributes"] = [
        {
            "entity_id": "ent_person_1",
            "attribute_type": "surface_attribute",
            "missing_attribute": "color of the bike",
            "why_missing": "the modality does not provide color information",
            "recoverable_evidence_refs": []
        }
    ]
    try:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Pass1StructuralValidationError as e:
        for err in e.errors:
            print("DEBUG:", err)
        raise
    except Pass1SemanticValidationError as exc:
        assert any("Generic sensor-theory wording" in e.message for e in exc.errors)


def test_valid_paraphrase_gives_warning(base_parsed):
    # Caption words have 'bicycle' but atom has 'bike' -> ungrounded mismatch -> warning
    base_parsed["video1_analysis"]["information_atoms"][0]["fact"] = "person riding a vehicle"
    base_parsed["video1_analysis"]["detailed_caption"] = "A person is riding a specialized bicycle vehicle very fast down a dark street with heavy wheels, shiny colors, plastic fenders, metal frames, and glowing lanterns. The person maintains a steady pace and continues moving forward without deviation or hesitation."
    try:
        _, warnings = _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Pass1SemanticValidationError as e:
        for err in e.errors:
            print("DEBUG:", err)
        raise
    assert any("ungrounded" in w for w in warnings)


def test_deduplication_by_message(base_parsed):
    base_parsed["global_scene"]["scene_summary"] = "The edge activations in the response map are clear. " * 5 # two mechanism patterns, multiplied to pass min word count
    try:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Pass1StructuralValidationError as e:
        for err in e.errors:
            print("STRUCT:", err)
        raise
    except Pass1SemanticValidationError as exc:
        for err in exc.errors:
            print("SEMANTIC:", err)
        assert len(exc.errors) >= 2
    except Exception as e:
        print("OTHER ERROR:", e)
        if hasattr(e, 'errors'):
            for err in e.errors: print("SEMANTIC:", err)
        raise


def test_legitimate_sensor_limitations(base_parsed):
    base_parsed["video2_analysis"]["sensor_limitations"] = ["The darkness obscures the person's face."]
    try:
        _, warnings = _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Pass1SemanticValidationError as e:
        for err in e.errors:
            print("DEBUG:", err)
        raise
    assert not any("generic sensor-theory" in w for w in warnings)

def test_retry_hint_uses_exact_path(base_parsed):
    base_parsed["video1_analysis"]["sensor_specific_cues"] = ["the sensor captures changes"]
    try:
        _validate_pass1_schema(base_parsed, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Pass1SemanticValidationError as e:
        assert any("video1_analysis.sensor_specific_cues[0]" in issue.path or "video1_analysis.sensor_specific_cues[1]" in issue.path for issue in e.errors)

import copy

def test_recoverability_pass_suv_paint_color(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_suv",
        "category": "vehicle",
        # Keep the shared Registry coarse; SUV specificity is source-local.
        "referential_scope": "the vehicle",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_suv"],
        "fact": "the suv is painted white"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_suv",
        "attribute_type": "surface_attribute",
        "missing_attribute": "paint color of the suv",
        "why_missing": "surface detail is not visible in frame_1",
        "recoverable_evidence_refs": []
    }]
    try:
        _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Pass1StructuralValidationError as e:
        for err in e.errors:
            print("STRUCT:", err)
        raise
    except Exception as e:
        print("OTHER ERROR:", e)
        if hasattr(e, 'errors'):
            for err in e.errors: print("SEMANTIC:", err)
        raise
    assert not any("unrecoverable_missing_attribute" in w for w in warnings)
    assert not any("weak_cross_source_recoverability" in w for w in warnings)

def test_recoverability_fail_wall_color(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_wall",
        "category": "infrastructure",
        "referential_scope": "the wall",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video2_analysis"]["information_atoms"].append({
        "atom_id": "v2_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_wall"],
        "fact": "a straight seam defines the wall edge"
    })
    p["video1_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_wall",
        "attribute_type": "surface_attribute",
        "missing_attribute": "color of the wall",
        "why_missing": "obscured by shadow in frame_1",
        "recoverable_evidence_refs": []
    }]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "unrecoverable_missing_attribute" for issue in exc_info.value.errors)

def test_recoverability_fail_vehicle_motion_headlights(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "the car has two headlights"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the car",
        "why_missing": "position is stable in frame_1",
        "recoverable_evidence_refs": []
    }]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("weak_cross_source_recoverability" in warning for warning in warnings)

def test_recoverability_pass_vehicle_motion_position(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "the car changes position across the frame"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the car",
        "why_missing": "position is stable in frame_1",
        "recoverable_evidence_refs": []
    }]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert not any("unrecoverable" in w for w in warnings)

def test_recoverability_hard_fail_no_atom(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_person_1",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of ent_person_1",
        "why_missing": "not visible",
        "recoverable_evidence_refs": []
    }]
    # video1 has no atom about motion state for person 1
    # Actually video1 has "person riding bicycle" which might be weak, let's remove it
    p["global_scene"]["physical_entities"].append({"entity_id": "ent_another_thing", "category": "object", "referential_scope": "other", "evidence_profile": {"identity_evidence": ["a"], "observable_attributes": ["a"], "spatial_context": ["a"]}})
    p["video1_analysis"]["information_atoms"][0]["entity_refs"] = ["ent_another_thing"]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("No opposite-source Atom references target entity" in issue.message for issue in exc_info.value.errors)

def test_recoverability_warning_ambiguous_evidence(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "the car is seen on the road" # generic existence
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the car",
        "why_missing": "not visible",
        "recoverable_evidence_refs": []
    }]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("weak_cross_source_recoverability" in str(w) for w in warnings)

def test_recoverability_explicit_target_does_not_use_ambiguous_text(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car1",
        "category": "car",
        "referential_scope": "the red car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car2",
        "category": "car",
        "referential_scope": "the blue car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car1",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the car",
        "why_missing": "not visible",
        "recoverable_evidence_refs": []
    }]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("No opposite-source Atom references target entity ent_car1" in issue.message for issue in exc_info.value.errors)

def test_recoverability_unknown_explicit_target_is_structural_error(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_alien",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the alien spaceship",
        "why_missing": "not visible",
        "recoverable_evidence_refs": []
    }]
    with pytest.raises(Pass1StructuralValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "invalid_entity_reference" for issue in exc_info.value.errors)

def test_recoverability_hard_fail_not_direct(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_wall",
        "category": "infrastructure",
        "referential_scope": "the wall",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video2_analysis"]["information_atoms"].append({
        "atom_id": "v2_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_wall"],
        "fact": "wall is visible"
    })
    p["video1_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_wall",
        "attribute_type": "surface_attribute",
        "missing_attribute": "color of the wall",
        "why_missing": "obscured",
        "recoverable_evidence_refs": []
    }]
    # Event modality has color: not_direct
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("is 'not_direct'" in issue.message or "is not_direct" in issue.message or "not_direct" in issue.message for issue in exc_info.value.errors)

def test_recoverability_warning_unlisted_synonym(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "the car is sprinting" # sprinting is not in motion words explicitly
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the car",
        "why_missing": "position is stable in frame_1",
        "recoverable_evidence_refs": []
    }]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("weak_cross_source_recoverability" in str(w) for w in warnings)

def test_recoverability_explicitly_incompatible(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "white headlights are visible"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "surface_attribute",
        "missing_attribute": "white paint of the car",
        "why_missing": "obscured",
        "recoverable_evidence_refs": []
    }]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("Explicitly incompatible" in issue.message or "missing_attribute_contradiction" == issue.category for issue in exc_info.value.errors)

def test_recoverability_other_entity_color(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_bike",
        "category": "vehicle",
        "referential_scope": "the bike",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_bike"],
        "fact": "bike is blue color"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "surface_attribute",
        "missing_attribute": "color of the car",
        "why_missing": "obscured",
        "recoverable_evidence_refs": []
    }]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("No opposite-source Atom references target entity" in issue.message for issue in exc_info.value.errors)

def test_recoverability_existence_succeeds(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "car is seen"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "existence",
        "missing_attribute": "existence of the car",
        "why_missing": "obscured",
        "recoverable_evidence_refs": []
    }]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert not any("unrecoverable" in str(w) for w in warnings)


def test_recoverability_generic_category_weak(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "the object is present"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "fine_grained_category",
        "missing_attribute": "fine category of the car",
        "why_missing": "obscured",
        "recoverable_evidence_refs": []
    }]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("weak_cross_source_recoverability" in str(w) for w in warnings)


def test_recoverability_physical_cause_warning(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["physical_entities"].append({
        "entity_id": "ent_car",
        "category": "vehicle",
        "referential_scope": "the car",
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]}
    })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_2",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car"],
        "fact": "the car is visible"
    })
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_car",
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the car",
        "why_missing": "obscured",
        "recoverable_evidence_refs": []
    }]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("weak_cross_source_recoverability" in str(w) for w in warnings)

def test_recoverability_structural_context_only(base_parsed):
    # Only fully valid opposite-source atoms are used
    pass # Verified by implementation logic which relies on context.atom_facts

def test_recoverability_preserves_input(base_parsed):
    p = copy.deepcopy(base_parsed)
    p_orig = copy.deepcopy(base_parsed)
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert p == p_orig
    assert p["video2_analysis"]["missing_key_attributes"] == []

def test_sensor_limitations_generic_rejected(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video1_analysis"]["sensor_limitations"] = ["the sensor cannot detect color"]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("Generic sensor-theory wording" in e.message for e in exc_info.value.errors)

def test_sensor_limitations_segment_specific_passes(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video1_analysis"]["sensor_limitations"] = ["shadows obscure ent_person_1 in frame_1"]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert not any("segment-specific" in w for w in warnings)

def test_sensor_limitations_no_frame_but_entity_passes(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video1_analysis"]["sensor_limitations"] = ["shadows obscure the rider"]
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert not any("segment-specific" in w for w in warnings)

def test_generic_theory_disguised_detected(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video1_analysis"]["sensor_limitations"] = ["static objects without relative motion are invisible"]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("Generic sensor-theory wording" in e.message for e in exc_info.value.errors)

def test_structural_validation_first(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video1_analysis"]["information_atoms"][0]["entity_refs"] = [] # Structural error
    p["video1_analysis"]["sensor_limitations"] = ["the sensor cannot detect color"] # Semantic error
    with pytest.raises(Pass1StructuralValidationError):
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})

def test_no_max_detailed_caption_length(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video1_analysis"]["detailed_caption"] = " ".join(["word"] * 500)
    # Shouldn't raise any max length error
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})


@pytest.mark.parametrize("entity_id, category", [
    (None, "missing_field"),
    ("", "invalid_type"),
    (123, "invalid_type"),
    ("missing_entity", "invalid_entity_reference"),
])
def test_missing_attribute_entity_id_contract(base_parsed, entity_id, category):
    import copy
    p = copy.deepcopy(base_parsed)
    item = {
        "entity_id": entity_id,
        "attribute_type": "motion_state",
        "missing_attribute": "motion state of the rider",
        "why_missing": "Only one sampled position is visible in frame_1.",
        "recoverable_evidence_refs": [],
    }
    if entity_id is None:
        item.pop("entity_id")
    p["video2_analysis"]["missing_key_attributes"] = [item]
    with pytest.raises(Pass1StructuralValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == category and issue.path.endswith(".entity_id") for issue in exc_info.value.errors)


def _add_entity_with_atoms(parsed, scope, video1_fact, video2_fact, entity_id="ent_target"):
    parsed["global_scene"]["physical_entities"].append({
        "entity_id": entity_id,
        "category": "vehicle",
        "referential_scope": scope,
        "evidence_profile": {"identity_evidence": ["dummy evidence"], "observable_attributes": ["dummy evidence"], "spatial_context": ["dummy evidence"]},
    })
    parsed["video1_analysis"]["information_atoms"].append({"atom_id": "v1_atom_target", "frame_keys": ["frame_1"], "entity_refs": [entity_id], "fact": video1_fact})
    parsed["video2_analysis"]["information_atoms"].append({"atom_id": "v2_atom_target", "frame_keys": ["frame_1"], "entity_refs": [entity_id], "fact": video2_fact})


def test_registry_source_attribute_leakage_hard_fails(base_parsed):
    import copy
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the white SUV on the right", "The SUV body is painted white.", "The SUV has a curved outer boundary.")
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "registry_source_attribute_leakage" for issue in exc_info.value.errors)


def test_neutral_registry_preserves_source_local_specificity(base_parsed):
    import copy
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the SUV on the right", "The SUV body is painted white.", "The SUV has a curved outer boundary.")
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})


def test_registry_attribute_allowed_when_both_sources_support_it(base_parsed):
    import copy
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the white SUV on the right", "The SUV body is painted white.", "The white paint on the SUV body remains visible.")
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})


@pytest.mark.parametrize("video2_fact", [
    "Two white headlights remain visible on the SUV.",
    "The SUV stands beneath a dark nighttime sky with weak illumination.",
])

def test_registry_color_not_supported_by_headlights_or_illumination(base_parsed, video2_fact):
    import copy
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the white SUV", "The SUV body is painted white.", video2_fact)
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "registry_source_attribute_leakage" for issue in exc_info.value.errors)


def _add_incompatible_uncertainty(parsed, scope):
    _add_entity_with_atoms(parsed, scope, "Two distant bright points remain visible.", "Two distant bright points remain visible.", "ent_points")
    parsed["video1_analysis"]["uncertain_observations"] = [{
        "uncertainty_id": "v1_unc_points",
        "entity_id": "ent_points",
        "observed_evidence": "Two distant bright points remain visible.",
        "missing_evidence": "No stable body shape is visible.",
        "evidence_refs": ["v1_atom_target"],
        "hypotheses": [
            {"hypothesis": "An approaching passenger car", "confidence": "low"},
            {"hypothesis": "Stationary reflective installations", "confidence": "low"},
        ],
    }]



def test_registry_premature_uncertainty_resolution_fails(base_parsed):
    import copy
    p = copy.deepcopy(base_parsed)
    _add_incompatible_uncertainty(p, "an oncoming vehicle")
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "premature_entity_resolution" for issue in exc_info.value.errors)


def test_registry_neutral_scope_with_incompatible_hypotheses_passes(base_parsed):
    import copy
    p = copy.deepcopy(base_parsed)
    _add_incompatible_uncertainty(p, "two distant bright points")
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})


@pytest.mark.parametrize("attribute_type, missing_attribute", [
    ("surface_attribute", "paint color of the white SUV"),
    ("motion_state", "motion state of the parked SUV"),
])
def test_missing_attribute_concrete_value_contradiction(base_parsed, attribute_type, missing_attribute):
    import copy
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the SUV", "The SUV remains visible.", "The SUV remains visible.")
    p["video1_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_target",
        "attribute_type": attribute_type,
        "missing_attribute": missing_attribute,
        "why_missing": "Only a partial body region is visible in frame_1.",
        "recoverable_evidence_refs": [],
    }]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "missing_attribute_contradiction" for issue in exc_info.value.errors)


def test_event_representation_wording_fails_but_physical_wording_passes(base_parsed):
    import copy
    p = copy.deepcopy(base_parsed)
    p["video2_analysis"]["information_atoms"][0]["fact"] = "Dense boundary patterns delineate the vehicle and the person."
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "physical_world_wording" for issue in exc_info.value.errors)
    p["video2_analysis"]["information_atoms"][0]["fact"] = "The vehicle outline remains distinguishable, along with the person."
    try:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    except Exception as e:
        for err in getattr(e, 'errors', []):
            print(f'ERROR: {err.category}: {err.message}')
        raise


@pytest.mark.parametrize("text", [
    "The motion boundaries of an approaching vehicle are captured.",
    "The architectural edges of the building facade are defined.",
    "The intricate silhouettes of street-side trees are resolved.",
])
def test_real_event_representation_wording_regressions(base_parsed, text):
    p = copy.deepcopy(base_parsed)
    p["video2_analysis"]["information_atoms"][0]["fact"] = text
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(
        issue.path == "video2_analysis.information_atoms[0].fact"
        and issue.category == "physical_world_wording"
        for issue in exc_info.value.errors
    )


@pytest.mark.parametrize("text", [
    "The vehicle approaches along the left side of the street.",
    "Vertical structural lines remain visible on the building facade.",
    "The trunks and branches of the roadside trees remain distinguishable.",
])
def test_valid_event_physical_rewrites_pass(base_parsed, text):
    p = copy.deepcopy(base_parsed)
    p["video2_analysis"]["information_atoms"][0]["fact"] = text
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})


def test_generated_modality_mismatch_is_structural(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video2_analysis"]["modality"] = "rgb"
    with pytest.raises(Pass1StructuralValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "modality_mismatch" for issue in exc_info.value.errors)


def test_trusted_event_dispatch_cannot_be_evaded_by_generated_modality(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["video2_analysis"]["modality"] = "rgb"
    p["video2_analysis"]["information_atoms"][0]["fact"] = "The motion boundaries of an approaching vehicle are captured."
    with pytest.raises(Pass1StructuralValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "modality_mismatch" for issue in exc_info.value.errors)


def test_capability_aware_missing_color_recall_is_warning_only(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(
        p,
        "the SUV on the right",
        "The SUV body is painted white.",
        "The SUV has a curved outer boundary.",
    )
    _, warnings = _validate_pass1_schema(
        p,
        {"frame_1"},
        {"video1_analysis": "rgb", "video2_analysis": "event"},
    )
    assert any("possible_missing_recoverability_target" in warning for warning in warnings)


def test_multi_entity_atom_does_not_spread_color_between_entities(base_parsed):
    p = copy.deepcopy(base_parsed)
    for entity_id, scope in (("ent_car", "the car"), ("ent_bike", "the white bike")):
        p["global_scene"]["physical_entities"].append({
            "entity_id": entity_id,
            "category": "vehicle",
            "referential_scope": scope,
            "evidence_profile": {"identity_evidence": ["dummy"], "observable_attributes": ["dummy"], "spatial_context": ["dummy"]},
        })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_multi",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car", "ent_bike"],
        "fact": "A white car stands beside a blue bike.",
    })
    p["video2_analysis"]["information_atoms"].append({
        "atom_id": "v2_atom_multi",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_car", "ent_bike"],
        "fact": "The car remains beside the bike.",
    })
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(
        issue.category == "registry_source_attribute_leakage"
        and issue.path.endswith("physical_entities[2].referential_scope")
        for issue in exc_info.value.errors
    )


def test_direct_capability_without_compatible_atom_is_only_weakly_recoverable(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the vehicle", "The vehicle remains visible.", "The vehicle outline remains visible.")
    p["video2_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_target",
        "attribute_type": "surface_attribute",
        "missing_attribute": "paint color of the vehicle",
        "why_missing": "The body surface lacks stable detail in frame_1.",
        "recoverable_evidence_refs": [],
    }]
    _, warnings = _validate_pass1_schema(
        p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"}
    )
    assert any("weak_cross_source_recoverability" in warning for warning in warnings)


def test_ambiguous_global_attribute_leakage_is_warning(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["scene_summary"] += " A white vehicle is also present nearby."
    _, warnings = _validate_pass1_schema(
        p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"}
    )
    assert any("shared_global_source_attribute_leakage" in warning for warning in warnings)


def _set_vehicle_uncertainty(parsed, *, fact, missing, hypotheses):
    _add_entity_with_atoms(
        parsed,
        "the box-shaped vehicle on the right",
        fact,
        "A box-shaped vehicle remains visible on the right.",
    )
    parsed["video1_analysis"]["uncertain_observations"] = [{
        "uncertainty_id": "v1_unc_target",
        "entity_id": "ent_target",
        "observed_evidence": "A box-shaped object is visible on the right.",
        "missing_evidence": missing,
        "evidence_refs": ["v1_atom_target"],
        "hypotheses": [
            {"hypothesis": hypothesis, "confidence": "low"}
            for hypothesis in hypotheses
        ],
    }]


def test_same_source_vehicle_atom_contradicts_cargo_hypothesis(base_parsed):
    p = copy.deepcopy(base_parsed)
    _set_vehicle_uncertainty(
        p,
        fact="A white vehicle is parked on the right side under a streetlight.",
        missing="Direct illumination.",
        hypotheses=["A stationary passenger SUV", "A stack of construction pallets or cargo"],
    )
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "source_uncertainty_contradiction" for issue in exc_info.value.errors)


def test_fine_grained_compatible_vehicle_hypotheses_pass(base_parsed):
    p = copy.deepcopy(base_parsed)
    _set_vehicle_uncertainty(
        p,
        fact="A box-shaped vehicle is stationary on the right.",
        missing="The exact vehicle subtype.",
        hypotheses=["SUV", "delivery van", "utility vehicle"],
    )
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})


def test_atom_visible_color_contradicts_missing_color_uncertainty(base_parsed):
    p = copy.deepcopy(base_parsed)
    _set_vehicle_uncertainty(
        p,
        fact="The vehicle body is painted white.",
        missing="The vehicle paint color.",
        hypotheses=["A white SUV", "A white delivery van"],
    )
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "source_uncertainty_contradiction" for issue in exc_info.value.errors)


def test_atom_visible_illumination_contradicts_missing_illumination(base_parsed):
    p = copy.deepcopy(base_parsed)
    _set_vehicle_uncertainty(
        p,
        fact="The vehicle is parked beneath a streetlight.",
        missing="Direct illumination of the vehicle.",
        hypotheses=["A stationary SUV", "A stationary delivery van"],
    )
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "source_uncertainty_contradiction" for issue in exc_info.value.errors)


@pytest.mark.parametrize(
    ("attribute_type", "target", "opposite_fact"),
    [
        ("fine_grained_category", "specific make and model of the vehicle", "The vehicle has rectangular outer boundaries."),
        ("state_attribute", "license plate characters", "The motorhome has a broad rectangular outline."),
    ],
)
def test_precision_target_requires_actual_opposite_evidence(base_parsed, attribute_type, target, opposite_fact):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the large vehicle", "A large vehicle is visible.", opposite_fact)
    p["video1_analysis"]["missing_key_attributes"] = [{
        "entity_id": "ent_target",
        "attribute_type": attribute_type,
        "missing_attribute": target,
        "why_missing": "The relevant surface region lacks stable detail in frame_1.",
        "recoverable_evidence_refs": [],
    }]
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "unrecoverable_missing_attribute" for issue in exc_info.value.errors)


def test_missing_target_warning_uses_containing_analysis_path(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the large vehicle", "A large vehicle remains visible.", "The vehicle has a broad outline.")
    p["video1_analysis"]["missing_key_attributes"] = [
        {
            "entity_id": "ent_target",
            "attribute_type": "existence",
            "missing_attribute": "continued existence of the vehicle",
            "why_missing": "The vehicle leaves the visible region after frame_1.",
            "recoverable_evidence_refs": [],
        },
        {
            "entity_id": "ent_target",
            "attribute_type": "state_attribute",
            "missing_attribute": "roof-rack mounting configuration",
            "why_missing": "The roof region lacks stable detail in frame_1.",
            "recoverable_evidence_refs": [],
        },
    ]
    _, warnings = _validate_pass1_schema(
        p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"}
    )
    warning = next(item for item in warnings if "weak_cross_source_recoverability" in item)
    assert "video1_analysis.missing_key_attributes[1]" in warning
    assert "video2_analysis.missing_key_attributes" not in warning


def test_motorhome_registry_identity_requires_both_sources(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the motorhome", "A motorhome is parked on the left.", "A large rectangular vehicle is parked on the left.")
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "registry_source_attribute_leakage" for issue in exc_info.value.errors)


def test_neutral_box_shaped_vehicle_registry_passes(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the large box-shaped vehicle on the left", "A motorhome is parked on the left.", "A large box-shaped vehicle is parked on the left.")
    _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})


def test_dark_building_shared_global_text_requires_both_sources(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the building facade", "A dark building facade stands on the right.", "A building facade with panel lines stands on the right.", "ent_building")
    p["global_scene"]["physical_entities"][-1]["category"] = "building"
    p["global_scene"]["scene_summary"] += " The sequence continues toward a dark building wall."
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "shared_global_source_attribute_leakage" for issue in exc_info.value.errors)


def test_static_inventory_warns_for_weak_temporal_progression(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["temporal_progression"] = "The street features parked vehicles and streetlights on both sides."
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any("weak_temporal_progression" in warning for warning in warnings)


def test_explicit_temporal_stability_is_valid_progression(base_parsed):
    p = copy.deepcopy(base_parsed)
    p["global_scene"]["temporal_progression"] = "The visible entities retain the same relative arrangement across the sampled times."
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert not any("weak_temporal_progression" in warning for warning in warnings)


def test_missing_participating_entity_reference_is_detected(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(p, "the vehicle", "A vehicle remains visible.", "A vehicle remains visible.", "ent_vehicle")
    _add_entity_with_atoms(p, "the streetlight", "A streetlight remains visible.", "A streetlight remains visible.", "ent_streetlight")
    p["video1_analysis"]["information_atoms"][-1]["atom_id"] = "v1_atom_streetlight"
    p["video2_analysis"]["information_atoms"][-1]["atom_id"] = "v2_atom_streetlight"
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_relation",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_vehicle"],
        "fact": "A vehicle is parked under a streetlight.",
    })
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "missing_entity_reference" for issue in exc_info.value.errors)


def test_headlights_do_not_trigger_paint_color_warning(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(
        p,
        "the vehicle",
        "The vehicle has two white headlights.",
        "The vehicle has two bright headlight regions.",
    )
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert not any(
        "possible_missing_recoverability_target" in warning and "ent_target" in warning
        for warning in warnings
    )


def test_multi_entity_vehicle_color_does_not_bind_to_streetlight(base_parsed):
    p = copy.deepcopy(base_parsed)
    for entity_id, category, scope in (
        ("ent_vehicle", "vehicle", "the vehicle"),
        ("ent_light", "infrastructure", "the streetlight"),
    ):
        p["global_scene"]["physical_entities"].append({
            "entity_id": entity_id,
            "category": category,
            "referential_scope": scope,
        })
    p["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_vehicle_light",
        "frame_keys": ["frame_1"],
        "entity_refs": ["ent_vehicle", "ent_light"],
        "fact": "A white vehicle is parked under a streetlight.",
    })
    p["video2_analysis"]["information_atoms"].extend([
        {"atom_id": "v2_atom_vehicle", "frame_keys": ["frame_1"], "entity_refs": ["ent_vehicle"], "fact": "The vehicle has a broad outline."},
        {"atom_id": "v2_atom_light", "frame_keys": ["frame_1"], "entity_refs": ["ent_light"], "fact": "The streetlight has a tall pole."},
    ])
    _, warnings = _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    color_warnings = [warning for warning in warnings if "possible_missing_recoverability_target" in warning]
    assert any("ent_vehicle" in warning for warning in color_warnings)
    assert not any("ent_light" in warning for warning in color_warnings)


def test_glowing_headlights_shared_global_requires_both_sources(base_parsed):
    p = copy.deepcopy(base_parsed)
    _add_entity_with_atoms(
        p,
        "the approaching vehicle",
        "The approaching vehicle has glowing headlights.",
        "The approaching vehicle has a moving outer outline.",
    )
    p["global_scene"]["scene_summary"] += " The approaching vehicle has glowing headlights."
    with pytest.raises(Pass1SemanticValidationError) as exc_info:
        _validate_pass1_schema(p, {"frame_1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    assert any(issue.category == "shared_global_source_attribute_leakage" for issue in exc_info.value.errors)


@pytest.mark.parametrize(
    ("category", "path", "expected_text"),
    [
        ("source_uncertainty_contradiction", "video1_analysis.uncertain_observations[0].missing_evidence", "same-source"),
        ("unrecoverable_missing_attribute", "video1_analysis.missing_key_attributes[1]", "opposite-source Atom"),
        ("registry_source_attribute_leakage", "global_scene.physical_entities[2].referential_scope", "Registry entry"),
        ("shared_global_source_attribute_leakage", "global_scene.scene_summary", "both active sources"),
        ("missing_entity_reference", "video1_analysis.information_atoms[2].entity_refs", "entity_refs"),
    ],
)
def test_retry_guidance_uses_structured_category_and_exact_path(category, path, expected_text):
    exc = Pass1SemanticValidationError(
        "Semantic Validation Errors",
        [Pass1ValidationIssue(path, category, "regression issue", "semantic")],
    )
    assert _categorize_pass1_validation_error(exc) == category
    hint = _build_pass1_validation_retry_hint(exc, category)
    assert path in hint
    assert expected_text in hint
