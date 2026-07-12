import pytest
from annotation_feature.aligned_caption_pass1_validation import (
    _validate_pass1_structure,
    _validate_pass1_semantics,
    Pass1StructuralValidationError,
    Pass1SemanticValidationError,
    Pass1ValidationContext,
    Pass1ValidationIssue
)
from annotation_feature.aligned_caption_schema import CaptionValidationError
from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _build_pass1_validation_retry_hint

def test_structural_staging_no_cascade():
    parsed = {}
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    
    errs = exc.value.errors
    assert any(e.path == "video1_analysis" and e.category == "missing_field" for e in errs)
    assert any(e.path == "video2_analysis" and e.category == "missing_field" for e in errs)
    # Ensure no cascades for missing inner fields like modality, information_atoms, etc.
    assert not any(e.path.startswith("video1_analysis.") for e in errs)

def test_structural_validation_error_aggregation():
    # Provide a root but mess up Stage B and C
    parsed = {
        "global_scene": {
            "scene_summary": "valid",
            "environment": "urban",
            "temporal_progression": "moving",
            "physical_entities": [
                {"entity_id": "ent1", "category": "car", "referential_scope": "the car"},
                {"entity_id": "ent2"} # Missing category
            ]
        },
        "video1_analysis": {
            "modality": "rgb",
            "information_atoms": [
                {"atom_id": "v1_atom_1", "frame_keys": ["f3"], "entity_refs": ["ent3"], "fact": "f"} # invalid frame, unknown entity
            ]
        },
        "video2_analysis": {
            "modality": "lidar",
            "information_atoms": []
        }
    }
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "video", "video2_analysis": "lidar"})
    
    errs = exc.value.errors
    assert any(e.category == "missing_field" and "category" in e.path for e in errs) # ent2 missing category
    assert any(e.category == "invalid_frame_reference" and "f3" in e.message for e in errs)
    assert any(e.category == "invalid_entity_reference" and "ent3" in e.message for e in errs)
    assert any(e.category == "invalid_type" and "information_atoms" in e.path for e in errs) # video2 empty list

def test_semantic_validation_aggregation():
    parsed = {
        "global_scene": {
            "scene_summary": "this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words",
            "environment": "urban",
            "temporal_progression": "this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words",
            "physical_entities": [{"entity_id": "ent1", "category": "object", "referential_scope": "test"}]
        },
        "video1_analysis": {
            "modality": "rgb",
            "detailed_caption": "this is resolved as silhouettes. " * 5,
            "information_atoms": [{"atom_id": "v1_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact"}],
            "missing_key_attributes": [
                {"entity_id": "ent1", "attribute_type": "surface_attribute", "missing_attribute": "color", "why_missing": "the sensor does not record color", "recoverable_evidence_refs": []}
            ],
            "sensor_limitations": [
                "the sensor does not record color"
            ],
            "sensor_specific_cues": ["cues"],
            "uncertain_observations": []
        },
        "video2_analysis": {
            "modality": "event",
            "detailed_caption": "this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words this is enough words",
            "information_atoms": [{"atom_id": "v2_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact"}],
            "missing_key_attributes": [],
            "sensor_limitations": [],
            "sensor_specific_cues": ["cues"],
            "uncertain_observations": []
        }
    }
    # Mock context
    raw = parsed
    context = _validate_pass1_structure(raw, {"f1"}, {"video1_analysis": "rgb", "video2_analysis": "event"})
    with pytest.raises(Pass1SemanticValidationError) as exc:
        _validate_pass1_semantics(parsed, context)
        
    errs = exc.value.errors
    # Should catch 'resolved as' (physical_world_wording) AND 'the sensor does not record color' (generic_sensor_theory)
    assert any(e.category == "physical_world_wording" and "video1" in e.path for e in errs)
    assert any(e.category == "generic_sensor_theory" and "why_missing" in e.path for e in errs)


import copy
from annotation_feature.aligned_caption_pass1_validation import _validate_pass1_schema

def test_input_immutability():
    # Successful case
    parsed = {
        "global_scene": {
            "scene_summary": "this is a valid summary that is long enough to satisfy the global scene summary min word count which is around twenty words so I am adding more text here to be safe",
            "environment": "urban",
            "temporal_progression": "moving along is the temporal progression that describes how the video content changes over time and space throughout the clip",
            "physical_entities": [
                {"entity_id": "ent1", "category": "car", "referential_scope": "the car"}
            ]
        },
        "video1_analysis": {
            "modality": "rgb",
            "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and lidar representations so the pipeline test doesn't fail again.",
            "information_atoms": [{"atom_id": "v1_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}],
            "sensor_specific_cues": ["cue"],
            "sensor_limitations": ["The car surface remains unclear in f1."],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": "depth",
            "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and lidar representations so the pipeline test doesn't fail again.",
            "information_atoms": [{"atom_id": "v2_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}],
            "sensor_specific_cues": ["cue"],
            "sensor_limitations": ["The car boundary remains unclear in f1."],
            "uncertain_observations": [],
            "missing_key_attributes": []
        }
    }
    before = copy.deepcopy(parsed)
    _validate_pass1_schema(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "depth"})
    assert parsed == before

    # Structural failure case
    parsed_bad_struct = copy.deepcopy(parsed)
    parsed_bad_struct["video1_analysis"]["sensor_specific_cues"] = [123]
    before_bad_struct = copy.deepcopy(parsed_bad_struct)
    with pytest.raises(Pass1StructuralValidationError):
        _validate_pass1_schema(parsed_bad_struct, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "depth"})
    assert parsed_bad_struct == before_bad_struct

    # Semantic failure case
    parsed_bad_sem = copy.deepcopy(parsed)
    parsed_bad_sem["global_scene"]["scene_summary"] = "The sensor does not record color in this generic description. " * 4
    before_bad_sem = copy.deepcopy(parsed_bad_sem)
    with pytest.raises(Pass1SemanticValidationError):
        _validate_pass1_schema(parsed_bad_sem, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "depth"})
    assert parsed_bad_sem == before_bad_sem

def test_cascade_prevention():
    # 1. Invalid evidence-ref type does not trigger connection error
    parsed = {
        "global_scene": {
            "scene_summary": "this is a valid summary that is long enough to satisfy the global scene summary min word count which is around twenty words so I am adding more text here to be safe",
            "environment": "urban",
            "temporal_progression": "moving along is the temporal progression that describes how the video content changes over time and space throughout the clip",
            "physical_entities": [
                {"entity_id": "ent1", "category": "car", "referential_scope": "the car"}
            ]
        },
        "video1_analysis": {
            "modality": "rgb",
            "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and lidar representations so the pipeline test doesn't fail again.",
            "information_atoms": [{"atom_id": "v1_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}],
            "sensor_specific_cues": ["cue"],
            "sensor_limitations": ["limitation"],
            "uncertain_observations": [
                {
                    "uncertainty_id": "v1_unc_1",
                    "entity_id": "ent1",
                    "observed_evidence": "ev",
                    "missing_evidence": "ev",
                    "evidence_refs": [{}] # invalid type
                }
            ],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": "lidar",
            "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and lidar representations so the pipeline test doesn't fail again.",
            "information_atoms": [{"atom_id": "v2_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}],
            "sensor_specific_cues": ["cue"],
            "sensor_limitations": ["limitation"],
            "uncertain_observations": [],
            "missing_key_attributes": []
        }
    }
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "lidar"})
    errs = exc.value.errors
    assert any(e.category == "invalid_type" and "evidence_refs" in e.path for e in errs)
    assert not any(e.category == "invalid_atom_entity_connection" for e in errs)

    # 2. Wrong prefix does not trigger connection error
    parsed["video1_analysis"]["uncertain_observations"][0]["evidence_refs"] = ["v2_atom_1"]
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "lidar"})
    errs = exc.value.errors
    assert any(e.category == "invalid_reference" and "evidence_refs" in e.path for e in errs)
    assert not any(e.category == "invalid_atom_entity_connection" for e in errs)

    # 3. Unknown atom does not trigger connection error
    parsed["video1_analysis"]["uncertain_observations"][0]["evidence_refs"] = ["v1_atom_999"]
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "lidar"})
    errs = exc.value.errors
    assert any(e.category == "invalid_reference" and "evidence_refs" in e.path for e in errs)
    assert not any(e.category == "invalid_atom_entity_connection" for e in errs)

    # 4. Unknown entity does not trigger connection error
    parsed["video1_analysis"]["uncertain_observations"][0]["evidence_refs"] = ["v1_atom_1"]
    parsed["video1_analysis"]["uncertain_observations"][0]["entity_id"] = "ent999"
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "lidar"})
    errs = exc.value.errors
    assert any(e.category == "invalid_entity_reference" and "entity_id" in e.path for e in errs)
    assert not any(e.category == "invalid_atom_entity_connection" for e in errs)

    # 5. Valid disconnected atom/entity triggers connection error
    parsed["global_scene"]["physical_entities"].append({"entity_id": "ent2", "category": "person", "referential_scope": "the person"})
    parsed["video1_analysis"]["uncertain_observations"][0]["entity_id"] = "ent2"
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "lidar"})
    errs = exc.value.errors
    assert any(e.category == "invalid_atom_entity_connection" for e in errs)

def test_scope_in_unexpected_field():
    parsed = {
        "global_scene": {
            "scene_summary": "this is a valid summary that is long enough to satisfy the global scene summary min word count which is around twenty words so I am adding more text here to be safe",
            "environment": "urban",
            "temporal_progression": "moving along is the temporal progression that describes how the video content changes over time and space throughout the clip",
            "physical_entities": [
                {"entity_id": "ent1", "category": "car", "referential_scope": "the car", "extra_ent": "x"}
            ],
            "extra_gs": "y"
        },
        "video1_analysis": {
            "modality": "rgb",
            "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and lidar representations so the pipeline test doesn't fail again.",
            "information_atoms": [{"atom_id": "v1_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}],
            "sensor_specific_cues": ["cue"],
            "sensor_limitations": ["limitation"],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": "lidar",
            "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and lidar representations so the pipeline test doesn't fail again.",
            "information_atoms": [{"atom_id": "v2_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}],
            "sensor_specific_cues": ["cue"],
            "sensor_limitations": ["limitation"],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "extra_root": "z"
    }
    with pytest.raises(Pass1StructuralValidationError) as exc:
        _validate_pass1_structure(parsed, {"f1", "f2"}, {"video1_analysis": "rgb", "video2_analysis": "lidar"})
    errs = exc.value.errors
    assert any(e.category == "unexpected_field" and e.scope == "root" and "extra_root" in e.path for e in errs)
    assert any(e.category == "unexpected_field" and e.scope == "global_scene" and "extra_gs" in e.path for e in errs)
    assert any(e.category == "unexpected_field" and e.scope == "entity" and "extra_ent" in e.path for e in errs)
