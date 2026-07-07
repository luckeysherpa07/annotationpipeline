import sys
import json
import py_compile
from pathlib import Path
from annotation_feature.aligned_multimodal_caption_pipeline import (
    _validate_caption_schema,
    _task_to_item,
    CaptionTask,
    CAPTION_SCHEMA_VERSION,
    CaptionValidationError,
    _template_caption,
    _build_prompt_schema_example,
    _load_resume,
    _derive_reasoning_focus_entities,
    _contains_term,
    _contains_any_term
)

def run_tests():
    print("Running tests...")

    # Test 1: Syntax / import sanity
    py_compile.compile("annotation_feature/aligned_multimodal_caption_pipeline.py", doraise=True)
    print("Test 1 (Syntax / import sanity) passed")

    # Test 2: Safe whole-word matching
    assert _contains_term("A car is visible.", "car")
    assert not _contains_term("A shopping cart is visible.", "car")
    assert not _contains_term("A scarf lies nearby.", "car")
    assert not _contains_term("A bush moves.", "bus")
    assert not _contains_term("An advanced object appears.", "van")
    print("Test 2 (Safe whole-word matching) passed")

    # Test 3: Multi-word phrase matching
    assert _contains_term("A box-shaped vehicle is visible.", "box-shaped vehicle")
    assert _contains_term("The object has a tall rear body.", "tall rear body")
    print("Test 3 (Multi-word phrase matching) passed")

    # Shared minimal valid caption setup
    minimal_caption = {
        "schema_version": CAPTION_SCHEMA_VERSION,
        "global_scene": {
            "scene_summary": "This is a detailed paragraph covering the environment. " * 5,
            "environment": "urban",
            "temporal_progression": "The vehicle moves forward. " * 3,
            "physical_entities": [
                {
                    "entity_id": "entity_001",
                    "category": "vehicle",
                    "evidence_profile": {
                        "identity_evidence": ["Shape"],
                        "observable_attributes": ["Fast"],
                        "spatial_context": ["Center"]
                    }
                }
            ]
        },
        "video1_analysis": {
            "modality": "rgb",
            "detailed_caption": "This is a detailed caption paragraph for video one. " * 6,
            "information_atoms": [
                {"atom_id": "v1_atom_001", "frame_keys": ["frame_0001"], "fact": "Red car is present."}
            ],
            "sensor_specific_cues": ["Cue 1"],
            "sensor_limitations": ["Limit 1"],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": "event",
            "detailed_caption": "This is a detailed caption paragraph for video two. " * 6,
            "information_atoms": [
                {"atom_id": "v2_atom_001", "frame_keys": ["frame_0001"], "fact": "Moving edges observed."}
            ],
            "sensor_specific_cues": ["Cue 2"],
            "sensor_limitations": ["Limit 2"],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "cross_modal_evidence_links": [
            {
                "entity_id": "entity_001",
                "video1_evidence_refs": ["v1_atom_001"],
                "video2_evidence_refs": ["v2_atom_001"],
                "shared_evidence": ["Both show vehicle"],
                "unique_to_video1": ["Color"],
                "unique_to_video2": [],
                "directional_contributions": [
                    {"direction": "video1_improves_video2", "contribution": "Adds color"}
                ]
            }
        ],
        "information_gain": [
            {
                "entity_id": "entity_001",
                "video1_evidence_refs": ["v1_atom_001"],
                "video2_evidence_refs": ["v2_atom_001"],
                "video1_can_determine": ["Color"],
                "video1_cannot_determine": [],
                "video2_can_determine": ["Edges"],
                "video2_cannot_determine": [],
                "fusion_additionally_reveals": [],
                "gain_type": "complementarity",
                "gain_rating": "high"
            }
        ],
        "reasoning_events": [
            {
                "event_id": "evt_001",
                "event_type": "joint_fusion",
                "participating_entities": ["entity_001"],
                "supporting_atom_refs": ["v1_atom_001", "v2_atom_001"],
                "description": "Fusion event"
            }
        ],
        "ambiguity_events": [],
        "qa_relevant_details": [
            {
                "detail_id": "qa_detail_001",
                "reasoning_pattern": "joint_fusion",
                "supporting_refs": ["evt_001"],
                "why_question_worthy": "Good question"
            }
        ],
        "rejected_observations": [
            {"observation": "Nothing", "reason": "No ambiguity"}
        ]
    }
    
    def test_recovery(missing_from, recovering_from, missing_video, recovering_video, ref_atoms, attr_type, missing_attr, should_fail, atom_facts):
        cap = eval(repr(minimal_caption))
        cap[f"video{missing_video}_analysis"]["modality"] = missing_from
        cap[f"video{recovering_video}_analysis"]["modality"] = recovering_from
        for idx, (ref_atom, fact) in enumerate(zip(ref_atoms, atom_facts)):
            if idx == 0:
                cap[f"video{recovering_video}_analysis"]["information_atoms"][0]["atom_id"] = ref_atom
                cap[f"video{recovering_video}_analysis"]["information_atoms"][0]["fact"] = fact
            else:
                cap[f"video{recovering_video}_analysis"]["information_atoms"].append({"atom_id": ref_atom, "frame_keys": ["frame_0001"], "fact": fact})
        
        cap[f"video{missing_video}_analysis"]["missing_key_attributes"].append({
            "attribute_type": attr_type,
            "missing_attribute": missing_attr,
            "why_missing": "Poor lighting",
            "recoverable_evidence_refs": ref_atoms
        })
        try:
            validated, warn = _validate_caption_schema(cap, {"frame_0001"}, missing_from if missing_video==1 else recovering_from, recovering_from if recovering_video==2 else missing_from)
            if should_fail:
                assert False, f"Should have failed recovery of '{missing_attr}' from '{recovering_from}' with facts '{atom_facts}'"
            return warn
        except CaptionValidationError as e:
            if not should_fail:
                assert False, f"Should NOT have failed recovery of '{missing_attr}' from '{recovering_from}'. Error: {e}"
            return ["rejected"]

    # Test 4: Vehicle target + cart fact
    res = test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "semantic_identity", "vehicle category", True, ["A person pushes a shopping cart beside the curb."])
    print("Test 4 (Vehicle target + cart fact) passed")

    # Test 5: Vehicle target + scarf fact
    res = test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "semantic_identity", "vehicle category", False, ["A scarf lies on the pavement."])
    assert res and "recovered via conditional capability" in res[0]
    print("Test 5 (Vehicle target + scarf fact) passed")

    # Test 6: Vehicle target + bush fact
    res = test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "semantic_identity", "vehicle category", False, ["A bush moves in the wind near the sidewalk."])
    assert res and "recovered via conditional capability" in res[0]
    print("Test 6 (Vehicle target + bush fact) passed")

    # Test 7: Vehicle target + advanced fact
    res = test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "semantic_identity", "vehicle category", False, ["An advanced object appears near the curb."])
    assert res and "recovered via conditional capability" in res[0]
    print("Test 7 (Vehicle target + advanced fact) passed")

    # Test 8: Vehicle target + real car fact
    res = test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "semantic_identity", "vehicle category", False, ["A car is parked near the curb."])
    print("Test 8 (Vehicle target + real car fact) passed")

    # Test 9: Vehicle target + plausible van fact
    res = test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "semantic_identity", "vehicle category", False, ["A box-shaped road vehicle with a tall rear body is visible."])
    print("Test 9 (Vehicle target + plausible van fact) passed")

    # Test 10: Person target
    res = test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "semantic_identity", "pedestrian category", False, ["A walking human figure is visible near the curb."])
    print("Test 10 (Person target) passed")

    # Test 11: Mixed-side recovery refs
    test_mixed_refs = eval(repr(minimal_caption))
    test_mixed_refs["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "semantic_identity",
        "missing_attribute": "vehicle category",
        "why_missing": "Darkness",
        "recoverable_evidence_refs": ["v2_atom_001", "v1_atom_001"]
    })
    try:
        _validate_caption_schema(test_mixed_refs, {"frame_0001"}, "rgb", "event")
        assert False, "Should have failed mixed side recovery refs"
    except CaptionValidationError as e:
        assert "Invalid cross-side ref" in str(e)
    print("Test 11 (Mixed-side recovery refs) passed")

    # Test 12: Event -> color
    test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "surface_attribute", "Vehicle color", True, ["Red paint."])
    print("Test 12 (Event -> color must reject) passed")
    
    # Test 13: Event -> metric depth
    test_recovery("rgb", "event", 1, 2, ["v2_atom_001"], "spatial_relation", "metric depth", True, ["Object is at 5 meters."])
    print("Test 13 (Event -> metric depth must reject) passed")

    # Test 14: Depth -> metric depth
    test_recovery("event", "depth", 1, 2, ["v2_atom_001"], "spatial_relation", "metric depth", False, ["Car is at 5 meters."])
    print("Test 14 (Depth -> metric depth must pass) passed")

    # Test 15: IR -> thermal
    test_recovery("rgb", "ir", 1, 2, ["v2_atom_001"], "state_attribute", "thermal signature", False, ["High heat signature."])
    print("Test 15 (IR -> thermal must pass) passed")

    task = CaptionTask(
        caption_id="test", segment_id="seg1", split_dir="sp1", segment_name="sn1",
        side="day", modality1="rgb", modality2="event",
        frame_dir1=Path(""), frame_dir2=Path(""), frames1=(), frames2=(),
        composite_frames=(Path("frame_0001.png"),), sampling_strategy="test",
        uniform_anchor_indexes=(), adaptive_frame_indexes=(),
        selected_frame_indexes=(), candidate_frame_indexes=(),
        selection_config_fingerprint="test"
    )

    # Test 16: Prompt example validates
    example = _build_prompt_schema_example(task)
    validated_example, ex_warn = _validate_caption_schema(example, {"frame_0001"}, "rgb", "event")
    assert ex_warn == [], f"Prompt example generated warnings: {ex_warn}"
    print("Test 16 (Prompt example validates and no warnings) passed")

    # Test 17: Template validates
    template_cap = _template_caption(task)
    _validate_caption_schema(template_cap, {"frame_0001"}, "rgb", "event")
    print("Test 17 (Template validates) passed")

    # Test 18: Asymmetric frame refs
    test_d_caption = eval(repr(minimal_caption))
    test_d_caption["video1_analysis"]["information_atoms"][0]["frame_keys"] = ["frame_000450", "frame_000480"]
    test_d_caption["video2_analysis"]["information_atoms"][0]["frame_keys"] = ["frame_000480"]
    _validate_caption_schema(test_d_caption, {"frame_000450", "frame_000480"}, "rgb", "event")
    print("Test 18 (Asymmetric frame refs) passed")

    # Test 19: Old v10 resume rejection
    old_cap = {
        "caption_id": "test_old_v10",
        "caption": {
            "schema_version": "cross_modal_disambiguation_caption_v10",
            "global_scene": {},
            "video1_analysis": {},
            "video2_analysis": {},
            "cross_modal_evidence_links": [
                {
                    "shared_evidence": "old string",
                    "unique_to_video1": "old string",
                    "unique_to_video2": "old string"
                }
            ],
            "information_gain": [],
            "reasoning_events": [],
            "ambiguity_events": [],
            "qa_relevant_details": [],
            "rejected_observations": [],
        }
    }
    temp_file = Path("temp_resume_v10.json")
    with open(temp_file, "w") as f:
        json.dump({"items": [old_cap]}, f)
    valid_items, _ = _load_resume(temp_file)
    assert len(valid_items) == 0, "Old v10 caption must be rejected by resume"
    temp_file.unlink()
    print("Test 19 (Old v10 resume rejection) passed")

    # Test 20: Valid v11 resume acceptance
    v11_cap = {
        "caption_id": "test_v11",
        "caption": minimal_caption
    }
    temp_file = Path("temp_resume_v11.json")
    with open(temp_file, "w") as f:
        json.dump({"items": [v11_cap]}, f)
    valid_items, _ = _load_resume(temp_file)
    assert len(valid_items) == 1, "Valid v11 caption must be accepted by resume"
    temp_file.unlink()
    print("Test 20 (Valid v11 resume acceptance) passed")

    print("All tests passed!")

if __name__ == "__main__":
    run_tests()
