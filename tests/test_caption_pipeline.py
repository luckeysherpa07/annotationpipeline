import sys
import json
import py_compile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_feature.aligned_multimodal_caption_pipeline import (
    _task_to_item,
    CaptionTask,
    CaptionValidationError,
    _template_caption,
    _build_caption_prompt,
    _load_resume,
    build_caption_tasks,
    _build_validation_retry_hint,
)
from annotation_feature.aligned_caption_prompt import _build_prompt_schema_example
from annotation_feature.aligned_caption_validation import (
    _contains_any_term,
    _contains_term,
    _validate_caption_schema,
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
        "global_scene": {
            "scene_summary": "This is a detailed paragraph covering the environment. " * 5,
            "environment": "urban",
            "temporal_progression": "The vehicle moves forward. " * 3,
            "physical_entities": [
                {
                    "entity_id": "entity_001",
                    "category": "vehicle",
                    "referential_scope": "the vehicle tracked as the main reasoning target",
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
                {"atom_id": "v1_atom_001", "frame_keys": ["frame_0001"], "entity_refs": ["entity_001"], "fact": "Red car is present."}
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
                {"atom_id": "v2_atom_001", "frame_keys": ["frame_0001"], "entity_refs": ["entity_001"], "fact": "Moving edges observed."}
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
                cap[f"video{recovering_video}_analysis"]["information_atoms"].append({"atom_id": ref_atom, "frame_keys": ["frame_0001"], "entity_refs": ["entity_001"], "fact": fact})
        
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

    # Test 17: Stage-2 prompt consolidation preserves semantic invariants
    prompt_text = _build_caption_prompt(task)
    consolidated_blocks = [
        "ENTITY REGISTRY CONSTRUCTION AND INVARIANTS",
        "ATOM AND PROVENANCE INVARIANTS",
        "GRAPH CONSTRUCTION AND EVIDENCE-CLOSURE WORKFLOW",
        "FINAL GRAPH CONSISTENCY SELF-CHECK",
    ]
    for block in consolidated_blocks:
        assert prompt_text.count(block) == 1, f"Missing or duplicated consolidated prompt block: {block}"
    removed_blocks = [
        "SAME-OBJECT IDENTITY CONSISTENCY",
        "ENTITY SCOPE EXCLUSIVITY",
        "GROUP MEMBER PROMOTION RULE",
        "SHARED ENTITY REGISTRY WORKFLOW",
        "ATOM RECONCILIATION / EVIDENCE COMPLETION",
        "ATOM CLOSURE WORKFLOW",
        "FINAL EVIDENCE-CLOSURE SELF-CHECK",
    ]
    for block in removed_blocks:
        assert block not in prompt_text, f"Old duplicate prompt block still present: {block}"
    assert "Each information atom is one minimal" in prompt_text
    assert "multiple independently testable observations" in prompt_text
    assert "directly observable relation, interaction, or joint event" in prompt_text
    assert "A multi-entity atom is valid" in prompt_text
    assert "Evidence validity is determined by atom.fact semantics" in prompt_text
    assert "Same-object continuity" in prompt_text
    assert "reuse one entity_id" in prompt_text
    assert "state, action, position, visibility, or modality-specific appearance changes do not create a new Entity" in prompt_text
    assert "Frame co-occurrence does not imply semantic support" in prompt_text
    assert "CAPTION_SCHEMA_VERSION" not in prompt_text
    assert "schema_version" not in prompt_text
    assert "cross_modal_disambiguation_caption_v" not in prompt_text
    assert "v12" not in prompt_text
    assert "v13" not in prompt_text
    print("Test 17 (Stage-2 prompt consolidation regression) passed")

    # Test 18: Template validates
    template_cap = _template_caption(task)
    _validate_caption_schema(template_cap, {"frame_0001"}, "rgb", "event")
    print("Test 18 (Template validates) passed")

    # Test 19: Asymmetric frame refs
    test_d_caption = eval(repr(minimal_caption))
    test_d_caption["video1_analysis"]["information_atoms"][0]["frame_keys"] = ["frame_000450", "frame_000480"]
    test_d_caption["video2_analysis"]["information_atoms"][0]["frame_keys"] = ["frame_000480"]
    _validate_caption_schema(test_d_caption, {"frame_000450", "frame_000480"}, "rgb", "event")
    print("Test 19 (Asymmetric frame refs) passed")

    example_entities = example["global_scene"]["physical_entities"]
    assert sum(1 for entity in example_entities if entity["entity_id"] == "entity_002") == 1
    v1_entity_002_atoms = [
        atom for atom in example["video1_analysis"]["information_atoms"]
        if "entity_002" in atom["entity_refs"]
    ]
    assert len(v1_entity_002_atoms) >= 2, "Same BMW across time should use one entity with multiple atoms"
    residual_group = next(entity for entity in example_entities if entity["entity_id"] == "entity_003")
    assert "excluding entity_002" in residual_group["referential_scope"]
    print("Test 20 (Prompt example same-object and residual-group fixture) passed")

    def assert_validation_fails(cap, expected_message):
        try:
            _validate_caption_schema(cap, {"frame_0001"}, "rgb", "event")
            assert False, f"Expected validation failure containing: {expected_message}"
        except CaptionValidationError as e:
            assert expected_message in str(e), f"Expected '{expected_message}' in '{e}'"
            return str(e)

    # Test 21: Asymmetric cross-modal link is accepted
    asymmetric_link_cap = eval(repr(minimal_caption))
    asymmetric_link_cap["cross_modal_evidence_links"][0]["video1_evidence_refs"] = ["v1_atom_001"]
    asymmetric_link_cap["cross_modal_evidence_links"][0]["video2_evidence_refs"] = []
    asymmetric_link_cap["cross_modal_evidence_links"][0]["shared_evidence"] = []
    asymmetric_link_cap["cross_modal_evidence_links"][0]["unique_to_video1"] = [
        "Video 1 provides unique evidence."
    ]
    _validate_caption_schema(asymmetric_link_cap, {"frame_0001"}, "rgb", "event")
    print("Test 21 (Asymmetric cross-modal link accepted) passed")

    # Test 22: Fully unsupported cross-modal link is rejected
    unsupported_link_cap = eval(repr(minimal_caption))
    unsupported_link_cap["cross_modal_evidence_links"][0]["video1_evidence_refs"] = []
    unsupported_link_cap["cross_modal_evidence_links"][0]["video2_evidence_refs"] = []
    assert_validation_fails(unsupported_link_cap, "must cite at least one source-local evidence atom")
    print("Test 22 (Unsupported cross-modal link rejected) passed")

    # Test 23: Asymmetric non-confirmation information gain is accepted
    asymmetric_gain_cap = eval(repr(minimal_caption))
    asymmetric_gain_cap["information_gain"][0]["gain_type"] = "disambiguation"
    asymmetric_gain_cap["information_gain"][0]["video1_evidence_refs"] = ["v1_atom_001"]
    asymmetric_gain_cap["information_gain"][0]["video2_evidence_refs"] = []
    _validate_caption_schema(asymmetric_gain_cap, {"frame_0001"}, "rgb", "event")
    print("Test 23 (Asymmetric non-confirmation information gain accepted) passed")

    # Test 24: Confirmation information gain still requires both videos
    asymmetric_confirmation_cap = eval(repr(minimal_caption))
    asymmetric_confirmation_cap["information_gain"][0]["gain_type"] = "confirmation"
    asymmetric_confirmation_cap["information_gain"][0]["video1_evidence_refs"] = ["v1_atom_001"]
    asymmetric_confirmation_cap["information_gain"][0]["video2_evidence_refs"] = []
    assert_validation_fails(asymmetric_confirmation_cap, "confirmation' requires evidence from both videos")
    print("Test 24 (Confirmation information gain requires both videos) passed")

    # Test 25: Fully unsupported information gain is rejected
    unsupported_gain_cap = eval(repr(minimal_caption))
    unsupported_gain_cap["information_gain"][0]["gain_type"] = "disambiguation"
    unsupported_gain_cap["information_gain"][0]["video1_evidence_refs"] = []
    unsupported_gain_cap["information_gain"][0]["video2_evidence_refs"] = []
    assert_validation_fails(unsupported_gain_cap, "must cite at least one source-local evidence atom")
    print("Test 25 (Unsupported information gain rejected) passed")

    # Test 26: Exact normalized duplicate referential_scope rejection
    exact_white_car_duplicate = eval(repr(minimal_caption))
    exact_white_car_duplicate["global_scene"]["physical_entities"][0]["referential_scope"] = "The   White Car"
    exact_white_car_duplicate["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "the white car"
    })
    assert_validation_fails(exact_white_car_duplicate, "Duplicate normalized referential_scope detected")
    print("Test 26 (Exact normalized duplicate scope rejected) passed")

    # Test 27: Semantically similar scopes are not fuzzily rejected
    similar_non_duplicate_scope_cap = eval(repr(minimal_caption))
    similar_non_duplicate_scope_cap["global_scene"]["physical_entities"][0]["referential_scope"] = "the white car"
    similar_non_duplicate_scope_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "the white vehicle"
    })
    _validate_caption_schema(similar_non_duplicate_scope_cap, {"frame_0001"}, "rgb", "event")
    print("Test 27 (Similar non-identical scope not fuzzily rejected) passed")

    # Test 28: Tree-shadow wrong atom rejection for information_gain
    tree_wrong_gain = eval(repr(minimal_caption))
    tree_wrong_gain["global_scene"]["physical_entities"].append({
        "entity_id": "entity_005",
        "category": "tree_shadow",
        "referential_scope": "the branching tree-shadow phenomenon on the road surface"
    })
    tree_wrong_gain["information_gain"][0]["entity_id"] = "entity_005"
    assert_validation_fails(tree_wrong_gain, "not explicitly connected to entity entity_005")
    print("Test 28 (Tree-shadow wrong atom rejection) passed")

    # Test 29: Normalized duplicate referential_scope rejection
    duplicate_scope_cap = eval(repr(minimal_caption))
    duplicate_scope_cap["global_scene"]["physical_entities"][0]["referential_scope"] = (
        "The specific white sedan tracked across the sampled interval"
    )
    duplicate_scope_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "  the   SPECIFIC white sedan tracked across the sampled interval  "
    })
    duplicate_scope_error = assert_validation_fails(duplicate_scope_cap, "Duplicate normalized referential_scope detected")
    assert "entity_001" in duplicate_scope_error and "entity_002" in duplicate_scope_error
    print("Test 29 (Normalized duplicate referential_scope rejection) passed")

    # Test 30: Different referential scopes remain valid
    distinct_scope_cap = eval(repr(minimal_caption))
    distinct_scope_cap["global_scene"]["physical_entities"][0]["referential_scope"] = (
        "the specific white sedan tracked across the sampled interval"
    )
    distinct_scope_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "parked_vehicle_group",
        "referential_scope": "the other parked vehicles excluding entity_001"
    })
    _validate_caption_schema(distinct_scope_cap, {"frame_0001"}, "rgb", "event")
    print("Test 30 (Different referential scopes remain valid) passed")

    # Test 31: Similar but distinct referential scopes are not fuzzy-rejected
    similar_distinct_scope_cap = eval(repr(minimal_caption))
    similar_distinct_scope_cap["global_scene"]["physical_entities"][0]["referential_scope"] = (
        "the white sedan nearest the foreground"
    )
    similar_distinct_scope_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "the white sedan farther down the curb"
    })
    _validate_caption_schema(similar_distinct_scope_cap, {"frame_0001"}, "rgb", "event")
    print("Test 31 (Similar but distinct referential scopes are not fuzzy-rejected) passed")

    # Test 32: Same-object valid graph with multiple atoms passes
    same_object_cap = eval(repr(minimal_caption))
    same_object_cap["global_scene"]["physical_entities"][0]["entity_id"] = "entity_002"
    same_object_cap["global_scene"]["physical_entities"][0]["referential_scope"] = (
        "the specific white BMW tracked across the sampled interval"
    )
    same_object_cap["video1_analysis"]["information_atoms"][0]["entity_refs"] = ["entity_002"]
    same_object_cap["video1_analysis"]["information_atoms"][0]["fact"] = "The white BMW is parked near the curb."
    same_object_cap["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_002",
        "frame_keys": ["frame_0001"],
        "entity_refs": ["entity_002"],
        "fact": "The same white BMW begins moving forward."
    })
    same_object_cap["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_003",
        "frame_keys": ["frame_0001"],
        "entity_refs": ["entity_002"],
        "fact": "The same white BMW turns left."
    })
    same_object_cap["video2_analysis"]["information_atoms"][0]["entity_refs"] = ["entity_002"]
    same_object_cap["cross_modal_evidence_links"][0]["entity_id"] = "entity_002"
    same_object_cap["information_gain"][0]["entity_id"] = "entity_002"
    same_object_cap["reasoning_events"][0]["participating_entities"] = ["entity_002"]
    _validate_caption_schema(same_object_cap, {"frame_0001"}, "rgb", "event")
    print("Test 32 (Same-object valid graph with multiple atoms passes) passed")

    # Test 33: Multi-Entity Atom allowed
    multi_entity_cap = eval(repr(minimal_caption))
    multi_entity_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "the specific white BMW tracked across the sampled interval"
    })
    multi_entity_cap["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_010",
        "frame_keys": ["frame_0001"],
        "entity_refs": ["entity_001", "entity_002"],
        "fact": "The rider passes behind the turning white BMW."
    })
    multi_entity_cap["reasoning_events"].append({
        "event_id": "evt_002",
        "event_type": "interaction",
        "participating_entities": ["entity_001", "entity_002"],
        "supporting_atom_refs": ["v1_atom_010"],
        "description": "The rider and white BMW form one directly observed interaction."
    })
    _validate_caption_schema(multi_entity_cap, {"frame_0001"}, "rgb", "event")
    print("Test 33 (Multi-Entity Atom allowed) passed")

    # Test 34: Unknown Entity Ref
    unknown_entity_cap = eval(repr(minimal_caption))
    unknown_entity_cap["video1_analysis"]["information_atoms"][0]["entity_refs"] = ["entity_999"]
    assert_validation_fails(unknown_entity_cap, "references unknown entity: entity_999")
    print("Test 34 (Unknown Entity Ref rejection) passed")

    # Test 35: Cross-modal Entity mismatch
    cross_modal_mismatch = eval(repr(minimal_caption))
    cross_modal_mismatch["global_scene"]["physical_entities"].append({
        "entity_id": "entity_005",
        "category": "tree_shadow",
        "referential_scope": "the branching tree-shadow phenomenon on the road surface"
    })
    cross_modal_mismatch["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_005",
        "frame_keys": ["frame_0001"],
        "entity_refs": ["entity_005"],
        "fact": "Branching tree shadows extend across the road."
    })
    cross_modal_mismatch["video2_analysis"]["information_atoms"].append({
        "atom_id": "v2_atom_005",
        "frame_keys": ["frame_0001"],
        "entity_refs": ["entity_005"],
        "fact": "Branching tree-shadow boundaries remain visible."
    })
    cross_modal_mismatch["cross_modal_evidence_links"][0]["video1_evidence_refs"] = ["v1_atom_005"]
    cross_modal_mismatch["cross_modal_evidence_links"][0]["video2_evidence_refs"] = ["v2_atom_005"]
    assert_validation_fails(cross_modal_mismatch, "not explicitly connected to entity entity_001")
    print("Test 35 (Cross-modal Entity mismatch rejection) passed")

    # Test 36: Reasoning Event disconnected Entity
    disconnected_event_cap = eval(repr(minimal_caption))
    disconnected_event_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "the specific white BMW tracked across the sampled interval"
    })
    disconnected_event_cap["reasoning_events"][0]["participating_entities"] = ["entity_001", "entity_002"]
    assert_validation_fails(disconnected_event_cap, "participating_entities not covered")
    print("Test 36 (Reasoning Event disconnected Entity rejection) passed")

    # Test 37: Ambiguity target mismatch
    ambiguity_mismatch_cap = eval(repr(minimal_caption))
    ambiguity_mismatch_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "the specific white BMW tracked across the sampled interval"
    })
    ambiguity_mismatch_cap["ambiguity_events"].append({
        "ambiguity_id": "amb_001",
        "target_entity": "entity_002",
        "direction": "video1_resolves_video2",
        "ambiguous_video": "video2",
        "resolving_video": "video1",
        "low_confidence_observation": "Ambiguous evidence in video 2.",
        "why_ambiguous_video_cannot_resolve": "Cannot resolve due to lack of detail.",
        "candidate_hypotheses": [
            {"hypothesis": "hypothesis 1", "why_compatible_with_ambiguous": "Fits partial evidence.", "support_from_resolving": "Confirmed by video 1."},
            {"hypothesis": "hypothesis 2", "why_compatible_with_ambiguous": "Fits partial evidence.", "support_from_resolving": "Rejected by video 1."}
        ],
        "resolving_discriminative_evidence": "Clear evidence in video 1.",
        "eliminated_hypotheses": [{"hypothesis": "hypothesis 2", "why_eliminated": "Contradicted by video 1."}],
        "fusion_conclusion": "Conclusion after resolution.",
        "missing_attribute_type": "existence",
        "ambiguous_evidence_refs": ["v2_atom_001"],
        "resolving_evidence_refs": ["v1_atom_001"]
    })
    assert_validation_fails(ambiguity_mismatch_cap, "not explicitly connected to entity entity_002")
    print("Test 37 (Ambiguity target mismatch rejection) passed")

    # Test 38: Asymmetric modality coverage is allowed outside bilateral cross-modal sections
    asymmetric_entity_cap = eval(repr(minimal_caption))
    asymmetric_entity_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_005",
        "category": "tree_shadow",
        "referential_scope": "the branching tree-shadow phenomenon on the road surface"
    })
    asymmetric_entity_cap["video2_analysis"]["information_atoms"].append({
        "atom_id": "v2_atom_005",
        "frame_keys": ["frame_0001"],
        "entity_refs": ["entity_005"],
        "fact": "Branching tree-shadow boundaries remain visible."
    })
    _validate_caption_schema(asymmetric_entity_cap, {"frame_0001"}, "rgb", "event")
    print("Test 38 (Asymmetric modality coverage allowed) passed")

    def load_resume_items(*items):
        temp_file = Path("temp_resume_validator_compat.json")
        with open(temp_file, "w") as f:
            json.dump({"items": list(items)}, f)
        try:
            return _load_resume(temp_file)
        finally:
            temp_file.unlink()

    def current_resume_item(caption, **overrides):
        item = _task_to_item(task, status="generated", caption=caption)
        item.update(overrides)
        return item

    # Test 39: Current valid caption resumes
    valid_resume_item = current_resume_item(minimal_caption)
    valid_items, _ = load_resume_items(valid_resume_item)
    assert len(valid_items) == 1, "Current valid caption must be accepted by resume"
    print("Test 39 (Current valid caption resumes) passed")

    # Test 40: Old caption without entity_refs is rejected
    no_entity_refs_cap = eval(repr(minimal_caption))
    no_entity_refs_cap["video1_analysis"]["information_atoms"][0].pop("entity_refs")
    valid_items, _ = load_resume_items(current_resume_item(no_entity_refs_cap))
    assert len(valid_items) == 0, "Caption missing entity_refs must be rejected by resume"
    print("Test 40 (Resume rejects caption without entity_refs) passed")

    # Test 41: Old caption without referential_scope is rejected
    no_scope_cap = eval(repr(minimal_caption))
    no_scope_cap["global_scene"]["physical_entities"][0].pop("referential_scope")
    valid_items, _ = load_resume_items(current_resume_item(no_scope_cap))
    assert len(valid_items) == 0, "Caption missing referential_scope must be rejected by resume"
    print("Test 41 (Resume rejects caption without referential_scope) passed")

    # Test 42: Invalid current Entity-Atom connection is rejected
    invalid_connection_cap = eval(repr(minimal_caption))
    invalid_connection_cap["global_scene"]["physical_entities"].append({
        "entity_id": "entity_005",
        "category": "tree_shadow",
        "referential_scope": "the branching tree-shadow phenomenon on the road surface"
    })
    invalid_connection_cap["information_gain"][0]["entity_id"] = "entity_005"
    valid_items, _ = load_resume_items(current_resume_item(invalid_connection_cap))
    assert len(valid_items) == 0, "Invalid Entity-Atom connection must be rejected by resume"
    print("Test 42 (Resume rejects invalid Entity-Atom connection) passed")

    # Test 43: Missing resume metadata is rejected safely
    missing_metadata_item = {"caption_id": "missing_metadata", "caption": minimal_caption}
    valid_items, _ = load_resume_items(missing_metadata_item)
    assert len(valid_items) == 0, "Resume item missing metadata must be treated as stale"
    print("Test 43 (Resume rejects missing metadata safely) passed")

    # Test 44: Resume no longer filters skipped reasons containing additional_properties
    temp_file = Path("temp_resume_skipped_compat.json")
    skipped_item = {
        "caption_id": "failed_current_item",
        "reason": "Validation Error: additional_properties is not allowed here",
    }
    with open(temp_file, "w") as f:
        json.dump({"items": [], "skipped": [skipped_item]}, f)
    try:
        _, skipped_items = _load_resume(temp_file)
    finally:
        temp_file.unlink()
    assert skipped_items == [skipped_item], "Resume must preserve current additional_properties failures"
    print("Test 44 (Resume preserves additional_properties skipped reasons) passed")

    # Test 45: Limited run skipped accounting excludes unrelated global discovery failures
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_path = root / "input.json"
        dataset_root = root / "dataset"
        composite_root = root / "composite"
        input_payload = {
            "segments": {
                "aaa_missing": {
                    "split_dir": "missing_split",
                    "segment_name": "missing_segment",
                    "modality_pairs": [["rgb", "event"]],
                },
                "bbb_valid": {
                    "split_dir": "valid_split",
                    "segment_name": "valid_segment",
                    "modality_pairs": [["rgb", "event"]],
                },
            }
        }
        input_path.write_text(json.dumps(input_payload), encoding="utf-8")
        rgb_dir = dataset_root / ".frames_cache" / "valid_split" / "valid_segment" / "valid_day_rgb"
        event_dir = dataset_root / ".frames_cache_event" / "valid_split" / "valid_segment" / "valid_day_event"
        rgb_dir.mkdir(parents=True)
        event_dir.mkdir(parents=True)
        for frame_number in range(1, 31):
            (rgb_dir / f"frame_{frame_number:06d}.png").write_bytes(b"")
            (event_dir / f"frame_{frame_number:06d}.png").write_bytes(b"")

        tasks, scoped_skipped, total_selected = build_caption_tasks(
            input_path=input_path,
            dataset_root=dataset_root,
            composite_root=composite_root,
            sampling_strategy="uniform_adaptive",
            num_uniform_frames=1,
            num_adaptive_frames=0,
            existing_items=[],
            limit=1,
            write_composites=False,
        )
        assert len(tasks) == 1
        assert total_selected == 1
        assert scoped_skipped == [], f"Limited run should not include unrelated skipped items: {scoped_skipped}"

        _, full_skipped, _ = build_caption_tasks(
            input_path=input_path,
            dataset_root=dataset_root,
            composite_root=composite_root,
            sampling_strategy="uniform_adaptive",
            num_uniform_frames=1,
            num_adaptive_frames=0,
            existing_items=[],
            write_composites=False,
        )
        assert any(item.get("segment_id") == "aaa_missing" for item in full_skipped)
    print("Test 45 (Limited run skipped accounting scoped) passed")

    # Test 46: Empty reasoning graphs are allowed
    empty_reasoning_cap = eval(repr(minimal_caption))
    empty_reasoning_cap["cross_modal_evidence_links"] = []
    empty_reasoning_cap["information_gain"] = []
    empty_reasoning_cap["reasoning_events"] = []
    empty_reasoning_cap["ambiguity_events"] = []
    empty_reasoning_cap["qa_relevant_details"] = []
    try:
        validated, warn = _validate_caption_schema(empty_reasoning_cap, {"frame_0001"}, "rgb", "event")
        # Empty reasoning should pass
    except CaptionValidationError as e:
        assert False, f"Empty reasoning graphs must be allowed, but got error: {e}"
    print("Test 46 (Empty reasoning graphs allowed) passed")

    # Test 47: Resumed limited run filters existing skipped
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_path = root / "input.json"
        dataset_root = root / "dataset"
        composite_root = root / "composite"
        input_payload = {
            "segments": {
                "seg_selected": {
                    "split_dir": "valid_split",
                    "segment_name": "valid_segment",
                    "modality_pairs": [["rgb", "event"]],
                },
            }
        }
        input_path.write_text(json.dumps(input_payload), encoding="utf-8")
        rgb_dir = dataset_root / ".frames_cache" / "valid_split" / "valid_segment" / "valid_day_rgb"
        event_dir = dataset_root / ".frames_cache_event" / "valid_split" / "valid_segment" / "valid_day_event"
        rgb_dir.mkdir(parents=True)
        event_dir.mkdir(parents=True)
        for frame_number in range(1, 31):
            (rgb_dir / f"frame_{frame_number:06d}.png").write_bytes(b"")
            (event_dir / f"frame_{frame_number:06d}.png").write_bytes(b"")

        existing_skipped = [
            {"segment_id": "seg_selected", "reason": "Missing something"},
            {"segment_id": "seg_other", "reason": "Missing something else"},
        ]
        
        tasks, scoped_skipped, total_selected = build_caption_tasks(
            input_path=input_path,
            dataset_root=dataset_root,
            composite_root=composite_root,
            sampling_strategy="uniform_adaptive",
            num_uniform_frames=1,
            num_adaptive_frames=0,
            existing_items=[],
            existing_skipped=existing_skipped,
            limit=1,
            write_composites=False,
        )
        assert len(scoped_skipped) == 1, f"Expected 1, got {len(scoped_skipped)}: {scoped_skipped}"
        assert scoped_skipped[0]["segment_id"] == "seg_selected"
    print("Test 47 (Resumed limited run filters existing skipped) passed")

    # Test 48: information_gain[].video1_can_determine as a string -> rejected
    ig_cap = eval(repr(minimal_caption))
    ig_cap["information_gain"][0]["video1_can_determine"] = "Color"
    try:
        _validate_caption_schema(ig_cap, {"frame_0001"}, "rgb", "event")
        assert False, "Should have rejected video1_can_determine as string"
    except CaptionValidationError as e:
        assert "must be a list" in str(e).lower()
    print("Test 48 (information_gain string field rejected) passed")

    # Test 49: information_gain[].video1_can_determine as [] or list of strings -> accepted
    ig_cap["information_gain"][0]["video1_can_determine"] = []
    _validate_caption_schema(ig_cap, {"frame_0001"}, "rgb", "event")
    ig_cap["information_gain"][0]["video1_can_determine"] = ["Color"]
    _validate_caption_schema(ig_cap, {"frame_0001"}, "rgb", "event")
    print("Test 49 (information_gain list field accepted) passed")

    # Test 50: non-empty qa_relevant_details missing detail_id -> rejected
    qa_cap = eval(repr(minimal_caption))
    del qa_cap["qa_relevant_details"][0]["detail_id"]
    try:
        _validate_caption_schema(qa_cap, {"frame_0001"}, "rgb", "event")
        assert False, "Should have rejected QA detail missing detail_id"
    except CaptionValidationError as e:
        assert "detail_id" in str(e)
    print("Test 50 (QA missing detail_id rejected) passed")

    # Test 51: non-empty qa_relevant_details missing reasoning_pattern -> rejected
    qa_cap = eval(repr(minimal_caption))
    del qa_cap["qa_relevant_details"][0]["reasoning_pattern"]
    try:
        _validate_caption_schema(qa_cap, {"frame_0001"}, "rgb", "event")
        assert False, "Should have rejected QA detail missing reasoning_pattern"
    except CaptionValidationError as e:
        assert "reasoning_pattern" in str(e)
    print("Test 51 (QA missing reasoning_pattern rejected) passed")

    # Test 52: complete valid QA item -> accepted
    _validate_caption_schema(minimal_caption, {"frame_0001"}, "rgb", "event")
    print("Test 52 (Complete valid QA item accepted) passed")

    # Test 53: all optional reasoning lists empty -> accepted
    empty_reasoning_cap = eval(repr(minimal_caption))
    empty_reasoning_cap["cross_modal_evidence_links"] = []
    empty_reasoning_cap["information_gain"] = []
    empty_reasoning_cap["reasoning_events"] = []
    empty_reasoning_cap["ambiguity_events"] = []
    empty_reasoning_cap["qa_relevant_details"] = []
    _validate_caption_schema(empty_reasoning_cap, {"frame_0001"}, "rgb", "event")
    print("Test 53 (All optional reasoning lists empty accepted) passed")

    # Test 54: prompt contains non-empty item shape guidance while preserving empty canonical examples
    prompt_text = _build_caption_prompt(task)
    assert "OPTIONAL NON-EMPTY ITEM SHAPES:" in prompt_text
    assert '"video1_can_determine": [],' in prompt_text
    example = _build_prompt_schema_example(task)
    assert example["information_gain"] == []
    assert example["qa_relevant_details"] == []
    assert example["ambiguity_events"] == []
    print("Test 54 (Prompt shapes and empty examples) passed")

    # Test 55: section-specific retry hint for information_gain
    hint_ig = _build_validation_retry_hint(Exception("information_gain video1_can_determine must be a list"), "schema_validation_error")
    assert "fusion_additionally_reveals" in hint_ig
    print("Test 55 (Retry hint for information_gain) passed")

    # Test 56: section-specific retry hint for qa_relevant_details
    hint_qa = _build_validation_retry_hint(Exception("qa_relevant_details detail_id is missing"), "schema_validation_error")
    assert "reasoning_pattern" in hint_qa
    print("Test 56 (Retry hint for qa_relevant_details) passed")

    # Test 57: final failed generation preserves the last invalid raw response for debugging
    failed_exc = CaptionValidationError("failed")
    failed_exc.last_invalid_response = '{"bad": "json"}'
    item = _task_to_item(task, status="failed", reason="failed", attempts=6, first_attempt_success=False, final_error_category="schema_validation_error", last_invalid_response=getattr(failed_exc, "last_invalid_response", None))
    assert item.get("last_invalid_response") == '{"bad": "json"}'
    print("Test 57 (last_invalid_response preserved) passed")

    print("All tests passed!")

if __name__ == "__main__":
    run_tests()
