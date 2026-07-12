import pytest
from annotation_feature.aligned_caption_schema import MODALITY_EXCLUSIVE_CUES
from annotation_feature.aligned_caption_pass1_prompt import (
    _build_prompt_schema_example,
    build_pass1_system_prompt,
    build_pass1_user_prompt,
)


class DummyTask:
    modality1 = "rgb"
    modality2 = "event"
    segment_id = "segment_16_21"
    side = "left"

    class CompositeFramePath:
        stem = "frame_000000"
        name = "frame_000000.png"

    def __init__(self):
        self.composite_frames = [self.CompositeFramePath()]

def test_event_guide_no_mechanism_language():
    event_cues = MODALITY_EXCLUSIVE_CUES["event"]
    event_text = event_cues["exclusive"].lower() + " " + event_cues["not_visible"].lower()
    forbidden_terms = ["activation", "response", "edge-onset", "zero-activation", "polarity"]
    for term in forbidden_terms:
        assert term not in event_text, f"Event guide contains forbidden term: {term}"

def test_ir_guide_no_mechanism_language():
    ir_cues = MODALITY_EXCLUSIVE_CUES["ir"]
    ir_text = ir_cues["exclusive"].lower() + " " + ir_cues["not_visible"].lower()
    forbidden_terms = ["heat-map", "thermal pixels", "ir response"]
    for term in forbidden_terms:
        assert term not in ir_text, f"IR guide contains forbidden term: {term}"

def test_depth_guide_no_mechanism_language():
    depth_cues = MODALITY_EXCLUSIVE_CUES["depth"]
    depth_text = depth_cues["exclusive"].lower() + " " + depth_cues["not_visible"].lower()
    forbidden_terms = ["depth-map intensity"]
    for term in forbidden_terms:
        assert term not in depth_text, f"Depth guide contains forbidden term: {term}"

def test_prompt_contains_physical_world_tests():
    prompt = build_pass1_system_prompt()
    assert "PHYSICAL-WORLD SUBJECT TEST" in prompt
    assert "MODALITY-HIDDEN TEST" in prompt
    assert "INTERNAL CONVERSION ORDER" in prompt

def test_schema_example_conforms_to_rules():
    example = _build_prompt_schema_example(DummyTask())
    
    forbidden_subjects = ["data", "signal", "response", "activation", "output", "representation", "sensor", "map", "pixels"]
    
    for v_key in ["video1_analysis", "video2_analysis"]:
        atoms = example[v_key]["information_atoms"]
        for atom in atoms:
            fact = atom["fact"].lower()
            for subject in forbidden_subjects:
                assert not fact.startswith(f"the {subject}"), f"Fact should not use forbidden subject: {fact}"
                assert not fact.startswith(f"a {subject}"), f"Fact should not use forbidden subject: {fact}"

def test_sensor_specific_cues_allow_mechanism_language():
    prompt = build_pass1_system_prompt()
    assert "sensor_specific_cues" in prompt
    # Verifies it's still specifically permitted.
    assert "is the ONLY place where mechanism-oriented response patterns are permitted" in prompt


def test_prompt_contains_uncertainty_consistency_policy():
    prompt = build_pass1_system_prompt()
    assert "MUST NOT contradict a same-source Atom" in prompt
    assert "uncertainty may concern only finer compatible categories" in prompt
    assert "either remove the uncertainty or weaken the Atom" in prompt


def test_prompt_contains_actual_evidence_missing_target_policy():
    prompt = build_pass1_system_prompt()
    assert "modality capability alone is never enough" in prompt
    assert "coarse shape is not make/model evidence" in prompt
    assert "outer outline is not license-plate-text evidence" in prompt
    assert "Empty missing lists are preferable to unrecoverable targets" in prompt


def test_prompt_has_one_ordered_evidence_workflow():
    prompt = build_pass1_system_prompt()
    steps = [
        "1. Inspect Video 1 independently",
        "2. Inspect Video 2 independently",
        "3. Build and reconcile one conservative physical Entity Registry",
        "4. Create minimal source-local `information_atoms`",
        "5. Check every source-local field",
        "6. Write each `detailed_caption`",
        "7. Add only genuine unresolved source-local uncertainty",
        "8. Add a missing attribute only when",
        "9. Run the final schema, reference, grounding, wording, and scope checks",
    ]
    positions = [prompt.index(step) for step in steps]
    assert positions == sorted(positions)
    assert prompt.count("EVIDENCE-CONSTRUCTION WORKFLOW") == 1


def test_prompt_allows_shared_facts_without_forcing_differentiation():
    system_prompt = build_pass1_system_prompt()
    user_prompt = build_pass1_user_prompt(DummyTask())
    assert "Shared coarse physical facts may legitimately be supported by both modalities" in system_prompt
    assert "never manufacture modality-exclusive atoms" in system_prompt
    for prompt in (system_prompt, user_prompt):
        assert "CROSS-MODAL DIFFERENTIATION GUIDE" not in prompt
        assert "EXCLUSIVE atoms should capture" not in prompt
        assert "one of them is wrong" not in prompt
        assert "MUST each contain at least one atom whose fact is NOT shared" not in prompt


def test_prompt_example_prefers_empty_optional_lists():
    example = _build_prompt_schema_example(DummyTask())
    for analysis_key in ("video1_analysis", "video2_analysis"):
        assert example[analysis_key]["uncertain_observations"] == []
        assert example[analysis_key]["missing_key_attributes"] == []


def test_pass1_prompt_does_not_reintroduce_downstream_fields():
    prompt = build_pass1_system_prompt()
    example = _build_prompt_schema_example(DummyTask())
    forbidden_fields = {
        "cross_modal_evidence_links",
        "information_gain",
        "reasoning_events",
        "ambiguity_events",
        "qa_pairs",
    }
    assert forbidden_fields.isdisjoint(example)
    for field in forbidden_fields:
        assert f"`{field}`" not in prompt


def test_generated_user_prompt_preserves_dynamic_contract():
    prompt = build_pass1_user_prompt(DummyTask())
    required_text = [
        "ACTIVE MODALITY GUIDANCE: RGB",
        "ACTIVE MODALITY GUIDANCE: EVENT",
        "ALLOWED ENUMS:",
        "attribute_type",
        '"global_scene"',
        '"video1_analysis"',
        '"video2_analysis"',
        "Segment: segment_16_21; side: left.",
        "frame_000000.png",
    ]
    for text in required_text:
        assert text in prompt


def test_refactored_system_prompt_stays_materially_shorter():
    prompt = build_pass1_system_prompt()
    assert len(prompt) < 10_000
    assert len(prompt.split()) < 1_400


def test_event_guidance_requires_physical_world_conversion():
    prompt = build_pass1_user_prompt(DummyTask())

    assert "line, contour, silhouette, outline, boundary, or edge transition" in prompt
    assert "sensor_specific_cues" in prompt
    assert "Several parked vehicles remain distinguishable along the roadside." in prompt
    assert "Never borrow an object identity from RGB" in prompt


def test_event_guidance_removes_outline_as_preferred_fact():
    prompt = build_pass1_user_prompt(DummyTask())

    assert "The vehicle outline remains distinguishable." not in prompt


def test_event_guidance_contains_invalid_representation_examples():
    prompt = build_pass1_user_prompt(DummyTask())

    assert "The vehicle silhouette is traced by sharp boundaries." in prompt
    assert "Edge transitions define the building facade." in prompt
    assert "The wall is reduced to a single vertical line." in prompt


def test_schema_example_event_atoms_use_physical_world_facts():
    example = _build_prompt_schema_example(DummyTask())
    event_text = " ".join(
        atom["fact"]
        for atom in example["video2_analysis"]["information_atoms"]
    ).lower()

    forbidden_phrases = [
        "traced by",
        "defined by edge",
        "reduced to a single",
        "silhouette",
        "edge transition",
        "structurally distinguishable",
        "straight upper boundary",
    ]

    for phrase in forbidden_phrases:
        assert phrase not in event_text
