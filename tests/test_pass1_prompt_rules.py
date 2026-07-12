import pytest
from annotation_feature.aligned_caption_schema import MODALITY_EXCLUSIVE_CUES
from annotation_feature.aligned_caption_pass1_prompt import build_pass1_system_prompt, _build_prompt_schema_example

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
    class DummyTask:
        composite_frames = []
        modality1 = "rgb"
        modality2 = "event"
        class CompositeFramePath:
            stem = "frame_000000"
        def __init__(self):
            self.composite_frames = [self.CompositeFramePath()]
            
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
