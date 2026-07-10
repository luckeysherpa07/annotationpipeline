"""Tests for Pass 1 (evidence construction) validation."""

import pytest
from typing import Any

from annotation_feature.aligned_caption_schema import CaptionValidationError
from annotation_feature.aligned_caption_pass1_validation import _validate_pass1_schema

def create_valid_pass1_schema() -> dict[str, Any]:
    return {
        "global_scene": {
            "scene_summary": "This is a fully compliant scene summary paragraph that describes the physical environment and ongoing actions in sufficient detail to exceed the minimum word count requirement of twenty words.",
            "environment": "urban",
            "temporal_progression": "The scene unfolds chronologically, demonstrating clear progression of events from start to finish without referencing any sensor or image quality artifacts.",
            "physical_entities": [
                {
                    "entity_id": "entity_001",
                    "category": "vehicle",
                    "referential_scope": "the specific white BMW tracked across the sampled interval"
                }
            ]
        },
        "video1_analysis": {
            "modality": "rgb",
            "detailed_caption": "The white BMW is stationary near the right curb. The white BMW is stationary near the right curb. The white BMW is stationary near the right curb. The white BMW is stationary near the right curb.",
            "information_atoms": [
                {
                    "atom_id": "v1_atom_001",
                    "frame_keys": ["frame_000000"],
                    "entity_refs": ["entity_001"],
                    "fact": "The white BMW is stationary near the right curb. The white BMW is stationary near the right curb. The white BMW is stationary near the right curb. The white BMW is stationary near the right curb."
                }
            ],
            "sensor_specific_cues": [],
            "sensor_limitations": [],
            "uncertain_observations": [],
            "missing_key_attributes": []
        },
        "video2_analysis": {
            "modality": "event",
            "detailed_caption": "The white BMW occupies a persistent curbside position. The white BMW occupies a persistent curbside position. The white BMW occupies a persistent curbside position. The white BMW occupies a persistent curbside position.",
            "information_atoms": [
                {
                    "atom_id": "v2_atom_001",
                    "frame_keys": ["frame_000000"],
                    "entity_refs": ["entity_001"],
                    "fact": "The white BMW occupies a persistent curbside position. The white BMW occupies a persistent curbside position. The white BMW occupies a persistent curbside position. The white BMW occupies a persistent curbside position."
                }
            ],
            "sensor_specific_cues": [],
            "sensor_limitations": [],
            "uncertain_observations": [],
            "missing_key_attributes": []
        }
    }

def test_valid_schema():
    parsed = create_valid_pass1_schema()
    # Should not raise
    _, warnings = _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")
    assert not warnings, f"Expected no warnings, got {warnings}"

def test_duplicate_entity_ids():
    parsed = create_valid_pass1_schema()
    # Add a duplicate entity ID
    parsed["global_scene"]["physical_entities"].append({
        "entity_id": "entity_001",
        "category": "rider",
        "referential_scope": "another rider"
    })
    
    with pytest.raises(CaptionValidationError, match="Duplicate entity_id: entity_001"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_duplicate_atom_ids():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["information_atoms"].append({
        "atom_id": "v1_atom_001",
        "frame_keys": ["frame_000000"],
        "entity_refs": ["entity_001"],
        "fact": "Duplicate atom fact."
    })
    
    with pytest.raises(CaptionValidationError, match="Duplicate evidence ID found: v1_atom_001"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_grounding_warnings():
    parsed = create_valid_pass1_schema()
    # Inject lots of ungrounded words into the detailed caption, meeting minimum 30 words
    parsed["video1_analysis"]["detailed_caption"] = "The white BMW is stationary near the right curb. And then a giant helicopter flies overhead firing missiles at the dinosaur that just emerged from the volcano eruption, causing massive damage and destruction."
    
    _, warnings = _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")
    assert len(warnings) > 0
    assert any("may contain ungrounded claims" in w for w in warnings)

def test_inappropriate_generic_missing_attribute_explanations():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "existence",
        "missing_attribute": "color",
        # Generic sensor theory explanation
        "why_missing": "the rgb sensor captures only specific wavelengths",
        "recoverable_evidence_refs": []
    })

    with pytest.raises(CaptionValidationError, match="contains generic sensor-theory wording"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_recoverable_evidence_refs_empty():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "existence",
        "missing_attribute": "color",
        "why_missing": "Physical occlusion by a tree.",
        "recoverable_evidence_refs": ["v2_atom_001"] # Invalid in pass 1!
    })
    with pytest.raises(CaptionValidationError, match="MUST be empty in Pass 1"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_no_downstream_fields_in_global_scene():
    parsed = create_valid_pass1_schema()
    parsed["global_scene"]["reasoning_focus_entities"] = ["entity_001"]
    with pytest.raises(CaptionValidationError, match="reasoning_focus_entities are forbidden"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_no_downstream_fields_at_top_level():
    parsed = create_valid_pass1_schema()
    parsed["cross_modal_evidence_links"] = []
    with pytest.raises(CaptionValidationError, match="contains unknown top-level fields"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_missing_global_scene():
    parsed = create_valid_pass1_schema()
    del parsed["global_scene"]
    with pytest.raises(CaptionValidationError, match="missing required Pass 1 field\\(s\\): global_scene"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_empty_atom_entity_refs():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["information_atoms"][0]["entity_refs"] = []
    with pytest.raises(CaptionValidationError, match="entity_refs cannot be empty"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_invalid_frame_key():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["information_atoms"][0]["frame_keys"] = ["frame_000001"]
    with pytest.raises(CaptionValidationError, match="Unknown frame_key"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_unknown_entity_ref():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["information_atoms"][0]["entity_refs"] = ["entity_999"]
    with pytest.raises(CaptionValidationError, match="references unknown entity"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_sensor_specific_cues_as_string_fails():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["sensor_specific_cues"] = "This should be a list, not a string."
    with pytest.raises(CaptionValidationError, match="must be a list"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_as_object_fails():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"] = {"observed_evidence": "evidence"}
    with pytest.raises(CaptionValidationError, match="must be a list"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_wrong_video1_atom_prefix():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["information_atoms"][0]["atom_id"] = "atom_001"
    with pytest.raises(CaptionValidationError, match="must start with v1_atom_"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_wrong_video2_atom_prefix():
    parsed = create_valid_pass1_schema()
    parsed["video2_analysis"]["information_atoms"][0]["atom_id"] = "v1_atom_002"
    with pytest.raises(CaptionValidationError, match="must start with v2_atom_"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_one_hypothesis():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see license plate",
        "hypotheses": [
            {"hypothesis": "It is a BMW", "confidence": "low"}
        ]
    })
    with pytest.raises(CaptionValidationError, match="at least 2 distinct valid candidate hypotheses"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_duplicate_hypotheses():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see license plate",
        "hypotheses": [
            {"hypothesis": "It is a BMW", "confidence": "low"},
            {"hypothesis": "it is a bmw", "confidence": "medium"}
        ]
    })
    with pytest.raises(CaptionValidationError, match="at least 2 distinct valid candidate hypotheses"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_meta_hypothesis():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see license plate",
        "hypotheses": [
            {"hypothesis": "It is a BMW", "confidence": "low"},
            {"hypothesis": "Cannot be determined", "confidence": "low"}
        ]
    })
    with pytest.raises(CaptionValidationError, match="contains meta-statement of inability"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_valid_hypotheses():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": [
            {"hypothesis": "It is a BMW", "confidence": "low"},
            {"hypothesis": "It is an Audi", "confidence": "low"}
        ]
    })
    # Should not raise
    _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_missing_attribute_modality_does_not_record():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "existence",
        "missing_attribute": "color",
        "why_missing": "the sensing modality does not record color information.",
        "recoverable_evidence_refs": []
    })
    with pytest.raises(CaptionValidationError, match="must be segment-specific, not generic sensor theory"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_video1_analysis_as_list():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"] = []
    with pytest.raises(CaptionValidationError, match="video1_analysis must be an object"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_video2_analysis_as_string():
    parsed = create_valid_pass1_schema()
    parsed["video2_analysis"] = "invalid"
    with pytest.raises(CaptionValidationError, match="video2_analysis must be an object"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_prompt_schema_example_neutral_identity():
    # Focused regression test to ensure we don't leak "white BMW" in the schema example
    from annotation_feature.aligned_caption_pass1_prompt import _build_prompt_schema_example
    
    class DummyTask:
        modality1 = "rgb"
        modality2 = "event"
        class DummyPath:
            stem = "frame_000000"
            name = "frame_000000.jpg"
        composite_frames = [DummyPath()]
    
    example = _build_prompt_schema_example(DummyTask())
    
    import json
    example_str = json.dumps(example)
    assert "white BMW" not in example_str, "Schema example must not encode rich identity like 'white BMW'"
    assert "BMW" not in example_str, "Schema example must not encode rich identity like 'BMW'"
    assert "concrete barrier" in example_str, "Schema example should use neutral identity"


class DummyPath:
    def __init__(self, stem):
        self.stem = stem
        self.name = f"{stem}.jpg"

class DummyTask:
    caption_id = "test_task"
    modality1 = "rgb"
    modality2 = "event"
    segment_id = "test_seg"
    side = "left"
    composite_frames = [DummyPath("frame_000000")]
    frames1 = [DummyPath("frame_000000")]
    frames2 = [DummyPath("frame_000000")]

def test_diagnostics_success():
    import json
    import unittest.mock as mock
    import asyncio
    from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _call_gemini_pass1
    
    mock_client = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.text = json.dumps(create_valid_pass1_schema())
    mock_client.models.generate_content.return_value = mock_response
    
    api_stats = [0]
    
    with mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._ensure_composite_frames") as mock_ensure, \
         mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._encode_images", return_value=["dGVzdA=="]):
        evidence, warnings, diagnostics = asyncio.run(_call_gemini_pass1(
            client=mock_client,
            task=DummyTask(),
            model_name="gemini-3.5-flash",
            max_retries=3,
            max_transport_retries=3,
            api_stats=api_stats
        ))
    
    assert diagnostics["api_calls"] == 1
    assert diagnostics["validation_attempts"] == 1
    assert diagnostics["transport_retries"] == 0
    assert diagnostics["first_validation_attempt_success"] is True
    assert len(diagnostics["retry_history"]) == 0

def test_diagnostics_transport_failure():
    import json
    import unittest.mock as mock
    import asyncio
    from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _call_gemini_pass1
    
    mock_client = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.text = json.dumps(create_valid_pass1_schema())
    
    mock_client.models.generate_content.side_effect = [
        RuntimeError("connection error"),
        mock_response
    ]
    
    api_stats = [0]
    with mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._ensure_composite_frames") as mock_ensure, \
         mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._encode_images", return_value=["dGVzdA=="]):
        evidence, warnings, diagnostics = asyncio.run(_call_gemini_pass1(
            client=mock_client,
            task=DummyTask(),
            model_name="gemini-3.5-flash",
            max_retries=3,
            max_transport_retries=3,
            api_stats=api_stats
        ))
    
    assert diagnostics["api_calls"] == 2
    assert diagnostics["validation_attempts"] == 1
    assert diagnostics["transport_retries"] == 1
    assert diagnostics["first_validation_attempt_success"] is True
    assert len(diagnostics["retry_history"]) == 1
    assert diagnostics["retry_history"][0]["type"] == "transport"

def test_diagnostics_validation_failure():
    import json
    import unittest.mock as mock
    import asyncio
    from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _call_gemini_pass1
    
    mock_client = mock.MagicMock()
    mock_response_invalid = mock.MagicMock()
    mock_response_invalid.text = json.dumps({"video1_analysis": []})
    
    mock_response_valid = mock.MagicMock()
    mock_response_valid.text = json.dumps(create_valid_pass1_schema())
    
    mock_client.models.generate_content.side_effect = [
        mock_response_invalid,
        mock_response_valid
    ]
    
    api_stats = [0]
    with mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._ensure_composite_frames") as mock_ensure, \
         mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._encode_images", return_value=["dGVzdA=="]):
        evidence, warnings, diagnostics = asyncio.run(_call_gemini_pass1(
            client=mock_client,
            task=DummyTask(),
            model_name="gemini-3.5-flash",
            max_retries=3,
            max_transport_retries=3,
            api_stats=api_stats
        ))
    
    assert diagnostics["api_calls"] == 2
    assert diagnostics["validation_attempts"] == 2
    assert diagnostics["transport_retries"] == 0
    assert diagnostics["first_validation_attempt_success"] is False
    assert len(diagnostics["retry_history"]) == 1
    assert diagnostics["retry_history"][0]["type"] == "validation"

def test_diagnostics_mixed_failures():
    import json
    import unittest.mock as mock
    import asyncio
    from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _call_gemini_pass1
    
    mock_client = mock.MagicMock()
    
    mock_response_invalid = mock.MagicMock()
    mock_response_invalid.text = json.dumps({"video1_analysis": []})
    
    mock_response_valid = mock.MagicMock()
    mock_response_valid.text = json.dumps(create_valid_pass1_schema())
    
    mock_client.models.generate_content.side_effect = [
        RuntimeError("connection error"),
        mock_response_invalid,
        mock_response_valid
    ]
    
    api_stats = [0]
    with mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._ensure_composite_frames") as mock_ensure, \
         mock.patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._encode_images", return_value=["dGVzdA=="]):
        evidence, warnings, diagnostics = asyncio.run(_call_gemini_pass1(
            client=mock_client,
            task=DummyTask(),
            model_name="gemini-3.5-flash",
            max_retries=3,
            max_transport_retries=3,
            api_stats=api_stats
        ))
    
    assert diagnostics["api_calls"] == 3
    assert diagnostics["validation_attempts"] == 2
    assert diagnostics["transport_retries"] == 1
    assert diagnostics["first_validation_attempt_success"] is False
    assert len(diagnostics["retry_history"]) == 2
    assert diagnostics["retry_history"][0]["type"] == "transport"
    assert diagnostics["retry_history"][1]["type"] == "validation"

def test_load_resume_diagnostics_compatibility(tmp_path):
    import json
    from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _load_resume_pass1
    
    output_file = tmp_path / "resume.json"
    historical_data = {
        "metadata": {},
        "items": [
            {
                "caption_id": "test_task",
                "segment_id": "ride_a_bike_split/Seg1",
                "side": "left",
                "modality1": "rgb",
                "modality2": "event",
                "composite_frames": ["frame_000000.jpg"],
                "evidence": create_valid_pass1_schema(),
                "attempts": 2,
                "first_attempt_success": False
            }
        ],
        "skipped": []
    }
    with open(output_file, "w") as f:
        json.dump(historical_data, f)
        
    items, skipped = _load_resume_pass1(output_file)
    assert len(items) == 1
    assert items[0]["caption_id"] == "test_task"

def test_why_missing_generic_process_1():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "existence",
        "missing_attribute": "color",
        "why_missing": "The sensing process records structural outlines and changes in intensity, which do not preserve static color information.",
        "recoverable_evidence_refs": []
    })
    with pytest.raises(CaptionValidationError, match="must be segment-specific, not generic sensor theory"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_why_missing_generic_process_2():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "existence",
        "missing_attribute": "color",
        "why_missing": "The sensor captures changes rather than static color.",
        "recoverable_evidence_refs": []
    })
    with pytest.raises(CaptionValidationError, match="must be segment-specific, not generic sensor theory"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_why_missing_segment_specific_1():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "existence",
        "missing_attribute": "color",
        "why_missing": "The supplied Video 2 observations do not independently establish the parked vehicles' paint colors.",
        "recoverable_evidence_refs": []
    })
    # Should not raise
    _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_why_missing_segment_specific_2():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["missing_key_attributes"].append({
        "attribute_type": "existence",
        "missing_attribute": "color",
        "why_missing": "The parked sedan's paint color is not resolved in the supplied Video 2 observations.",
        "recoverable_evidence_refs": []
    })
    # Should not raise
    _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_hypothesis_normalization_duplicates_punctuation():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry vehicle",
        "missing_evidence": "Brand details",
        "hypotheses": [
            {"hypothesis": "The vehicle may be a sedan.", "confidence": "low"},
            {"hypothesis": "the vehicle may be a sedan", "confidence": "low"}
        ]
    })
    with pytest.raises(CaptionValidationError, match="must contain at least 2 distinct valid candidate hypotheses"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_hypothesis_normalization_duplicates_whitespace():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry vehicle",
        "missing_evidence": "Brand details",
        "hypotheses": [
            {"hypothesis": "The vehicle may be a sedan!", "confidence": "low"},
            {"hypothesis": "  the   vehicle may be a sedan  ", "confidence": "low"}
        ]
    })
    with pytest.raises(CaptionValidationError, match="must contain at least 2 distinct valid candidate hypotheses"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_hypothesis_normalization_distinct():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry vehicle",
        "missing_evidence": "Brand details",
        "hypotheses": [
            {"hypothesis": "The outlined vehicle may be a sedan.", "confidence": "low"},
            {"hypothesis": "The outlined vehicle may be a hatchback.", "confidence": "low"}
        ]
    })
    # Should not raise
    _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")


def test_pass1_prompt_regression():
    from annotation_feature.aligned_caption_pass1_prompt import _build_pass1_prompt, _build_prompt_schema_example
    
    class DummyPath:
        stem = "frame_000000"
        name = "frame_000000.jpg"
        
    class DummyTask:
        caption_id = "test_task"
        modality1 = "rgb"
        modality2 = "event"
        segment_id = "test_seg"
        side = "left"
        composite_frames = [DummyPath(), DummyPath()]
        frames1 = [DummyPath()]
        frames2 = [DummyPath()]

    task = DummyTask()
    prompt = _build_pass1_prompt(task)
    
    # Must exist
    assert "PASS 1" in prompt
    assert "PRIORITIES:" in prompt
    assert "Neither source is ground truth." in prompt
    assert "ISOLATION & LEAKAGE:" in prompt
    assert "REGISTRY METADATA:" in prompt
    assert "MUST remain [] in PASS 1." in prompt
    assert "recoverability reasoning" in prompt
    
    # Must preserve modality block
    assert "MODALITY CAPABILITY CONSTRAINTS" in prompt
    
    # Must NOT exist (old sections/residues)
    assert "GRAPH CONSTRUCTION AND EVIDENCE-CLOSURE WORKFLOW" not in prompt
    assert "TARGETED QUALITY RULES" not in prompt
    assert "FINAL SELF-CHECK BEFORE OUTPUT" not in prompt
    assert "Build a dense modality-local evidence representation" not in prompt

    # Verify example structure validity
    example = _build_prompt_schema_example(task)
    from annotation_feature.aligned_caption_pass1_validation import _validate_pass1_schema
    # Should validate example output structure successfully
    _validate_pass1_schema(example, {"frame_000000"}, "rgb", "event")

def test_build_modality_constraint_block_contains_differentiation():
    from annotation_feature.aligned_caption_schema import build_modality_constraint_block

    # rgb + event
    block = build_modality_constraint_block("rgb", "event")
    assert "CROSS-MODAL DIFFERENTIATION GUIDE" in block
    assert "rgb-EXCLUSIVE" in block
    assert "event-EXCLUSIVE" in block
    assert "SHARED atoms" in block

    # ir + depth（验证其他 pair 同样工作）
    block2 = build_modality_constraint_block("ir", "depth")
    assert "CROSS-MODAL DIFFERENTIATION GUIDE" in block2
    assert "ir-EXCLUSIVE" in block2
    assert "depth-EXCLUSIVE" in block2

    # 未知 modality（验证 fallback 不崩溃）
    block3 = build_modality_constraint_block("unknown_mod", "event")
    assert "CROSS-MODAL DIFFERENTIATION GUIDE" in block3

def test_grounding_exempt_words_not_flagged():
    from annotation_feature.aligned_caption_pass1_validation import _validate_pass1_schema
    example = create_valid_pass1_schema()
    
    # 构造一个带有多个豁免词但本质上 grounded 的详细描述
    # 这些词如果不在 atom 中，以前会触发 soft warning (因为长度 >= 5)
    example["video1_analysis"]["detailed_caption"] = "Initially, the vehicle advances along the broad street, passing through several clear regions, including multiple bright segments, before finally progressing beyond the short wall. " * 3
    
    # Atoms 只包含核心物理名词
    example["video1_analysis"]["information_atoms"] = [
        {
            "atom_id": "v1_atom_1",
            "frame_keys": ["frame_000000"],
            "entity_refs": ["entity_001"],
            "fact": "The vehicle moves on the street near a wall." # 不包含 initially, advances, passing, broad, bright 等
        }
    ]
    
    _, warnings = _validate_pass1_schema(example, {"frame_000000"}, "rgb", "event")
    
    # 我们期望不包含 "Words not in atoms:" 这样的 warning，因为所有缺失的词都在豁免列表里
    for w in warnings:
        assert "Words not in atoms:" not in w

def test_why_missing_physical_instrument_pattern():
    from annotation_feature.aligned_caption_pass1_validation import _validate_pass1_why_missing
    from annotation_feature.aligned_caption_schema import CaptionValidationError
    import pytest
    
    with pytest.raises(CaptionValidationError, match="generic sensor theory"):
        _validate_pass1_why_missing("The physical instrument does not register surface paint reflectance.", "missing_key_attributes[0].why_missing")

def test_why_missing_does_not_detect_static():
    from annotation_feature.aligned_caption_pass1_validation import _validate_pass1_why_missing
    from annotation_feature.aligned_caption_schema import CaptionValidationError
    import pytest
    
    with pytest.raises(CaptionValidationError, match="generic sensor theory"):
        _validate_pass1_why_missing("It does not record absolute intensity differences without motion.", "missing_key_attributes[0].why_missing")
        
    with pytest.raises(CaptionValidationError, match="generic sensor theory"):
        _validate_pass1_why_missing("It is not possible to distinguish the car without color information.", "missing_key_attributes[0].why_missing")


def test_uncertain_observations_empty_hypotheses_passes():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_missing_id():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="must be a non-empty string"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_bad_prefix():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "bad_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="must start with v1_unc_"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_duplicate_id():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car 1",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car 2",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="Duplicate evidence ID found: v1_unc_001"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_missing_entity_id():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="must be a non-empty string"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_unknown_entity():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_002",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="references unknown entity: entity_002"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_empty_evidence_refs():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": [],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="evidence_refs cannot be empty"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_unknown_atom():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_999"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="references unknown atom: v1_atom_999"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_video1_ref_video2_atom():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v2_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="evidence_refs v2_atom_001 must start with v1_atom_"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_video2_ref_video1_atom():
    parsed = create_valid_pass1_schema()
    parsed["video2_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v2_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="evidence_refs v1_atom_001 must start with v2_atom_"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_atom_not_connected_to_entity():
    parsed = create_valid_pass1_schema()
    parsed["global_scene"]["physical_entities"].append({
        "entity_id": "entity_002",
        "category": "vehicle",
        "referential_scope": "another car"
    })
    # Create an atom connected to entity_001
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_002",  # Not connected to v1_atom_001
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="is not connected to entity entity_002"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_duplicate_evidence_refs():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001", "v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": []
    })
    with pytest.raises(CaptionValidationError, match="evidence_refs contains duplicate: v1_atom_001"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")

def test_uncertain_observations_bad_confidence():
    parsed = create_valid_pass1_schema()
    parsed["video1_analysis"]["uncertain_observations"].append({
        "uncertainty_id": "v1_unc_001",
        "entity_id": "entity_001",
        "evidence_refs": ["v1_atom_001"],
        "observed_evidence": "Blurry car",
        "missing_evidence": "Cannot see brand",
        "hypotheses": [
            {"hypothesis": "BMW", "confidence": "low"},
            {"hypothesis": "Audi", "confidence": "super_high"}
        ]
    })
    with pytest.raises(CaptionValidationError, match="confidence must be high, medium, or low"):
        _validate_pass1_schema(parsed, {"frame_000000"}, "rgb", "event")
