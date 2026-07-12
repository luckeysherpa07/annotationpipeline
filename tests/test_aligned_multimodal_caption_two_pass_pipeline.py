import pytest
import asyncio
from unittest.mock import MagicMock
from annotation_feature.pipeline.client import GeminiClientProvider, GeminiKeysExhaustedError
from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _call_gemini_pass1, CaptionTask
import sys
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

# Mock out _encode_images and build_image_parts to prevent trying to read real files or use the real google.genai sdk
import annotation_feature.aligned_multimodal_caption_two_pass_pipeline
annotation_feature.aligned_multimodal_caption_two_pass_pipeline._encode_images = MagicMock(return_value=["mock_encoded_image"])
sys.modules['annotation_feature.pipeline.utils'] = MagicMock()
sys.modules['annotation_feature.pipeline.utils'].build_image_parts = MagicMock(return_value=[])

def test_key_rotation_preserves_state():
    class MockClientProvider(GeminiClientProvider):
        def __init__(self):
            self._client = MagicMock()
            self._client.models.generate_content.side_effect = [
                RuntimeError("quota exceeded for key 1"),
                MagicMock(text='{"global_scene": {}, "video1_analysis": {"modality": "video"}, "video2_analysis": {"modality": "lidar"}}')
            ]
            self.rotations = 0

        def get_client(self):
            return self._client

        def rotate_client(self):
            self.rotations += 1
            return self._client

    provider = MockClientProvider()
    from pathlib import Path
    from pathlib import Path
    task = CaptionTask(
        caption_id="test_cap",
        segment_id="seg",
        split_dir="test",
        segment_name="seg",
        side="left",
        modality1="rgb",
        modality2="depth",
        frame_dir1=Path("a"),
        frame_dir2=Path("b"),
        frames1=(),
        frames2=(),
        composite_frames=(Path("f1.png"),),
        sampling_strategy="uniform",
        uniform_anchor_indexes=(),
        adaptive_frame_indexes=(),
        selected_frame_indexes=(),
        candidate_frame_indexes=(),
        selection_config_fingerprint="fp"
    )
    # Needs to fail gracefully because of validation errors since the mock text lacks all required valid fields, 
    # but the key thing is we rotate and keep attempt state.
    from annotation_feature.aligned_caption_pass1_validation import Pass1StructuralValidationError
    
    with pytest.raises(Exception) as exc:
        asyncio.run(_call_gemini_pass1(
            client_provider=provider,
            task=task,
            model_name="test",
            max_retries=1,
            max_transport_retries=1,
            api_stats=[0]
        ))
    
    # Provider was rotated once
    assert provider.rotations == 1
    
    # The actual exception raised at the end of max_retries should have diagnostics
    assert hasattr(exc.value, "diagnostics")
    diag = exc.value.diagnostics
    assert diag["api_calls"] == 2 # 1 for quota, 1 for success but validation failed
    assert diag["quota_failures"] == 1
    assert diag["key_rotations"] == 1
    assert diag["validation_attempts"] == 1 # Only 1 successful response parsed

def test_keys_exhausted():
    class ExhaustingProvider(GeminiClientProvider):
        def __init__(self):
            self._client = MagicMock()
            self._client.models.generate_content.side_effect = RuntimeError("quota limit 429")

        def get_client(self):
            return self._client

        def rotate_client(self):
            raise GeminiKeysExhaustedError("Exhausted")

    provider = ExhaustingProvider()
    from pathlib import Path
    from pathlib import Path
    task = CaptionTask(
        caption_id="test_cap",
        segment_id="seg",
        split_dir="test",
        segment_name="seg",
        side="left",
        modality1="rgb",
        modality2="depth",
        frame_dir1=Path("a"),
        frame_dir2=Path("b"),
        frames1=(),
        frames2=(),
        composite_frames=(Path("f1.png"),),
        sampling_strategy="uniform",
        uniform_anchor_indexes=(),
        adaptive_frame_indexes=(),
        selected_frame_indexes=(),
        candidate_frame_indexes=(),
        selection_config_fingerprint="fp"
    )
    with pytest.raises(GeminiKeysExhaustedError) as exc:
        asyncio.run(_call_gemini_pass1(
            client_provider=provider,
            task=task,
            model_name="test",
            max_retries=1,
            max_transport_retries=1,
            api_stats=[0]
        ))
        
    assert "Exhausted" in str(exc.value)

from annotation_feature.pipeline.client import Pass1TransportError
from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import CaptionParseError

def test_transport_error_preserves_diagnostics():
    class TransportFailingProvider(GeminiClientProvider):
        def __init__(self):
            self._client = MagicMock()
            self._client.models.generate_content.side_effect = Exception("503 Service Unavailable") # Some arbitrary exception not matching quota

        def get_client(self):
            return self._client

        def rotate_client(self):
            return self._client

    provider = TransportFailingProvider()
    from pathlib import Path
    from pathlib import Path
    task = CaptionTask(
        caption_id="test_cap",
        segment_id="seg",
        split_dir="test",
        segment_name="seg",
        side="left",
        modality1="rgb",
        modality2="depth",
        frame_dir1=Path("a"),
        frame_dir2=Path("b"),
        frames1=(),
        frames2=(),
        composite_frames=(Path("f1.png"),),
        sampling_strategy="uniform",
        uniform_anchor_indexes=(),
        adaptive_frame_indexes=(),
        selected_frame_indexes=(),
        candidate_frame_indexes=(),
        selection_config_fingerprint="fp"
    )
    
    with pytest.raises(Pass1TransportError) as exc:
        asyncio.run(_call_gemini_pass1(
            client_provider=provider,
            task=task,
            model_name="test",
            max_retries=1,
            max_transport_retries=2,
            api_stats=[0]
        ))
    
    assert exc.value.diagnostics["transport_retries"] == 2
    assert exc.value.diagnostics["api_calls"] == 2
    assert exc.value.diagnostics["first_validation_attempt_success"] is None
    assert exc.value.last_invalid_response is None

def test_first_attempt_success_state():
    class MockProvider(GeminiClientProvider):
        def __init__(self, response_texts):
            self._client = MagicMock()
            self._client.models.generate_content.side_effect = [MagicMock(text=rt) for rt in response_texts]

        def get_client(self):
            return self._client

        def rotate_client(self):
            return self._client

    from pathlib import Path
    task = CaptionTask(
        caption_id="test_cap",
        segment_id="seg",
        split_dir="test",
        segment_name="seg",
        side="left",
        modality1="rgb",
        modality2="depth",
        frame_dir1=Path("a"),
        frame_dir2=Path("b"),
        frames1=(),
        frames2=(),
        composite_frames=(Path("f1.png"),),
        sampling_strategy="uniform",
        uniform_anchor_indexes=(),
        adaptive_frame_indexes=(),
        selected_frame_indexes=(),
        candidate_frame_indexes=(),
        selection_config_fingerprint="fp"
    )

    # 1. First validation parse failure -> False
    provider_parse_fail = MockProvider(["invalid json"])
    with pytest.raises(CaptionParseError) as exc:
        asyncio.run(_call_gemini_pass1(
            client_provider=provider_parse_fail,
            task=task,
            model_name="test",
            max_retries=1,
            max_transport_retries=1,
            api_stats=[0]
        ))
    assert exc.value.diagnostics["first_validation_attempt_success"] is False

    # 2. Full success -> True
    valid_json = '''{
        "global_scene": {"scene_summary": "this is a valid summary with a lot more text to ensure it passes the scene summary min word count limit of twenty words.", "environment": "urban", "temporal_progression": "moving along down the road slowly for a while.", "physical_entities": [{"entity_id": "ent1", "category": "car", "referential_scope": "the car"}]},
        "video1_analysis": {"modality": "rgb", "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and depth representations so the pipeline test doesn't fail again.", "information_atoms": [{"atom_id": "v1_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}], "sensor_specific_cues": ["cue"], "sensor_limitations": ["The car surface is unclear in f1."], "uncertain_observations": [], "missing_key_attributes": []},
        "video2_analysis": {"modality": "depth", "detailed_caption": "this is a valid detailed caption with enough words to pass the min word count check for detailed caption. I am adding a lot more text here just to be absolutely certain that it passes the thirty words limit for video and depth representations so the pipeline test doesn't fail again.", "information_atoms": [{"atom_id": "v2_atom_1", "frame_keys": ["f1"], "entity_refs": ["ent1"], "fact": "fact here"}], "sensor_specific_cues": ["cue"], "sensor_limitations": ["The car boundary is unclear in f1."], "uncertain_observations": [], "missing_key_attributes": []}
    }'''
    provider_success = MockProvider([valid_json])
    _, _, diag = asyncio.run(_call_gemini_pass1(
        client_provider=provider_success,
        task=task,
        model_name="test",
        max_retries=1,
        max_transport_retries=1,
        api_stats=[0]
    ))
    assert diag["first_validation_attempt_success"] is True

def test_retry_prompt_format():
    from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import _format_pass1_validation_issues, _build_pass1_validation_retry_hint
    from annotation_feature.aligned_caption_pass1_validation import Pass1ValidationIssue, Pass1StructuralValidationError
    
    issues = [
        Pass1ValidationIssue(path="path.a", category="missing_field", message="msg a", scope="atom"),
        Pass1ValidationIssue(path="path.b", category="unexpected_field", message="msg b", scope="root")
    ]
    exc = Pass1StructuralValidationError("err", issues)
    
    # 1. check format string has real newlines and contains all parts
    formatted = _format_pass1_validation_issues(exc)
    assert "\n" in formatted  # Python represents real newlines as \n in repr, but it means a string with a newline character. Let's assert "\\n" is not in there.
    assert "\\n" not in formatted
    
    assert "missing_field" in formatted
    assert "path.a" in formatted
    assert "msg a" in formatted
    assert "unexpected_field" in formatted
    assert "path.b" in formatted
    assert "msg b" in formatted

    # 2. check hint generation
    hint = _build_pass1_validation_retry_hint(exc)
    assert "\n" in hint
    assert "\\n" not in hint
    assert "Pass 1 MUST only output global_scene" in hint # because scope="root" and unexpected_field
