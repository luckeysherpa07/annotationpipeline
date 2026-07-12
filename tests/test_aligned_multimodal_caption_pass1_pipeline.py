import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from annotation_feature.aligned_multimodal_caption_pipeline import CaptionTask
from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import (
    _call_gemini_pass1,
    _load_resume_pass1,
    _template_caption_pass1,
    run_caption_pipeline_pass1_async,
)
from annotation_feature.pipeline.client import GeminiKeysExhaustedError, ItemQuotaRetryLimitError, Pass1TransportError


@pytest.fixture
def dummy_tasks():
    tasks = []
    for i in range(1, 4):
        task = MagicMock(spec=CaptionTask)
        task.segment_id = f"seg{i}"
        task.caption_id = f"cap{i}"
        task.side = "left"
        task.modality1 = "rgb"
        task.modality2 = "event"
        task.split_dir = "train"
        task.segment_name = f"seg{i}"
        task.composite_frames = [Path(f"frame{i}.jpg")]
        tasks.append(task)
    return tasks


def test_pipeline_loop_control(tmp_path, dummy_tasks):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    
    with patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._call_gemini_pass1") as mock_call, \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.build_caption_tasks", return_value=(dummy_tasks, [], 3)), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._task_metadata", side_effect=lambda t: {"caption_id": t.caption_id}), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.GeminiClientProvider"), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.asyncio.sleep"):
        
        # Task 1 fails (ordinary transport error), Task 2 succeeds, Task 3 succeeds
        mock_call.side_effect = [
            Pass1TransportError("timeout", category="timeout", diagnostics={}, last_invalid_response=None),
            ({"global_scene": {}}, None, {"api_calls": 1, "validation_attempts": 1, "transport_retries": 0, "first_validation_attempt_success": True, "retry_history": [], "key_rotations": 0, "quota_failures": 0, "structural_validation_failures": 0, "semantic_validation_failures": 0}),
            ({"global_scene": {}}, None, {"api_calls": 1, "validation_attempts": 1, "transport_retries": 0, "first_validation_attempt_success": True, "retry_history": [], "key_rotations": 0, "quota_failures": 0, "structural_validation_failures": 0, "semantic_validation_failures": 0}),
        ]
        
        asyncio.run(run_caption_pipeline_pass1_async(
            input_path=input_path, output_path=output_path, dataset_root=tmp_path, composite_root=tmp_path,
            model_name="test-model", generation_mode="gemini", api_key_source="test",
            num_uniform_frames=8, num_adaptive_frames=2, pairs=None, directions=None, sides=None,
            limit=None, limit_scenes=None, limit_scene_folders=None, target_paths=None,
            max_retries=1, max_transport_retries=1, delay_between_calls=0, checkpoint_every=0, resume=False
        ))
            
    out = json.loads(output_path.read_text())
    assert mock_call.call_count == 3
    assert len(out["items"]) == 2
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["status"] == "failed"
    assert out["items"][0]["caption_id"] == "cap2"
    assert out["items"][1]["caption_id"] == "cap3"


def test_pipeline_checkpoint_behavior(tmp_path, dummy_tasks):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    
    with patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._call_gemini_pass1") as mock_call, \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.build_caption_tasks", return_value=(dummy_tasks, [], 3)), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._task_metadata", side_effect=lambda t: {"caption_id": t.caption_id}), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.GeminiClientProvider"), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._save_json") as mock_save, \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.asyncio.sleep"):
        
        # Task 1 fails, Task 2 succeeds, Task 3 succeeds.
        mock_call.side_effect = [
            Pass1TransportError("timeout", category="timeout", diagnostics={}, last_invalid_response=None),
            ({"global_scene": {}}, None, {"api_calls": 1}),
            ({"global_scene": {}}, None, {"api_calls": 1}),
        ]
        
        save_calls = []
        def mock_save_side_effect(payload, path):
            save_calls.append({
                "items": len(payload["items"]),
                "skipped": len(payload["skipped"])
            })
            
        mock_save.side_effect = mock_save_side_effect

        asyncio.run(run_caption_pipeline_pass1_async(
            input_path=input_path, output_path=output_path, dataset_root=tmp_path, composite_root=tmp_path,
            model_name="test-model", generation_mode="gemini", api_key_source="test",
            num_uniform_frames=8, num_adaptive_frames=2, pairs=None, directions=None, sides=None,
            limit=None, limit_scenes=None, limit_scene_folders=None, target_paths=None,
            max_retries=1, max_transport_retries=1, delay_between_calls=0,
            checkpoint_every=2,  # Checkpoint every 2 items
            resume=False
        ))
        
        # mock_save is called once for checkpoint 1 (after 2 processed items) and once at the end.
        assert len(save_calls) == 2
        
        # First checkpoint after 2 items (one fail, one success)
        assert save_calls[0]["items"] == 1
        assert save_calls[0]["skipped"] == 1
        
        # Second checkpoint at the end (one fail, two successes)
        assert save_calls[1]["items"] == 2
        assert save_calls[1]["skipped"] == 1


def test_pipeline_provider_init_failure(tmp_path, dummy_tasks):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    
    with patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._call_gemini_pass1") as mock_call, \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.build_caption_tasks", return_value=(dummy_tasks, [], 3)), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.GeminiClientProvider") as mock_provider, \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._save_json") as mock_save:
        
        mock_provider.side_effect = GeminiKeysExhaustedError("All keys exhausted")
        
        asyncio.run(run_caption_pipeline_pass1_async(
            input_path=input_path, output_path=output_path, dataset_root=tmp_path, composite_root=tmp_path,
            model_name="test-model", generation_mode="gemini", api_key_source="test",
            num_uniform_frames=8, num_adaptive_frames=2, pairs=None, directions=None, sides=None,
            limit=None, limit_scenes=None, limit_scene_folders=None, target_paths=None,
            max_retries=1, max_transport_retries=1, delay_between_calls=0, checkpoint_every=0, resume=False
        ))
        
        # No tasks should be processed
        mock_call.assert_not_called()
        
        # Checkpoint is written once due to init error
        assert mock_save.call_count == 1
        payload = mock_save.call_args_list[0][0][0]
        assert payload["metadata"]["initialization_error_category"] == "quota_exhausted"
        assert payload["metadata"]["initialization_error"] == "All keys exhausted"
        assert payload["metadata"]["gemini_calls"] == 0
        assert len(payload["items"]) == 0


def test_pipeline_diagnostics(tmp_path, dummy_tasks):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    
    diagnostics_payload = {
        "api_calls": 2,
        "validation_attempts": 2,
        "transport_retries": 1,
        "first_validation_attempt_success": False,
        "retry_history": ["error1"],
        "key_rotations": 1,
        "quota_failures": 0,
        "structural_validation_failures": 1,
        "semantic_validation_failures": 0
    }
    
    with patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._call_gemini_pass1") as mock_call, \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.build_caption_tasks", return_value=([dummy_tasks[0]], [], 1)), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._task_metadata", side_effect=lambda t: {"caption_id": t.caption_id}), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.GeminiClientProvider"):
        
        mock_call.return_value = ({"global_scene": {}}, None, diagnostics_payload)
        
        asyncio.run(run_caption_pipeline_pass1_async(
            input_path=input_path, output_path=output_path, dataset_root=tmp_path, composite_root=tmp_path,
            model_name="test-model", generation_mode="gemini", api_key_source="test",
            num_uniform_frames=8, num_adaptive_frames=2, pairs=None, directions=None, sides=None,
            limit=None, limit_scenes=None, limit_scene_folders=None, target_paths=None,
            max_retries=1, max_transport_retries=1, delay_between_calls=0, checkpoint_every=0, resume=False
        ))
        
    out = json.loads(output_path.read_text())
    assert len(out["items"]) == 1
    diag = out["items"][0]["diagnostics"]
    
    # Assert all required keys are present
    expected_keys = {
        "api_calls", "validation_attempts", "transport_retries", "first_validation_attempt_success",
        "retry_history", "key_rotations", "quota_failures", "structural_validation_failures",
        "semantic_validation_failures"
    }
    assert expected_keys.issubset(diag.keys())


def test_pipeline_item_quota_error(tmp_path, dummy_tasks):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    
    with patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._call_gemini_pass1") as mock_call, \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.build_caption_tasks", return_value=([dummy_tasks[0], dummy_tasks[1]], [], 2)), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._task_metadata", side_effect=lambda t: {"caption_id": t.caption_id}), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.GeminiClientProvider"), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.asyncio.sleep"):
        
        mock_call.side_effect = [
            ItemQuotaRetryLimitError("per-item quota failed"),
            ({"global_scene": {}}, None, {"api_calls": 1})
        ]
        
        asyncio.run(run_caption_pipeline_pass1_async(
            input_path=input_path, output_path=output_path, dataset_root=tmp_path, composite_root=tmp_path,
            model_name="test-model", generation_mode="gemini", api_key_source="test",
            num_uniform_frames=8, num_adaptive_frames=2, pairs=None, directions=None, sides=None,
            limit=None, limit_scenes=None, limit_scene_folders=None, target_paths=None,
            max_retries=1, max_transport_retries=1, delay_between_calls=0, checkpoint_every=0, resume=False
        ))
        
    out = json.loads(output_path.read_text())
    assert len(out["items"]) == 1
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["final_error_category"] == "item_quota_retry_limit"
    assert out["skipped"][0]["status"] == "failed"


def test_warning_only_validation_does_not_retry(dummy_tasks):
    import sys
    from types import ModuleType

    provider = MagicMock()
    client = MagicMock()
    provider.get_client.return_value = client
    client.models.generate_content.return_value = MagicMock(text='{"placeholder": true}')
    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.types = MagicMock()
    google_module.genai = genai_module

    with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module, "google.genai.types": genai_module.types}), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._ensure_composite_frames"), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._encode_images", return_value=["encoded"]), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline.build_image_parts", create=True, return_value=[]), \
         patch("annotation_feature.pipeline.utils.build_image_parts", return_value=[]), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._validate_pass1_schema", return_value=({"ok": True}, ["weak_cross_source_recoverability"])), \
         patch("annotation_feature.aligned_multimodal_caption_two_pass_pipeline._parse_json_response", return_value={"placeholder": True}):
        evidence, warnings, diagnostics = asyncio.run(_call_gemini_pass1(
            client_provider=provider,
            task=dummy_tasks[0],
            model_name="test-model",
            max_retries=3,
            max_transport_retries=1,
            api_stats=[0],
        ))

    assert evidence == {"ok": True}
    assert warnings == ["weak_cross_source_recoverability"]
    assert diagnostics["api_calls"] == 1
    assert diagnostics["validation_attempts"] == 1
    assert client.models.generate_content.call_count == 1


def test_resume_rejects_legacy_missing_attribute_without_entity_id(tmp_path, dummy_tasks):
    task = dummy_tasks[0]
    task.composite_frames = [Path("frame1.jpg")]
    evidence = _template_caption_pass1(task)
    del evidence["video1_analysis"]["missing_key_attributes"][0]["entity_id"]
    output_path = tmp_path / "resume.json"
    output_path.write_text(json.dumps({
        "items": [{
            "caption_id": task.caption_id,
            "modality1": "rgb",
            "modality2": "event",
            "composite_frames": ["frame1.jpg"],
            "evidence": evidence,
        }],
        "skipped": [],
    }), encoding="utf-8")

    items, skipped = _load_resume_pass1(output_path)
    assert items == []
    assert skipped == []


def test_unsupported_modality_fails_before_gemini_call(dummy_tasks):
    task = dummy_tasks[0]
    task.modality2 = "lidar"
    provider = MagicMock()
    with pytest.raises(Exception) as exc_info:
        asyncio.run(_call_gemini_pass1(
            client_provider=provider,
            task=task,
            model_name="test-model",
            max_retries=1,
            max_transport_retries=1,
            api_stats=[0],
        ))
    assert any(issue.category == "unsupported_modality" for issue in exc_info.value.errors)
    provider.get_client.assert_not_called()
