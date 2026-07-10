"""Generate Pass 1 (evidence construction) outputs from aligned frame caches."""

from __future__ import annotations

import argparse
import asyncio
import json
import datetime
from pathlib import Path
from typing import Any

from annotation_feature.pipeline.client import create_gemini_client

from annotation_feature.aligned_caption_schema import (
    CaptionParseError,
    CaptionValidationError,
    MIN_DETAILED_CAPTION_WORDS,
)

# Import shared pipeline components
from annotation_feature.aligned_multimodal_caption_pipeline import (
    DEFAULT_COMPOSITE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MAX_TRANSPORT_RETRIES,
    CaptionTask,
    build_caption_tasks,
    _ensure_composite_frames,
    _encode_images,
    _parse_json_response,
    _is_transport_error,
    _transport_error_category,
    _transport_retry_wait_seconds,
    _load_json,
    _save_json,
    _task_metadata,
)

# Import Pass 1 specifics
from annotation_feature.aligned_caption_pass1_prompt import (
    build_pass1_system_prompt,
    build_pass1_user_prompt,
    _template_caption_pass1,
)
from annotation_feature.aligned_caption_pass1_validation import _validate_pass1_schema

DEFAULT_INPUT_PATH = Path("outputs/aligned_multimodal_visual_evidence_units_filtered.json")
DEFAULT_OUTPUT_PATH = Path("outputs/pass1_evidence_construction.json")
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"


def _build_pass1_validation_retry_hint(exc: Exception, category: str) -> str:
    message = str(exc).lower()
    hints: list[str] = []
    
    if "unknown top-level fields" in message or "unknown fields for pass 1" in message:
        hints.append(
            "Pass 1 MUST only output global_scene, video1_analysis, and video2_analysis. "
            "Remove all downstream cross-modal reasoning fields (like reasoning_focus_entities, "
            "cross_modal_evidence_links, information_gain, reasoning_events, ambiguity_events, etc.)."
        )
        
    if "recoverable_evidence_refs" in message:
        hints.append(
            "In Pass 1, missing_key_attributes[].recoverable_evidence_refs MUST ALWAYS be an empty list []. "
            "Do not attempt to recover attributes across modalities in this pass."
        )

    if category == "blocklist_failure":
        hints.append(
            "Rewrite the reported field using physical-world wording only. "
            "Do not mention sensor names, modality names, camera/frame/image-processing terms, "
            "or image-quality descriptions. Keep detailed captions above the minimum word count."
        )
    if "too short" in message:
        hints.append(
            "Expand the reported detailed_caption into a complete source-local paragraph of at least "
            f"{MIN_DETAILED_CAPTION_WORDS} words, while keeping forbidden sensor-quality wording out."
        )
    if "generic sensor-theory" in message or "segment-specific" in message or "forbidden wording" in message:
        hints.append(
            "Rewrite the reported field using segment-specific evidence only. "
            "Apply the GENERICITY TEST: if the sentence would be equally true for any segment of the same modality, it is forbidden. "
            "FORBIDDEN: 'The modality lacks intensity data.', 'The sensor does not record color.', 'The capture mechanism registers only intensity changes.' "
            "REQUIRED form for why_missing: describe what is concretely absent in these specific frames, e.g. 'Vehicle surfaces in the sampled frames show no stable internal detail sufficient to distinguish paint color.' "
            "REQUIRED form for sensor_limitations: cite a specific frame range or observable condition, e.g. 'Sunlight glare on windshields reduces surface detail in the later frames.'"
        )
    if "hypotheses" in message or "hypothesis" in message or "meta-statement" in message or "confidence" in message:
        hints.append(
            "Every uncertain_observations item must contain a valid uncertainty_id, "
            "one known entity_id, and at least one same-source evidence_ref connected "
            "to that entity. hypotheses may be empty; if non-empty, they must contain "
            "at least 2 distinct candidate interpretations. Do not output meta-statements of inability (e.g. 'cannot be determined'). "
            "Hypotheses must be plausible factual candidate interpretations. "
            "Each hypothesis item must be exactly: {\"hypothesis\": \"<text>\", \"confidence\": \"low|medium|high\"}. "
            "No other keys like hypothesis_id are allowed."
        )
    if category in {"invalid_reference", "missing_attribute_recovery"}:
        hints.append(
            "Re-check all IDs and references after the edit. Every referenced atom, entity, and frame_key "
            "must exist and must keep the required prefix. Duplicate IDs are not allowed."
        )
    if not hints:
        return ""
    return " Targeted repair guidance: " + " ".join(hints)


async def _call_gemini_pass1(
    client,
    task: CaptionTask,
    model_name: str,
    max_retries: int,
    max_transport_retries: int,
    api_stats: list[int] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    _ensure_composite_frames(task)
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError("No composite frames found for Gemini call")
        
    from google.genai import types as genai_types
    from annotation_feature.pipeline.utils import build_image_parts
    
    system_instruction = build_pass1_system_prompt()
    base_contents = build_image_parts(encoded) + [build_pass1_user_prompt(task)]
    contents = base_contents
    raw_text = None
    validation_attempt = 1
    diagnostics = {
        "api_calls": 0,
        "validation_attempts": 0,
        "transport_retries": 0,
        "first_validation_attempt_success": True,
        "retry_history": []
    }
    while validation_attempt <= max_retries:
        try:
            response = None
            for transport_attempt in range(1, max_transport_retries + 1):
                if api_stats is not None:
                    api_stats[0] += 1
                diagnostics["api_calls"] += 1
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model_name,
                        contents=contents,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_instruction,
                        ),
                    )
                    break
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if "429" in exc_str or "quota" in exc_str:
                        raise
                    if not _is_transport_error(exc):
                        raise

                    diagnostics["transport_retries"] += 1
                    category = _transport_error_category(exc)
                    diagnostics["retry_history"].append({
                        "type": "transport",
                        "category": category,
                        "validation_attempt": validation_attempt,
                        "transport_attempt": transport_attempt,
                        "message": str(exc)[:200]
                    })
                    log_msg = {
                        "caption_id": task.caption_id,
                        "attempt": transport_attempt,
                        "validation_attempt": validation_attempt,
                        "transport_attempt": transport_attempt,
                        "max_transport_retries": max_transport_retries,
                        "stage": "generation",
                        "category": category,
                        "message": str(exc),
                    }
                    if transport_attempt < max_transport_retries:
                        log_msg["retry_after_seconds"] = _transport_retry_wait_seconds(transport_attempt)
                    print(f"    Failure: {json.dumps(log_msg)}")

                    if transport_attempt == max_transport_retries:
                        raise
                    await asyncio.sleep(_transport_retry_wait_seconds(transport_attempt))

            if response is None:
                raise RuntimeError("Gemini caption call failed before receiving a response")
            raw_text = response.text
            diagnostics["validation_attempts"] += 1
            valid_frame_keys = {path.stem for path in task.composite_frames}
            evidence, warnings = _validate_pass1_schema(_parse_json_response(raw_text), valid_frame_keys, task.modality1, task.modality2)
            return evidence, warnings, diagnostics
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "quota" in exc_str:
                raise

            if _is_transport_error(exc):
                raise

            if diagnostics["validation_attempts"] == 1:
                diagnostics["first_validation_attempt_success"] = False

            if isinstance(exc, CaptionParseError):
                category = "parse_error"
            else:
                if "blocklist" in exc_str or "forbidden" in exc_str:
                    category = "blocklist_failure"
                elif "missing_key_attributes" in exc_str or "recoverable_evidence_refs" in exc_str:
                    category = "missing_attribute_recovery"
                elif "reference" in exc_str or "duplicate" in exc_str or "unknown" in exc_str:
                    category = "invalid_reference"
                else:
                    category = "schema_validation_error"
            
            diagnostics["retry_history"].append({
                "type": "validation",
                "category": category,
                "validation_attempt": validation_attempt,
                "message": str(exc)[:200]
            })

            log_msg = {
                "caption_id": task.caption_id,
                "validation_attempt": validation_attempt,
                "max_validation_retries": max_retries,
                "stage": "validation" if isinstance(exc, CaptionValidationError) else ("parse" if isinstance(exc, CaptionParseError) else "generation"),
                "category": category,
                "message": str(exc)
            }
            print(f"    Failure: {json.dumps(log_msg)}")

            if validation_attempt == max_retries:
                if raw_text:
                    exc.last_invalid_response = raw_text
                exc.diagnostics = diagnostics
                raise

            previous_context = ""
            if raw_text:
                previous_context = f"\n\nHere is your previous invalid response:\n```json\n{raw_text}\n```\n\n"
            error_feedback = (
                f"{previous_context}"
                f"Your previous response failed validation. "
                f"The first detected validation error was: [{exc}]. "
                f"{_build_pass1_validation_retry_hint(exc, category)} "
                f"Correct this issue by repairing only the PASS 1 evidence package. "
                f"Preserve valid entities and atoms where possible. "
                f"Return the complete corrected PASS 1 JSON. "
                f"Do not add downstream reasoning fields. Keep all recoverable_evidence_refs equal to []."
            )
            contents = base_contents + [error_feedback]
            validation_attempt += 1
    raise RuntimeError("Gemini pass 1 call failed")


def _task_to_item_pass1(
    task: CaptionTask, 
    status: str, 
    evidence: dict[str, Any] | None = None, 
    validation_warnings: list[str] | None = None, 
    reason: str | None = None, 
    attempts: int | None = None, 
    first_attempt_success: bool | None = None, 
    final_error_category: str | None = None, 
    last_invalid_response: str | None = None,
    api_calls: int | None = None,
    validation_attempts: int | None = None,
    transport_retries: int | None = None,
    first_validation_attempt_success: bool | None = None,
    retry_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    item = _task_metadata(task)
    item.update({
        "status": status,
        "reason": reason,
        "attempts": attempts,
        "first_attempt_success": first_attempt_success,
        "final_error_category": final_error_category,
        "evidence": evidence,
        "validation_warnings": validation_warnings or [],
    })
    if api_calls is not None:
        item["api_calls"] = api_calls
    if validation_attempts is not None:
        item["validation_attempts"] = validation_attempts
    if transport_retries is not None:
        item["transport_retries"] = transport_retries
    if first_validation_attempt_success is not None:
        item["first_validation_attempt_success"] = first_validation_attempt_success
    if retry_history is not None:
        item["retry_history"] = retry_history

    if last_invalid_response is not None:
        item["last_invalid_response"] = last_invalid_response
    return item


def _build_output_payload_pass1(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_total: int,
    gemini_calls: int,
) -> dict[str, Any]:
    return {
        "metadata": {
            "task": "cross_modal_disambiguation_caption",
            "architecture": "two_pass",
            "stage": "pass1",
            "input": input_path.as_posix(),
            "output": output_path.as_posix(),
            "dataset_root": dataset_root.as_posix(),
            "composite_root": composite_root.as_posix(),
            "generation_mode": generation_mode,
            "model_name": model_name,
            "planned_items": planned_total,
            "completed_items": len(items),
            "skipped_items": len(skipped),
            "gemini_calls": gemini_calls,
        },
        "items": items,
        "skipped": skipped,
    }


def _load_resume_pass1(output_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return [], []
    try:
        data = _load_json(output_path)
    except Exception as exc:
        print(f"WARNING: Could not load existing pass1 output for resume: {exc}")
        return [], []
    items = data.get("items") if isinstance(data.get("items"), list) else []
    skipped = data.get("skipped") if isinstance(data.get("skipped"), list) else []
    
    valid_items = []
    for item in items:
        ev = item.get("evidence")
        if not isinstance(ev, dict):
            continue

        try:
            import copy
            composite_frames = item.get("composite_frames") or []
            valid_frame_keys = {
                Path(frame_name).stem
                for frame_name in composite_frames
                if isinstance(frame_name, str) and frame_name.strip()
            }
            modality1 = str(item.get("modality1") or "")
            modality2 = str(item.get("modality2") or "")

            if not valid_frame_keys or not modality1 or not modality2:
                continue

            _validate_pass1_schema(
                copy.deepcopy(ev),
                valid_frame_keys,
                modality1,
                modality2,
            )
        except (CaptionValidationError, CaptionParseError, ValueError, TypeError):
            continue

        valid_items.append(item)

    return valid_items, skipped


async def run_caption_pipeline_pass1_async(
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    composite_root: Path,
    model_name: str,
    generation_mode: str,
    api_key_source: str,
    num_uniform_frames: int,
    num_adaptive_frames: int,
    pairs: str | None,
    directions: str | None,
    sides: str | None,
    limit: int | None,
    limit_scenes: int | None,
    limit_scene_folders: int | None,
    target_paths: str | None,
    max_retries: int,
    max_transport_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    from annotation_feature.aligned_multimodal_caption_pipeline import (
        _parse_pairs, _parse_sides
    )
    
    allowed_pairs = _parse_pairs(pairs)
    allowed_directions = _parse_pairs(directions)
    allowed_sides = _parse_sides(sides)
    parsed_target_paths = {p.strip().replace('\\', '/') for p in target_paths.split(',')} if target_paths else None
    
    client = create_gemini_client(api_key_source=api_key_source) if generation_mode in ("gemini", "batch") else None
    
    existing_items, existing_skipped = _load_resume_pass1(output_path) if resume else ([], [])
    items = existing_items
    
    tasks, scoped_skipped, total_selected_jobs = build_caption_tasks(
        input_path=input_path,
        dataset_root=dataset_root,
        composite_root=composite_root,
        sampling_strategy="uniform_adaptive",
        num_uniform_frames=num_uniform_frames,
        num_adaptive_frames=num_adaptive_frames,
        existing_items=existing_items,
        existing_skipped=existing_skipped,
        allowed_pairs=allowed_pairs,
        allowed_directions=allowed_directions,
        limit=limit,
        limit_scenes=limit_scenes,
        limit_scene_folders=limit_scene_folders,
        allowed_sides=allowed_sides,
        target_paths=parsed_target_paths,
        write_composites=False,
    )
    
    seen_skipped = set()
    deduped_skipped = []
    for s in scoped_skipped:
        key = (s.get("segment_id"), s.get("side"), s.get("modality1"), s.get("modality2"), s.get("reason"), s.get("caption_id"))
        if key not in seen_skipped:
            seen_skipped.add(key)
            deduped_skipped.append(s)
    skipped = deduped_skipped
    pending_tasks = tasks

    api_stats = [0]
    checkpoint_counter = 0

    print(
        f"Generating pass1 evidence: {len(tasks)} planned item(s), "
        f"{len(pending_tasks)} pending, mode={generation_mode}, model={model_name}."
    )
    
    if generation_mode == "batch":
        print("Batch mode not natively fully customized here yet. Recommend running gemini mode.")
        return output_path

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload_pass1(
                input_path=input_path,
                dataset_root=dataset_root,
                composite_root=composite_root,
                output_path=output_path,
                model_name=model_name,
                generation_mode=generation_mode,
                items=items,
                skipped=skipped,
                planned_total=total_selected_jobs,
                gemini_calls=api_stats[0],
            ),
            output_path,
        )

    task_index = 0
    MAX_KEY_ROTATIONS = 5
    rotation_attempts = 0

    while task_index < len(pending_tasks):
        task = pending_tasks[task_index]
        print(
            f"  Pass 1 item [{task_index + 1}/{len(pending_tasks)}] "
            f"{task.caption_id}"
        )
        initial_api_stats = api_stats[0]
        try:
            if generation_mode == "gemini":
                assert client is not None
                evidence, warnings, diagnostics = await _call_gemini_pass1(
                    client,
                    task,
                    model_name,
                    max_retries=max_retries,
                    max_transport_retries=max_transport_retries,
                    api_stats=api_stats,
                )
                attempts_used = api_stats[0] - initial_api_stats
                status = "generated"
                
                skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
                items[:] = [item for item in items if item.get("caption_id") != task.caption_id]
                items.append(_task_to_item_pass1(
                    task, 
                    status=status, 
                    evidence=evidence,
                    validation_warnings=warnings,
                    attempts=attempts_used,
                    first_attempt_success=(attempts_used == 1),
                    final_error_category=None,
                    api_calls=diagnostics["api_calls"],
                    validation_attempts=diagnostics["validation_attempts"],
                    transport_retries=diagnostics["transport_retries"],
                    first_validation_attempt_success=diagnostics["first_validation_attempt_success"],
                    retry_history=diagnostics["retry_history"]
                ))
            else:
                _ensure_composite_frames(task)
                valid_frame_keys = {path.stem for path in task.composite_frames}
                evidence, warnings = _validate_pass1_schema(_template_caption_pass1(task), valid_frame_keys, task.modality1, task.modality2)
                status = "template"
                skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
                items[:] = [item for item in items if item.get("caption_id") != task.caption_id]
                items.append(_task_to_item_pass1(
                    task, 
                    status=status, 
                    evidence=evidence, 
                    validation_warnings=warnings,
                    api_calls=0,
                    validation_attempts=0,
                    transport_retries=0,
                    first_validation_attempt_success=True,
                    retry_history=[]
                ))
            rotation_attempts = 0
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "quota" in exc_str:
                rotation_attempts += 1
                if rotation_attempts > MAX_KEY_ROTATIONS:
                    raise RuntimeError("Exceeded maximum API-key rotation attempts")
                print(f"WARNING: Quota exhausted or rate limit hit. Attempting to rotate API key...")
                try:
                    client = create_gemini_client(api_key_source=api_key_source)
                    continue
                except Exception as rotate_exc:
                    print(f"FATAL: All API keys exhausted or failed to rotate: {rotate_exc}")
                    break
            
            final_error_category = "transport_other"
            exc_str_lower = str(exc).lower()
            if isinstance(exc, CaptionParseError):
                final_error_category = "parse_error"
            elif isinstance(exc, CaptionValidationError):
                if "blocklist" in exc_str_lower or "forbidden" in exc_str_lower:
                    final_error_category = "blocklist_failure"
                elif "missing_key_attributes" in exc_str_lower or "recoverable_evidence_refs" in exc_str_lower:
                    final_error_category = "missing_attribute_recovery"
                elif "reference" in exc_str_lower or "duplicate" in exc_str_lower or "unknown" in exc_str_lower:
                    final_error_category = "invalid_reference"
                else:
                    final_error_category = "schema_validation_error"
            elif _is_transport_error(exc):
                final_error_category = _transport_error_category(exc)
            elif "429" in exc_str_lower or "quota" in exc_str_lower:
                final_error_category = "quota_exhausted"
                
            diagnostics = getattr(exc, "diagnostics", {
                "api_calls": api_stats[0] - initial_api_stats,
                "validation_attempts": 0,
                "transport_retries": 0,
                "first_validation_attempt_success": False,
                "retry_history": []
            })
            
            skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
            
            skipped.append(
                _task_to_item_pass1(
                    task,
                    status="failed",
                    reason=str(exc),
                    attempts=api_stats[0] - initial_api_stats,
                    first_attempt_success=False,
                    final_error_category=final_error_category,
                    last_invalid_response=getattr(exc, "last_invalid_response", None),
                    api_calls=diagnostics["api_calls"],
                    validation_attempts=diagnostics["validation_attempts"],
                    transport_retries=diagnostics["transport_retries"],
                    first_validation_attempt_success=diagnostics["first_validation_attempt_success"],
                    retry_history=diagnostics["retry_history"]
                )
            )
            print(f"WARNING: Pass 1 generation failed for {task.caption_id}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and task_index < len(pending_tasks) - 1:
            await asyncio.sleep(delay_between_calls)
            
        task_index += 1

    save_checkpoint()
    print(f"Wrote Pass 1 output to {output_path}")
    return output_path


def run_caption_pipeline_pass1(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    composite_root: Path | str = DEFAULT_COMPOSITE_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    generation_mode: str = "template",
    api_key_source: str = "list",
    num_uniform_frames: int = 8,
    num_adaptive_frames: int = 2,
    pairs: str | None = None,
    directions: str | None = None,
    sides: str | None = None,
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    target_paths: str | None = None,
    max_retries: int = 3,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> Path:
    return asyncio.run(
        run_caption_pipeline_pass1_async(
            input_path=Path(input_path),
            output_path=Path(output_path),
            dataset_root=Path(dataset_root),
            composite_root=Path(composite_root),
            model_name=model_name,
            generation_mode=generation_mode,
            api_key_source=api_key_source,
            num_uniform_frames=num_uniform_frames,
            num_adaptive_frames=num_adaptive_frames,
            pairs=pairs,
            directions=directions,
            sides=sides,
            limit=limit,
            limit_scenes=limit_scenes,
            limit_scene_folders=limit_scene_folders,
            target_paths=target_paths,
            max_retries=max_retries,
            max_transport_retries=max_transport_retries,
            delay_between_calls=delay_between_calls,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--composite-root", default=str(DEFAULT_COMPOSITE_ROOT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--generation-mode",
        choices=("template", "gemini"),
        default="template",
        help="Use template to build composite frames without calling Gemini. Use gemini for generation.",
    )
    parser.add_argument("--api-key-source", choices=("env", "list"), default="list", help="Source for Gemini API keys.")
    parser.add_argument("--num-uniform-frames", type=int, default=8)
    parser.add_argument("--num-adaptive-frames", type=int, default=2)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--sides", default=None)
    parser.add_argument("--directions", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-scenes", dest="limit_scenes", type=int, default=None)
    parser.add_argument("--limit-scene-folders", dest="limit_scene_folders", type=int, default=None)
    parser.add_argument("--target-paths", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-transport-retries", type=int, default=DEFAULT_MAX_TRANSPORT_RETRIES)
    parser.add_argument("--delay-between-calls", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    
    return parser


def main() -> None:
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    args = _build_arg_parser().parse_args()
    
    run_caption_pipeline_pass1(
        input_path=args.input,
        output_path=args.output,
        dataset_root=args.dataset_root,
        composite_root=args.composite_root,
        model_name=args.model_name,
        generation_mode=args.generation_mode,
        api_key_source=args.api_key_source,
        num_uniform_frames=max(1, args.num_uniform_frames),
        num_adaptive_frames=max(0, args.num_adaptive_frames),
        pairs=args.pairs,
        directions=args.directions,
        sides=args.sides,
        limit=args.limit,
        limit_scenes=args.limit_scenes,
        limit_scene_folders=args.limit_scene_folders,
        target_paths=args.target_paths,
        max_retries=max(1, args.max_retries),
        max_transport_retries=max(1, args.max_transport_retries),
        delay_between_calls=max(0, args.delay_between_calls),
        checkpoint_every=max(0, args.checkpoint_every),
        resume=not args.no_resume,
    )

if __name__ == "__main__":
    main()
