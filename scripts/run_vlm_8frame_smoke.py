#!/usr/bin/env python3
"""Run a 4-modality x 5-item VLM smoke test from one shared 8-frame manifest."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from annotation_feature.qa_quality.benchmark import (
    DEFAULT_FRAME_CACHE_ROOT,
    DEFAULT_INPUT_PATH,
    InternVLFrameAnswerAdapter,
    QwenVLFrameAnswerAdapter,
    _frame_answer_row,
    _frame_answer_output_paths,
    _save_frame_answer_outputs,
    load_valid_qa_items,
    resolve_frame_inputs_for_item,
)

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None


DEFAULT_MANIFEST_PATH = Path("outputs/benchmarks/smoke_4modalities_5items_8frames_manifest.json")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks/smoke_4modalities_5items_8frames")
DEFAULT_QWEN_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_INTERNVL_8B = "OpenGVLab/InternVL3-8B"
DEFAULT_MOLMO2_8B = "allenai/Molmo-2-8B"
DEFAULT_MODALITIES = ("rgb", "ir", "event", "depth")


class Molmo2FrameAnswerAdapter:
    """Best-effort local Molmo2 adapter using the Transformers image-text API."""

    provider = "molmo2"

    def __init__(self, model_name: str, max_tokens: int = 128, require_cuda: bool = True):
        self.model_name = model_name
        self.max_tokens = max(1, int(max_tokens))
        if torch is None:
            raise RuntimeError("PyTorch is not installed.")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Local Molmo2 smoke test requires GPU execution.")
        if Image is None:
            raise RuntimeError("Pillow is not installed.")
        if AutoProcessor is None or AutoModelForImageTextToText is None or BitsAndBytesConfig is None:
            raise RuntimeError("Molmo2 smoke test requires transformers and bitsandbytes.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        if hasattr(self.model, "eval"):
            self.model.eval()

    def answer(self, item: dict[str, Any], frame_paths: list[Path]) -> str:
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        prompt = "\n".join(
            [
                "Answer using only the provided image frames.",
                "Return only a concise answer. Do not include explanation.",
                f"Modality: {item.get('modality', '')}",
                f"Section: {item.get('section', '')}",
                f"Pair key: {item.get('pair_key', '')}",
                f"Provided frames: {', '.join(path.name for path in frame_paths)}",
                "",
                "Question:",
                str(item.get("question", "")).strip(),
            ]
        )

        if hasattr(self.processor, "apply_chat_template"):
            content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=[prompt], images=images, return_tensors="pt", padding=True)

        device = _model_input_device(self.model)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
        input_length = int(input_ids.shape[-1]) if getattr(input_ids, "shape", None) is not None else 0
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        sequence = generated[0] if getattr(generated, "shape", None) is not None and len(generated.shape) > 1 else generated
        new_tokens = sequence[input_length:] if input_length else sequence
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def build_manifest(
    input_path: Path,
    output_path: Path,
    frame_cache_root: Path,
    modalities: tuple[str, ...],
    items_per_modality: int,
    frames_per_item: int,
) -> dict[str, Any]:
    items = load_valid_qa_items(input_path)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    wanted = {modality.lower() for modality in modalities}
    for item in items:
        modality = str(item.get("modality", "")).lower()
        if modality not in wanted or counts[modality] >= items_per_modality:
            continue
        frame_paths = resolve_frame_inputs_for_item(
            item,
            frame_cache_root=frame_cache_root,
            max_frames_per_item=frames_per_item,
        )
        if len(frame_paths) < frames_per_item:
            continue
        selected.append(
            {
                "qa_id": item["qa_id"],
                "modality": item["modality"],
                "section": item["section"],
                "pair_key": item["pair_key"],
                "question": item["question"],
                "frame_paths": [path.as_posix() for path in frame_paths[:frames_per_item]],
            }
        )
        counts[modality] += 1
        if all(counts[modality] >= items_per_modality for modality in wanted):
            break

    manifest = {
        "metadata": {
            "input_path": input_path.as_posix(),
            "frame_cache_root": frame_cache_root.as_posix(),
            "modalities": list(modalities),
            "items_per_modality": items_per_modality,
            "frames_per_item": frames_per_item,
            "selected_items": len(selected),
            "counts_by_modality": dict(sorted(counts.items())),
        },
        "items": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Invalid frame manifest: {path}")
    return payload


def items_from_manifest(input_path: Path, manifest: dict[str, Any]) -> list[tuple[dict[str, Any], list[Path]]]:
    source_by_id = {item["qa_id"]: item for item in load_valid_qa_items(input_path)}
    resolved: list[tuple[dict[str, Any], list[Path]]] = []
    for entry in manifest["items"]:
        qa_id = str(entry.get("qa_id", ""))
        item = source_by_id.get(qa_id)
        if not item:
            continue
        frame_paths = [Path(path) for path in entry.get("frame_paths", [])]
        resolved.append((item, frame_paths))
    return resolved


def run_model(
    label: str,
    model_name: str,
    adapter: Any,
    manifest_items: list[tuple[dict[str, Any], list[Path]]],
    output_dir: Path,
    manifest_path: Path,
) -> None:
    output_json, output_csv = _frame_answer_output_paths(output_dir, model_name)
    results: dict[str, dict[str, Any]] = {}
    for item, frame_paths in manifest_items:
        try:
            answer = adapter.answer(item, frame_paths)
            results[item["qa_id"]] = _frame_answer_row(
                item,
                provider=label,
                model_name=model_name,
                model_answer=answer,
                frame_paths=frame_paths,
                status="answered" if answer else "failed",
                reason="" if answer else "Frame answer call failed: empty model answer",
            )
        except Exception as exc:
            results[item["qa_id"]] = _frame_answer_row(
                item,
                provider=label,
                model_name=model_name,
                model_answer="",
                frame_paths=frame_paths,
                status="failed",
                reason=f"Frame answer call failed: {exc}",
            )
        _save_frame_answer_outputs(
            output_json,
            output_csv,
            results,
            {
                "benchmark_type": "shared_manifest_8frame_smoke",
                "provider": label,
                "model_name": model_name,
                "frame_manifest_path": manifest_path.as_posix(),
                "answered_items": sum(1 for row in results.values() if row.get("status") == "answered"),
                "attempted_items": len(results),
                "judge_enabled": False,
            },
        )
    print(f"{label}: wrote {output_json} and {output_csv}")


def clear_cuda() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--frame-cache-root", default=str(DEFAULT_FRAME_CACHE_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--items-per-modality", type=int, default=5)
    parser.add_argument("--frames-per-item", type=int, default=8)
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--models", default="qwen_vl,internvl,molmo2")
    parser.add_argument("--qwen-vl-model", default=DEFAULT_QWEN_VL_8B)
    parser.add_argument("--internvl-model", default=DEFAULT_INTERNVL_8B)
    parser.add_argument("--molmo2-model", default=DEFAULT_MOLMO2_8B)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    modalities = tuple(part.strip().lower() for part in args.modalities.split(",") if part.strip())

    manifest = build_manifest(
        input_path=input_path,
        output_path=manifest_path,
        frame_cache_root=Path(args.frame_cache_root),
        modalities=modalities,
        items_per_modality=max(1, args.items_per_modality),
        frames_per_item=max(1, args.frames_per_item),
    )
    print(f"manifest: {manifest_path}")
    print(json.dumps(manifest["metadata"], indent=2, ensure_ascii=False))
    if args.build_only:
        return

    manifest_items = items_from_manifest(input_path, load_manifest(manifest_path))
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, so the manifest was built but local VLM inference cannot run.")

    requested = {part.strip().lower() for part in args.models.split(",") if part.strip()}
    if "qwen_vl" in requested:
        run_model(
            "qwen_vl",
            args.qwen_vl_model,
            QwenVLFrameAnswerAdapter(model_name=args.qwen_vl_model),
            manifest_items,
            output_dir,
            manifest_path,
        )
        clear_cuda()
    if "internvl" in requested:
        run_model(
            "internvl",
            args.internvl_model,
            InternVLFrameAnswerAdapter(model_name=args.internvl_model),
            manifest_items,
            output_dir,
            manifest_path,
        )
        clear_cuda()
    if "molmo2" in requested:
        run_model(
            "molmo2",
            args.molmo2_model,
            Molmo2FrameAnswerAdapter(model_name=args.molmo2_model),
            manifest_items,
            output_dir,
            manifest_path,
        )
        clear_cuda()


if __name__ == "__main__":
    main()
