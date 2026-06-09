"""Caption-only benchmark runner for strict-valid aligned QA items."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from annotation_feature.pipeline.client import create_gemini_client, load_environment

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_INPUT_PATH = Path("outputs/aligned_qa_valid_items.json")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks")
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_JUDGE_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_OPENAI_MODEL_NAME = "gpt-5.4-mini"
DEFAULT_GEMINI_API_KEY_LIST_PATH = Path("api_key_list/gemini_api_key_list")
DEFAULT_OPENAI_API_KEY_LIST_PATH = Path("api_key_list/openai_api_key_list")
DEFAULT_API_KEY_LIST_PATH = DEFAULT_GEMINI_API_KEY_LIST_PATH
REQUIRED_QA_FIELDS = ("qa_id", "modality", "section", "pair_key", "question", "answer", "caption")
VALID_SCORES = {"correct", "partial", "incorrect"}
NUMERIC_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
COMPLETED_SCORES = {"correct", "partial", "incorrect"}
QUOTA_ERROR_PATTERNS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "Quota exceeded",
    "free_tier_requests",
    "API_KEY_INVALID",
    "API key not valid",
    "rate_limit_exceeded",
    "insufficient_quota",
    "invalid_api_key",
    "Incorrect API key",
    "401",
    "403",
)


class BenchmarkModelAdapter(ABC):
    """Answer caption-only benchmark questions."""

    provider: str
    model_name: str

    @abstractmethod
    def answer(self, item: dict[str, Any]) -> str:
        """Return a concise answer for one benchmark item."""


class BenchmarkJudge(ABC):
    """Judge model answers against gold answers."""

    @abstractmethod
    def judge(self, item: dict[str, Any], model_answer: str) -> dict[str, Any]:
        """Return normalized judgment fields for one benchmark item."""


class GeminiCaptionAdapter(BenchmarkModelAdapter):
    """Gemini caption-only benchmark adapter."""

    provider = "gemini"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        client: Any | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.client = client or create_gemini_client(api_key=api_key)

    def answer(self, item: dict[str, Any]) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[build_model_prompt(item)],
        )
        return str(getattr(response, "text", "")).strip()


class GeminiJudge(BenchmarkJudge):
    """Gemini judge for caption-grounded answer correctness."""

    def __init__(
        self,
        model_name: str = DEFAULT_JUDGE_MODEL_NAME,
        client: Any | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.client = client or create_gemini_client(api_key=api_key)

    def judge(self, item: dict[str, Any], model_answer: str) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[build_judge_prompt(item, model_answer)],
        )
        parsed = _parse_json_object(str(getattr(response, "text", "")))
        return normalize_judgment(parsed)


class OpenAICaptionAdapter(BenchmarkModelAdapter):
    """OpenAI caption-only benchmark adapter using the Responses API."""

    provider = "openai"

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL_NAME,
        client: Any | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        if client is not None:
            self.client = client
        else:
            if OpenAI is None:
                raise ImportError("The OpenAI SDK is not installed. Install dependencies from requirements.txt first.")
            load_environment()
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def answer(self, item: dict[str, Any]) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=build_model_prompt(item),
        )
        return _extract_openai_text(response).strip()


def load_valid_qa_items(input_path: Path | str = DEFAULT_INPUT_PATH) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_items = payload.get("valid_qa") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        return []

    items: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        if all(str(raw.get(field, "")).strip() for field in REQUIRED_QA_FIELDS):
            items.append({field: str(raw.get(field, "")).strip() for field in REQUIRED_QA_FIELDS})
    return items


def build_model_prompt(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are answering a caption-only video QA benchmark item.",
            "Use only the provided caption. Do not assume access to images, videos, audio, or outside knowledge.",
            "Return only a concise answer. Do not include explanation.",
            "",
            f"Modality: {item.get('modality', '')}",
            f"Section: {item.get('section', '')}",
            "",
            "Caption:",
            str(item.get("caption", "")).strip(),
            "",
            "Question:",
            str(item.get("question", "")).strip(),
        ]
    )


def build_judge_prompt(item: dict[str, Any], model_answer: str) -> str:
    compact = {
        "qa_id": item.get("qa_id"),
        "modality": item.get("modality"),
        "section": item.get("section"),
        "caption": item.get("caption"),
        "question": item.get("question"),
        "gold_answer": item.get("answer"),
        "model_answer": model_answer,
    }
    return (
        "You are judging a caption-only video QA benchmark answer. Use only the caption, question, "
        "gold answer, and model answer.\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        "{\n"
        '  "score": "correct|partial|incorrect",\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        "Scoring rules:\n"
        "- correct: the model answer is semantically equivalent to the gold answer and supported by the caption.\n"
        "- partial: the model answer is incomplete but mostly relevant and not contradictory.\n"
        "- incorrect: the model answer is wrong, contradicted, unsupported, too vague, or fails to answer.\n"
        "- Keep reason under 30 words.\n\n"
        f"Benchmark item:\n{json.dumps(compact, ensure_ascii=False)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in judge response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")
    return parsed


def _extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text is not None:
                chunks.append(str(text))
    return "".join(chunks)


def normalize_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    score = str(raw.get("score", "incorrect")).strip().lower()
    if score not in VALID_SCORES:
        score = "incorrect"
    return {
        "score": score,
        "numeric_score": NUMERIC_SCORES[score],
        "reason": str(raw.get("reason", "")).strip(),
    }


def is_quota_error(exc: BaseException | str) -> bool:
    text = str(exc)
    return any(pattern in text for pattern in QUOTA_ERROR_PATTERNS)


def load_api_keys(
    api_key_list_path: Path | str = DEFAULT_API_KEY_LIST_PATH,
    env_var_name: str = "GEMINI_API_KEY",
    key_prefixes: tuple[str, ...] = ("AIza",),
) -> list[str]:
    """Load provider API keys from a local ignored file without printing secrets."""
    api_key_list_path = Path(api_key_list_path)
    if not api_key_list_path.exists():
        return []

    keys: list[str] = []
    seen: set[str] = set()
    with open(api_key_list_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key_parts = [part for part in line.split() if any(part.startswith(prefix) for prefix in key_prefixes)]
            if key_parts:
                line = key_parts[-1]
            elif "=" in line:
                name, value = line.split("=", 1)
                if name.strip() != env_var_name:
                    continue
                line = value.strip()
            line = line.strip().strip("\"'")
            if line and line not in seen:
                keys.append(line)
                seen.add(line)
    return keys


def _masked_key_label(key: str | None, index: int, total: int) -> str:
    if not key:
        return f"key {index}/{total}"
    suffix = key[-4:] if len(key) >= 4 else "****"
    return f"key {index}/{total} (...{suffix})"


def _safe_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(model_name).strip()).strip("_")
    return cleaned or "model"


def _default_output_paths(output_dir: Path, model_name: str) -> tuple[Path, Path]:
    safe_name = _safe_model_name(model_name)
    return (
        output_dir / f"aligned_qa_benchmark_{safe_name}.json",
        output_dir / f"aligned_qa_benchmark_{safe_name}.csv",
    )


def _load_existing_results(output_json: Path) -> dict[str, dict[str, Any]]:
    if not output_json.exists():
        return {}
    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(raw_results, dict):
        return {
            str(qa_id): result
            for qa_id, result in raw_results.items()
            if isinstance(result, dict)
        }
    return {}


def _is_completed_result(result: dict[str, Any]) -> bool:
    if str(result.get("reason", "")).startswith("Benchmark call failed:"):
        return False
    return result.get("score") in COMPLETED_SCORES


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qa_id",
        "modality",
        "section",
        "pair_key",
        "question",
        "gold_answer",
        "model_answer",
        "score",
        "numeric_score",
        "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def compute_metrics(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [result for result in results.values() if isinstance(result, dict)]
    scored_rows = [row for row in rows if row.get("score") in COMPLETED_SCORES]
    failed_rows = [row for row in rows if row.get("score") == "failed"]
    total_scored = len(scored_rows)
    correct = sum(1 for row in scored_rows if row.get("score") == "correct")
    partial = sum(1 for row in scored_rows if row.get("score") == "partial")
    numeric_total = sum(float(row.get("numeric_score") or 0.0) for row in scored_rows)

    def summarize(grouped_rows: list[dict[str, Any]]) -> dict[str, Any]:
        group_total = len(grouped_rows)
        group_correct = sum(1 for row in grouped_rows if row.get("score") == "correct")
        group_numeric = sum(float(row.get("numeric_score") or 0.0) for row in grouped_rows)
        return {
            "total": group_total,
            "accuracy": round(group_correct / group_total, 6) if group_total else 0.0,
            "partial_adjusted_accuracy": round(group_numeric / group_total, 6) if group_total else 0.0,
        }

    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_modality[str(row.get("modality", "unknown"))].append(row)
        by_section[str(row.get("section", "unknown"))].append(row)

    return {
        "total_attempted": len(rows),
        "total_scored": total_scored,
        "total_failed": len(failed_rows),
        "total_evaluated": total_scored,
        "score_counts": dict(Counter(str(row.get("score", "unknown")) for row in rows)),
        "accuracy": round(correct / total_scored, 6) if total_scored else 0.0,
        "partial_adjusted_accuracy": round(numeric_total / total_scored, 6) if total_scored else 0.0,
        "correct": correct,
        "partial": partial,
        "incorrect": sum(1 for row in scored_rows if row.get("score") == "incorrect"),
        "failed": len(failed_rows),
        "by_modality": {key: summarize(value) for key, value in sorted(by_modality.items())},
        "by_section": {key: summarize(value) for key, value in sorted(by_section.items())},
    }


def create_benchmark_adapter(
    provider: str,
    model_name: str,
    api_key: str | None = None,
) -> BenchmarkModelAdapter:
    provider = str(provider or DEFAULT_PROVIDER).strip().lower()
    if provider == "gemini":
        return GeminiCaptionAdapter(model_name=model_name, api_key=api_key)
    if provider in {"chatgpt", "openai"}:
        return OpenAICaptionAdapter(model_name=model_name, api_key=api_key)
    if provider in {"qwen", "internvl"}:
        raise NotImplementedError(f"Benchmark adapter for provider '{provider}' is not implemented yet.")
    raise ValueError(f"Unknown benchmark provider: {provider}")


def _result_row(
    item: dict[str, Any],
    provider: str,
    model_name: str,
    model_answer: str,
    judgment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qa_id": item["qa_id"],
        "modality": item["modality"],
        "section": item["section"],
        "pair_key": item["pair_key"],
        "question": item["question"],
        "caption": item["caption"],
        "gold_answer": item["answer"],
        "provider": provider,
        "model_name": model_name,
        "model_answer": model_answer,
        "score": judgment["score"],
        "numeric_score": judgment["numeric_score"],
        "reason": judgment["reason"],
    }


def _save_outputs(
    output_json: Path,
    output_csv: Path,
    results_by_id: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    metrics = compute_metrics(results_by_id)
    payload = {
        "results": results_by_id,
        "metrics": metrics,
        "metadata": metadata,
    }
    _write_json(output_json, payload)
    _write_csv(output_csv, list(results_by_id.values()))


def _benchmark_metadata(
    input_path: Path,
    provider: str,
    model_name: str,
    judge_model_name: str,
    resume: bool,
    total_valid_items: int,
    results_by_id: dict[str, dict[str, Any]],
    batch_size: int | None = None,
    stopped_reason: str | None = None,
    key_rotation_enabled: bool = False,
    keys_available: int = 0,
    exhausted_key_count: int = 0,
    judge_key_rotation_enabled: bool = False,
    judge_keys_available: int = 0,
    exhausted_judge_key_count: int = 0,
) -> dict[str, Any]:
    completed_count = sum(1 for result in results_by_id.values() if _is_completed_result(result))
    pending_count = max(0, total_valid_items - completed_count)
    metadata = {
        "input_path": input_path.as_posix(),
        "provider": provider,
        "model_name": model_name,
        "judge_provider": "gemini",
        "judge_model_name": judge_model_name,
        "resume": resume,
        "total_valid_items": total_valid_items,
        "evaluated_items": completed_count,
        "pending_items": pending_count,
        "key_rotation_enabled": key_rotation_enabled,
        "keys_available": keys_available,
        "exhausted_key_count": exhausted_key_count,
        "judge_key_rotation_enabled": judge_key_rotation_enabled,
        "judge_keys_available": judge_keys_available,
        "exhausted_judge_key_count": exhausted_judge_key_count,
    }
    if batch_size is not None:
        metadata["batch_size"] = batch_size
    if stopped_reason:
        metadata["stopped_reason"] = stopped_reason
    return metadata


def repair_benchmark_failures(
    output_json: Path | str,
    output_csv: Path | str | None = None,
) -> dict[str, Any]:
    """Convert legacy benchmark call failures from incorrect to failed and recompute metrics."""
    output_json = Path(output_json)
    if output_csv is None:
        output_csv = output_json.with_suffix(".csv")
    output_csv = Path(output_csv)

    if not output_json.exists():
        return {"repaired_count": 0, "output_json": output_json, "output_csv": output_csv}

    with open(output_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results") if isinstance(payload, dict) else {}
    if not isinstance(results, dict):
        results = {}

    repaired_count = 0
    for result in results.values():
        if not isinstance(result, dict):
            continue
        reason = str(result.get("reason", ""))
        if result.get("score") == "incorrect" and reason.startswith("Benchmark call failed:"):
            result["score"] = "failed"
            result["numeric_score"] = None
            repaired_count += 1

    payload["results"] = results
    payload["metrics"] = compute_metrics(results)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        total_valid_items = int(metadata.get("total_valid_items") or len(results))
        completed_count = sum(1 for result in results.values() if _is_completed_result(result))
        metadata["evaluated_items"] = completed_count
        metadata["pending_items"] = max(0, total_valid_items - completed_count)
        metadata["failure_repair_applied"] = True
        metadata["failure_repair_count"] = repaired_count
    _write_json(output_json, payload)
    _write_csv(output_csv, list(results.values()))
    return {"repaired_count": repaired_count, "output_json": output_json, "output_csv": output_csv}


def run_aligned_qa_benchmark(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    provider: str = DEFAULT_PROVIDER,
    model_name: str = DEFAULT_MODEL_NAME,
    judge_model_name: str = DEFAULT_JUDGE_MODEL_NAME,
    max_items: int | None = 100,
    batch_size: int = 5,
    delay_between_batches: int = 30,
    resume: bool = True,
    api_key_list_path: Path | str = DEFAULT_API_KEY_LIST_PATH,
    openai_api_key_list_path: Path | str = DEFAULT_OPENAI_API_KEY_LIST_PATH,
    judge_api_key_list_path: Path | str = DEFAULT_GEMINI_API_KEY_LIST_PATH,
    enable_key_rotation: bool = True,
    adapter: BenchmarkModelAdapter | None = None,
    judge: BenchmarkJudge | None = None,
    adapter_factory: Callable[[str | None], BenchmarkModelAdapter] | None = None,
    judge_factory: Callable[[str | None], BenchmarkJudge] | None = None,
) -> dict[str, Path]:
    """Run a caption-only QA benchmark and score answers with a Gemini judge."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_json, output_csv = _default_output_paths(output_dir, model_name)
    provider = str(provider or DEFAULT_PROVIDER).strip().lower()
    batch_size = max(1, int(batch_size))
    if provider == "gemini":
        answer_key_label = "Gemini"
        answer_env_var = "GEMINI_API_KEY"
        answer_key_path = api_key_list_path
        answer_key_prefixes = ("AIza",)
    elif provider in {"openai", "chatgpt"}:
        answer_key_label = "OpenAI"
        answer_env_var = "OPENAI_API_KEY"
        answer_key_path = openai_api_key_list_path
        answer_key_prefixes = ("sk-",)
    else:
        answer_key_label = provider
        answer_env_var = ""
        answer_key_path = api_key_list_path
        answer_key_prefixes = ()

    api_keys = (
        load_api_keys(answer_key_path, env_var_name=answer_env_var, key_prefixes=answer_key_prefixes)
        if enable_key_rotation and answer_env_var
        else []
    )
    judge_api_keys = (
        load_api_keys(judge_api_key_list_path, env_var_name="GEMINI_API_KEY", key_prefixes=("AIza",))
        if enable_key_rotation
        else []
    )
    key_rotation_enabled = bool(api_keys)
    judge_key_rotation_enabled = bool(judge_api_keys)
    active_key_index = 0
    active_judge_key_index = 0
    exhausted_key_count = 0
    exhausted_judge_key_count = 0

    def current_api_key() -> str | None:
        if not key_rotation_enabled:
            return None
        return api_keys[active_key_index]

    def current_judge_api_key() -> str | None:
        if not judge_key_rotation_enabled:
            return None
        return judge_api_keys[active_judge_key_index]

    def build_adapter() -> BenchmarkModelAdapter:
        key = current_api_key()
        if key is not None:
            os.environ[answer_env_var] = key
        if adapter_factory is not None:
            return adapter_factory(key)
        return create_benchmark_adapter(provider, model_name, api_key=key)

    def build_judge() -> BenchmarkJudge:
        key = current_judge_api_key()
        if key is not None:
            os.environ["GEMINI_API_KEY"] = key
        if judge_factory is not None:
            return judge_factory(key)
        return GeminiJudge(model_name=judge_model_name, api_key=key)

    def make_metadata(stopped_reason: str | None = None) -> dict[str, Any]:
        return _benchmark_metadata(
            input_path,
            provider,
            model_name,
            judge_model_name,
            resume,
            len(items),
            results_by_id,
            batch_size=batch_size,
            stopped_reason=stopped_reason,
            key_rotation_enabled=key_rotation_enabled,
            keys_available=len(api_keys),
            exhausted_key_count=exhausted_key_count,
            judge_key_rotation_enabled=judge_key_rotation_enabled,
            judge_keys_available=len(judge_api_keys),
            exhausted_judge_key_count=exhausted_judge_key_count,
        )

    items = load_valid_qa_items(input_path)
    results_by_id = _load_existing_results(output_json) if resume else {}
    pending = [
        item
        for item in items
        if not _is_completed_result(results_by_id.get(item["qa_id"], {}))
    ]
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]

    print(
        f"Aligned QA benchmark resume scan: {len(results_by_id)} complete skipped, "
        f"{len(pending)} pending selected, {len(items)} valid total, provider={provider}, model={model_name}."
    )

    if not pending:
        metadata = make_metadata()
        _save_outputs(output_json, output_csv, results_by_id, metadata)
        return {"benchmark_json": output_json, "benchmark_csv": output_csv}

    if key_rotation_enabled:
        print(f"{answer_key_label} key rotation enabled: {len(api_keys)} key(s) loaded from {Path(answer_key_path)}.")
        print(f"Using {answer_key_label} API {_masked_key_label(api_keys[active_key_index], active_key_index + 1, len(api_keys))}.")
    if judge_key_rotation_enabled:
        print(f"Gemini judge key rotation enabled: {len(judge_api_keys)} key(s) loaded from {Path(judge_api_key_list_path)}.")
        print(
            "Using Gemini judge API "
            f"{_masked_key_label(judge_api_keys[active_judge_key_index], active_judge_key_index + 1, len(judge_api_keys))}."
        )

    adapter = adapter or build_adapter()
    judge = judge or build_judge()

    def rotate_answer_key_after_quota() -> bool:
        nonlocal active_key_index, exhausted_key_count, adapter
        if not key_rotation_enabled or active_key_index + 1 >= len(api_keys):
            exhausted_key_count = len(api_keys) if key_rotation_enabled else exhausted_key_count
            return False
        active_key_index += 1
        exhausted_key_count = active_key_index
        print(
            f"{answer_key_label} quota/rate/key limit reached for key {exhausted_key_count}/{len(api_keys)}. "
            f"Switching to key {active_key_index + 1}/{len(api_keys)} and retrying current item."
        )
        _save_outputs(output_json, output_csv, results_by_id, make_metadata(stopped_reason="rotating_after_quota"))
        adapter = build_adapter()
        print(f"{answer_key_label} API key changed to {_masked_key_label(api_keys[active_key_index], active_key_index + 1, len(api_keys))}.")
        return True

    def rotate_judge_key_after_quota() -> bool:
        nonlocal active_judge_key_index, exhausted_judge_key_count, judge
        if not judge_key_rotation_enabled or active_judge_key_index + 1 >= len(judge_api_keys):
            exhausted_judge_key_count = len(judge_api_keys) if judge_key_rotation_enabled else exhausted_judge_key_count
            return False
        active_judge_key_index += 1
        exhausted_judge_key_count = active_judge_key_index
        print(
            f"Gemini judge quota/rate/key limit reached for key {exhausted_judge_key_count}/{len(judge_api_keys)}. "
            f"Switching to key {active_judge_key_index + 1}/{len(judge_api_keys)} and retrying judgment."
        )
        _save_outputs(output_json, output_csv, results_by_id, make_metadata(stopped_reason="rotating_judge_after_quota"))
        judge = build_judge()
        print(
            "Gemini judge API key changed to "
            f"{_masked_key_label(judge_api_keys[active_judge_key_index], active_judge_key_index + 1, len(judge_api_keys))}."
        )
        return True

    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            print(f"Running benchmark batch {start // batch_size + 1}: {len(batch)} item(s)")
            for item in batch:
                model_answer = ""
                judgment = {
                    "score": "failed",
                    "numeric_score": None,
                    "reason": "Benchmark call failed: empty model answer",
                }
                while True:
                    try:
                        model_answer = adapter.answer(item)
                        break
                    except Exception as exc:
                        if is_quota_error(exc):
                            if rotate_answer_key_after_quota():
                                continue
                            print(f"\n{answer_key_label} quota/rate/key limit reached and no more keys are available.")
                            print("Progress saved. Run this benchmark option again later to resume.")
                            _save_outputs(
                                output_json,
                                output_csv,
                                results_by_id,
                                make_metadata(stopped_reason="quota_or_rate_limit"),
                            )
                            return {"benchmark_json": output_json, "benchmark_csv": output_csv}
                        model_answer = ""
                        judgment = {
                            "score": "failed",
                            "numeric_score": None,
                            "reason": f"Benchmark call failed: {exc}",
                        }
                        break

                if model_answer:
                    while True:
                        try:
                            judgment = judge.judge(item, model_answer)
                            break
                        except Exception as exc:
                            if is_quota_error(exc):
                                if rotate_judge_key_after_quota():
                                    continue
                                print("\nGemini judge quota/rate/key limit reached and no more judge keys are available.")
                                print("Progress saved. Run this benchmark option again later to resume.")
                                _save_outputs(
                                    output_json,
                                    output_csv,
                                    results_by_id,
                                    make_metadata(stopped_reason="judge_quota_or_rate_limit"),
                                )
                                return {"benchmark_json": output_json, "benchmark_csv": output_csv}
                            judgment = {
                                "score": "failed",
                                "numeric_score": None,
                                "reason": f"Benchmark call failed: {exc}",
                            }
                            break
                results_by_id[item["qa_id"]] = _result_row(
                    item,
                    provider=provider,
                    model_name=model_name,
                    model_answer=model_answer,
                    judgment=judgment,
                )
                _save_outputs(output_json, output_csv, results_by_id, make_metadata())
                print(f"Checkpoint saved: {len(results_by_id)} benchmark item(s)")

            if delay_between_batches > 0 and start + batch_size < len(pending):
                time.sleep(delay_between_batches)
    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user. Progress saved.")
        _save_outputs(output_json, output_csv, results_by_id, make_metadata(stopped_reason="user_cancelled"))
        return {"benchmark_json": output_json, "benchmark_csv": output_csv}

    return {"benchmark_json": output_json, "benchmark_csv": output_csv}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--judge-model-name", default=DEFAULT_JUDGE_MODEL_NAME)
    parser.add_argument("--max-items", type=int, default=100, help="Use 0 to run all remaining items.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--delay-between-batches", type=int, default=30)
    parser.add_argument("--api-key-list", default=str(DEFAULT_API_KEY_LIST_PATH))
    parser.add_argument("--openai-api-key-list", default=str(DEFAULT_OPENAI_API_KEY_LIST_PATH))
    parser.add_argument("--judge-api-key-list", default=str(DEFAULT_GEMINI_API_KEY_LIST_PATH))
    parser.add_argument("--disable-key-rotation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--repair-failures", action="store_true", help="Repair legacy failed-call rows in the output file.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    max_items = None if args.max_items == 0 else args.max_items
    if args.repair_failures:
        output_json, output_csv = _default_output_paths(Path(args.output_dir), args.model_name)
        repaired = repair_benchmark_failures(output_json, output_csv)
        print(f"Repaired {repaired['repaired_count']} failed benchmark row(s).")
        print(f"benchmark_json: {repaired['output_json']}")
        print(f"benchmark_csv: {repaired['output_csv']}")
        return
    outputs = run_aligned_qa_benchmark(
        input_path=args.input,
        output_dir=args.output_dir,
        provider=args.provider,
        model_name=args.model_name,
        judge_model_name=args.judge_model_name,
        max_items=max_items,
        batch_size=args.batch_size,
        delay_between_batches=args.delay_between_batches,
        api_key_list_path=args.api_key_list,
        openai_api_key_list_path=args.openai_api_key_list,
        judge_api_key_list_path=args.judge_api_key_list,
        enable_key_rotation=not args.disable_key_rotation,
        resume=not args.no_resume,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
