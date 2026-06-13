"""Caption-only benchmark runner for strict-valid aligned QA items."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from annotation_feature.pipeline.client import create_gemini_client, load_environment
from annotation_feature.pipeline.utils import build_image_parts, encode_frames_to_base64

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
except ImportError:
    torch = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )
except ImportError:
    AutoConfig = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    PreTrainedModel = None
    Qwen2_5_VLForConditionalGeneration = None
    Qwen3VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


DEFAULT_INPUT_PATH = Path("outputs/aligned_qa_valid_items.json")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks")
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_JUDGE_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_OPENAI_MODEL_NAME = "gpt-5.4-mini"
DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_QWEN_ENGINE = "transformers_4bit"
DEFAULT_QWEN_MAX_TOKENS = 128
DEFAULT_QWEN_MAX_MODEL_LEN = 1024
DEFAULT_QWEN_GPU_MEMORY_UTILIZATION = 0.7
DEFAULT_QWEN_ENFORCE_EAGER = True
DEFAULT_QWEN_DTYPE = "half"
DEFAULT_QWEN_MAX_NUM_SEQS = 1
DEFAULT_QWEN_MAX_CAPTION_CHARS = 3000
DEFAULT_QWEN_MAX_QUESTION_CHARS = 600
DEFAULT_QWEN_VL_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_VL_MAX_TOKENS = 128
DEFAULT_INTERNVL_MODEL_NAME = "OpenGVLab/InternVL2_5-4B"
DEFAULT_INTERNVL_REVISION = ""
DEFAULT_INTERNVL_MAX_TOKENS = 128
DEFAULT_INTERNVL_IMAGE_SIZE = 448
DEFAULT_INTERNVL_MAX_NUM_TILES = 12
DEFAULT_INTERNVL_TOKENIZER_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INTERNVL_ADDITIONAL_SPECIAL_TOKENS = (
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<img>",
    "</img>",
    "<IMG_CONTEXT>",
    "<quad>",
    "</quad>",
    "<ref>",
    "</ref>",
    "<box>",
    "</box>",
)
DEFAULT_FRAME_CACHE_ROOT = Path("aligned_dataset")
DEFAULT_FRAME_MAX_FRAMES_PER_ITEM = 6
FRAME_ANSWER_BENCHMARK_TYPE = "frame_input_answer_generation"
FRAME_CACHE_SUBDIRS = {
    "rgb": ".frames_cache",
    "ir": ".frames_cache_ir",
    "event": ".frames_cache_event",
    "depth": ".frames_cache_marigold",
}
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


class GeminiFrameAnswerAdapter:
    """Gemini answer adapter that sees cached frames instead of captions."""

    provider = "gemini"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        client: Any | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.client = client or create_gemini_client(api_key=api_key)

    def answer(self, item: dict[str, Any], frame_paths: list[Path]) -> str:
        encoded_frames = encode_frames_to_base64(frame_paths)
        image_parts = build_image_parts(encoded_frames)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[*image_parts, build_frame_answer_prompt(item, frame_paths)],
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


class QwenLocalCaptionAdapter(BenchmarkModelAdapter):
    """Local Qwen caption-only benchmark adapter."""

    provider = "qwen"

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL_NAME,
        engine: str = DEFAULT_QWEN_ENGINE,
        llm: Any | None = None,
        sampling_params: Any | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        require_cuda: bool = True,
        max_tokens: int = DEFAULT_QWEN_MAX_TOKENS,
        max_model_len: int = DEFAULT_QWEN_MAX_MODEL_LEN,
        gpu_memory_utilization: float = DEFAULT_QWEN_GPU_MEMORY_UTILIZATION,
        enforce_eager: bool = DEFAULT_QWEN_ENFORCE_EAGER,
        dtype: str = DEFAULT_QWEN_DTYPE,
        max_num_seqs: int = DEFAULT_QWEN_MAX_NUM_SEQS,
        cleanup_stale_workers: bool = True,
    ):
        self.model_name = model_name
        self.engine = engine
        self.max_tokens = max(1, int(max_tokens))
        self.max_model_len = max(512, int(max_model_len))
        self.gpu_memory_utilization = min(0.99, max(0.1, float(gpu_memory_utilization)))
        self.enforce_eager = bool(enforce_eager)
        self.dtype = str(dtype or DEFAULT_QWEN_DTYPE)
        self.max_num_seqs = max(1, int(max_num_seqs))

        if self.engine not in {"transformers_4bit", "vllm"}:
            raise NotImplementedError(
                "Only transformers_4bit and vllm are implemented for local Qwen benchmark generation."
            )

        if self.engine == "vllm" and llm is not None:
            self.llm = llm
            self.sampling_params = sampling_params
            return
        if self.engine == "transformers_4bit" and model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            return

        self._validate_runtime(engine=self.engine, require_cuda=require_cuda)
        if cleanup_stale_workers:
            cleanup_stale_qwen_workers()
        if self.engine == "vllm":
            self._load_vllm()
        else:
            self._load_transformers_4bit()

    def _load_vllm(self) -> None:
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens)
        try:
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
                dtype=self.dtype,
                max_num_seqs=self.max_num_seqs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local Qwen model '{self.model_name}'. "
                "Check Hugging Face access and GPU memory. On a 16GB GPU, close other GPU processes "
                "and keep the benchmark context length small."
            ) from exc

    def _load_transformers_4bit(self) -> None:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
            )
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local Qwen model '{self.model_name}' in 4-bit mode. "
                "Check that bitsandbytes is installed, CUDA is available, and enough GPU memory is free."
            ) from exc

    @staticmethod
    def _validate_runtime(engine: str = DEFAULT_QWEN_ENGINE, require_cuda: bool = True) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install project requirements before running local Qwen.")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Local Qwen 8B benchmark requires GPU execution; "
                "fix NVIDIA driver/CUDA visibility first."
            )
        if engine == "vllm" and (LLM is None or SamplingParams is None):
            raise RuntimeError("vLLM is not installed. Install vllm before running local Qwen.")
        if engine == "transformers_4bit" and (
            AutoModelForCausalLM is None or AutoTokenizer is None or BitsAndBytesConfig is None
        ):
            raise RuntimeError(
                "Transformers 4-bit Qwen requires transformers and bitsandbytes. "
                "Install project requirements before running local Qwen."
            )

    @staticmethod
    def runtime_summary() -> str:
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        gpu_name = "none"
        if cuda_available:
            try:
                gpu_name = str(torch.cuda.get_device_name(0))
            except Exception:
                gpu_name = "visible"
        vllm_available = LLM is not None and SamplingParams is not None
        transformers_4bit_available = (
            AutoModelForCausalLM is not None and AutoTokenizer is not None and BitsAndBytesConfig is not None
        )
        return (
            f"engine={DEFAULT_QWEN_ENGINE}, cuda_available={cuda_available}, gpu={gpu_name}, "
            f"transformers_4bit_available={transformers_4bit_available}, vllm_available={vllm_available}, "
            f"max_model_len={DEFAULT_QWEN_MAX_MODEL_LEN}, gpu_memory_utilization={DEFAULT_QWEN_GPU_MEMORY_UTILIZATION}, "
            f"max_tokens={DEFAULT_QWEN_MAX_TOKENS}, dtype={DEFAULT_QWEN_DTYPE}, "
            f"max_num_seqs={DEFAULT_QWEN_MAX_NUM_SEQS}, enforce_eager={DEFAULT_QWEN_ENFORCE_EAGER}, "
            f"cuda_alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}"
        )

    def answer(self, item: dict[str, Any]) -> str:
        if self.engine == "transformers_4bit":
            return self._answer_transformers_4bit(item)
        return self._answer_vllm(item)

    def _answer_vllm(self, item: dict[str, Any]) -> str:
        outputs = self.llm.generate([build_model_prompt(item, for_qwen=True)], self.sampling_params)
        if not outputs:
            return ""
        first_output = outputs[0]
        completions = getattr(first_output, "outputs", []) or []
        if not completions:
            return ""
        return str(getattr(completions[0], "text", "")).strip()

    def _answer_transformers_4bit(self, item: dict[str, Any]) -> str:
        prompt = build_model_prompt(item, for_qwen=True)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len,
        )
        model_device = _model_input_device(self.model)
        if model_device is not None:
            encoded = _move_tokenizer_output_to_device(encoded, model_device)
        input_length = _tokenizer_input_length(encoded)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        sequence = _first_generated_sequence(generated)
        new_tokens = sequence[input_length:] if input_length else sequence
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


class QwenVLFrameAnswerAdapter:
    """Local Qwen-VL frame-input answer adapter."""

    provider = "qwen_vl"

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_VL_MODEL_NAME,
        model: Any | None = None,
        processor: Any | None = None,
        require_cuda: bool = True,
        max_tokens: int = DEFAULT_QWEN_VL_MAX_TOKENS,
        cleanup_stale_workers: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max(1, int(max_tokens))
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            return

        self._validate_runtime(model_name=self.model_name, require_cuda=require_cuda)
        if cleanup_stale_workers:
            cleanup_stale_qwen_workers()
        self._load_model()

    @staticmethod
    def _validate_runtime(model_name: str = DEFAULT_QWEN_VL_MODEL_NAME, require_cuda: bool = True) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install project requirements before running local Qwen-VL.")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Local Qwen-VL frame benchmark requires GPU execution; "
                "fix NVIDIA driver/CUDA visibility first."
            )
        if AutoProcessor is None or BitsAndBytesConfig is None:
            raise RuntimeError("Qwen-VL requires transformers and bitsandbytes. Install project requirements first.")
        if process_vision_info is None:
            raise RuntimeError("Qwen-VL requires qwen-vl-utils. Install project requirements first.")
        if _qwen_vl_model_class(model_name) is None:
            raise RuntimeError("Installed Transformers does not expose a supported Qwen-VL model class.")

    @staticmethod
    def runtime_summary(model_name: str = DEFAULT_QWEN_VL_MODEL_NAME) -> str:
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        gpu_name = "none"
        if cuda_available:
            try:
                gpu_name = str(torch.cuda.get_device_name(0))
            except Exception:
                gpu_name = "visible"
        return (
            f"provider=qwen_vl, model={model_name}, cuda_available={cuda_available}, gpu={gpu_name}, "
            f"transformers_vl_available={_qwen_vl_model_class(model_name) is not None}, "
            f"auto_processor_available={AutoProcessor is not None}, "
            f"qwen_vl_utils_available={process_vision_info is not None}, "
            f"max_tokens={DEFAULT_QWEN_VL_MAX_TOKENS}, quantization=4bit_nf4, "
            f"cuda_alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}"
        )

    def _load_model(self) -> None:
        model_class = _qwen_vl_model_class(self.model_name)
        if model_class is None:
            raise RuntimeError(f"No supported Qwen-VL model class is available for '{self.model_name}'.")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = model_class.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
            )
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local Qwen-VL model '{self.model_name}' in 4-bit mode. "
                "Check qwen-vl-utils, Transformers support, CUDA availability, and free GPU memory."
            ) from exc

    def answer(self, item: dict[str, Any], frame_paths: list[Path]) -> str:
        messages = build_qwen_vl_frame_messages(item, frame_paths)
        return self._answer_messages(messages)

    def _answer_messages(self, messages: list[dict[str, Any]]) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vision_result = process_vision_info(messages, return_video_kwargs=True)
        if len(vision_result) == 3:
            image_inputs, video_inputs, video_kwargs = vision_result
        else:
            image_inputs, video_inputs = vision_result
            video_kwargs = {}
        if not video_inputs:
            video_kwargs = {}
        else:
            video_kwargs = {
                key: value
                for key, value in (video_kwargs or {}).items()
                if not isinstance(value, list) or value
            }
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        model_device = _model_input_device(self.model)
        if model_device is not None:
            inputs = _move_tokenizer_output_to_device(inputs, model_device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
        generated_trimmed = _trim_generated_batch(generated, input_ids)
        decoded = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0] if decoded else "").strip()


class QwenVLVideoAnswerAdapter(QwenVLFrameAnswerAdapter):
    """Local Qwen-VL video-input answer adapter."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_VL_MODEL_NAME,
        model: Any | None = None,
        processor: Any | None = None,
        require_cuda: bool = True,
        max_tokens: int = DEFAULT_QWEN_VL_MAX_TOKENS,
        cleanup_stale_workers: bool = True,
        video_fps: float = 1.0,
    ):
        self.video_fps = max(0.0, float(video_fps))
        super().__init__(
            model_name=model_name,
            model=model,
            processor=processor,
            require_cuda=require_cuda,
            max_tokens=max_tokens,
            cleanup_stale_workers=cleanup_stale_workers,
        )

    def answer(self, item: dict[str, Any], video_path: Path) -> str:
        video_frames, raw_fps = _sample_video_frames_with_opencv(video_path, self.video_fps)
        messages = build_qwen_vl_video_messages(
            item,
            video_path,
            video_fps=self.video_fps,
            video_frames=video_frames,
            raw_fps=raw_fps,
        )
        return self._answer_messages(messages)


class InternVLFrameAnswerAdapter:
    """Local InternVL 4B frame-input answer adapter."""

    provider = "internvl"

    def __init__(
        self,
        model_name: str = DEFAULT_INTERNVL_MODEL_NAME,
        model: Any | None = None,
        tokenizer: Any | None = None,
        require_cuda: bool = True,
        max_tokens: int = DEFAULT_INTERNVL_MAX_TOKENS,
        image_size: int = DEFAULT_INTERNVL_IMAGE_SIZE,
        max_num_tiles: int = DEFAULT_INTERNVL_MAX_NUM_TILES,
        revision: str | None = DEFAULT_INTERNVL_REVISION,
    ):
        self.model_name = model_name
        self.revision = str(revision or "").strip() or None
        self.max_tokens = max(1, int(max_tokens))
        self.image_size = max(1, int(image_size))
        self.max_num_tiles = max(1, int(max_num_tiles))
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            return

        self._validate_runtime(require_cuda=require_cuda)
        self._load_model()

    @staticmethod
    def _validate_runtime(require_cuda: bool = True) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install project requirements before running local InternVL 4B.")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Local InternVL 4B frame benchmark requires GPU execution; "
                "fix NVIDIA driver/CUDA visibility first."
            )
        if Image is None:
            raise RuntimeError("InternVL 4B requires Pillow. Install project requirements first.")
        if AutoConfig is None or AutoModel is None or AutoTokenizer is None or BitsAndBytesConfig is None:
            raise RuntimeError("InternVL 4B requires transformers and bitsandbytes. Install project requirements first.")

    @staticmethod
    def runtime_summary(
        model_name: str = DEFAULT_INTERNVL_MODEL_NAME,
        revision: str | None = DEFAULT_INTERNVL_REVISION,
    ) -> str:
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        gpu_name = "none"
        if cuda_available:
            try:
                gpu_name = str(torch.cuda.get_device_name(0))
            except Exception:
                gpu_name = "visible"
        return (
            f"provider=internvl, model={model_name}, cuda_available={cuda_available}, gpu={gpu_name}, "
            f"revision={str(revision or '').strip() or '<default>'}, "
            f"auto_model_available={AutoModel is not None}, auto_tokenizer_available={AutoTokenizer is not None}, "
            f"pillow_available={Image is not None}, max_tokens={DEFAULT_INTERNVL_MAX_TOKENS}, "
            f"quantization=8bit, image_size={DEFAULT_INTERNVL_IMAGE_SIZE}, "
            f"max_num_tiles={DEFAULT_INTERNVL_MAX_NUM_TILES}, "
            f"cuda_alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}"
        )

    def _load_tokenizer(self) -> Any:
        revision_kwargs = {"revision": self.revision} if self.revision else {}
        try:
            return AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False,
                **revision_kwargs,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "backend tokenizer" not in message and "sentencepiece" not in message and "tiktoken" not in message:
                raise

            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **revision_kwargs,
            )
            llm_config = getattr(config, "llm_config", None)
            tokenizer_model = getattr(llm_config, "_name_or_path", None) or DEFAULT_INTERNVL_TOKENIZER_BASE_MODEL
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model,
                trust_remote_code=True,
                use_fast=False,
            )
            tokenizer.add_special_tokens(
                {"additional_special_tokens": list(DEFAULT_INTERNVL_ADDITIONAL_SPECIAL_TOKENS)}
            )
            return tokenizer

    def _load_model(self) -> None:
        revision_kwargs = {"revision": self.revision} if self.revision else {}
        try:
            self.tokenizer = self._load_tokenizer()
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            _ensure_transformers_tied_weights_compatibility()
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                device_map="auto",
                **revision_kwargs,
            )
            if "all_tied_weights_keys" not in vars(self.model):
                self.model.all_tied_weights_keys = {}
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as exc:
            cause = f"{type(exc).__name__}: {exc}"
            tokenizer_hint = ""
            if "sentencepiece" in str(exc).lower() or "tiktoken" in str(exc).lower():
                tokenizer_hint = (
                    " This model tokenizer requires SentencePiece/tiktoken support; install project requirements "
                    "again or run `pip install sentencepiece tiktoken` in the active environment."
                )
            raise RuntimeError(
                f"Failed to load local InternVL 4B model '{self.model_name}' in 8-bit mode. "
                "Check Transformers trust_remote_code support, CUDA availability, bitsandbytes, and free GPU memory. "
                f"Original error: {cause}.{tokenizer_hint}"
            ) from exc

    def answer(self, item: dict[str, Any], frame_paths: list[Path]) -> str:
        pixel_values, num_patches_list = load_internvl_pixel_values(
            frame_paths,
            image_size=self.image_size,
            max_num_tiles=self.max_num_tiles,
        )
        model_device = _model_input_device(self.model)
        if model_device is not None and hasattr(pixel_values, "to"):
            pixel_values = pixel_values.to(model_device)
        prompt = build_internvl_frame_prompt(item, frame_paths)
        generation_config = {"max_new_tokens": self.max_tokens, "do_sample": False}
        answer = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        if isinstance(answer, tuple):
            answer = answer[0]
        return str(answer).strip()


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


def cleanup_stale_qwen_workers() -> None:
    """Clear stale vLLM worker processes without killing the current Python benchmark."""
    for pattern in ("EngineCore", "vllm"):
        subprocess.run(
            ["pkill", "-9", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _limit_text(text: Any, max_chars: int | None) -> str:
    cleaned = str(text or "").strip()
    if max_chars is None or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _move_tokenizer_output_to_device(encoded: Any, device: Any) -> Any:
    if hasattr(encoded, "to"):
        return encoded.to(device)
    if isinstance(encoded, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
    return encoded


def _tokenizer_input_length(encoded: Any) -> int:
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
    shape = getattr(input_ids, "shape", None)
    if shape:
        return int(shape[-1])
    if isinstance(input_ids, list):
        first = input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
        return len(first)
    return 0


def _first_generated_sequence(generated: Any) -> Any:
    if hasattr(generated, "shape") and len(getattr(generated, "shape", ())) > 1:
        return generated[0]
    if isinstance(generated, list) and generated and isinstance(generated[0], list):
        return generated[0]
    return generated


def _trim_generated_batch(generated: Any, input_ids: Any) -> Any:
    if input_ids is None:
        return generated
    try:
        return [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, generated)
        ]
    except Exception:
        input_length = _tokenizer_input_length({"input_ids": input_ids})
        sequence = _first_generated_sequence(generated)
        return [sequence[input_length:] if input_length else sequence]


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _qwen_vl_model_class(model_name: str) -> Any | None:
    lowered = str(model_name).lower()
    if "qwen3-vl" in lowered and Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration
    if "qwen2.5-vl" in lowered and Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText
    return Qwen3VLForConditionalGeneration or Qwen2_5_VLForConditionalGeneration


def _ensure_transformers_tied_weights_compatibility() -> None:
    if PreTrainedModel is not None and not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def _frame_answer_output_paths(output_dir: Path, model_name: str) -> tuple[Path, Path]:
    safe_name = _safe_model_name(model_name)
    return (
        output_dir / f"aligned_qa_frame_answers_{safe_name}.json",
        output_dir / f"aligned_qa_frame_answers_{safe_name}.csv",
    )


def _video_answer_output_paths(output_dir: Path, model_name: str) -> tuple[Path, Path]:
    safe_name = _safe_model_name(model_name)
    return (
        output_dir / f"aligned_qa_video_answers_{safe_name}.json",
        output_dir / f"aligned_qa_video_answers_{safe_name}.csv",
    )


def _frame_number(frame_path: Path) -> int:
    match = re.search(r"frame_(\d+)", frame_path.stem)
    return int(match.group(1)) if match else -1


def _frame_reference_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\bframe[_\s-]*(\d{3,6})\b", str(text), flags=re.I):
        number = int(match.group(1))
        if number not in seen:
            numbers.append(number)
            seen.add(number)
    return numbers


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def evenly_sample_frames(frames: list[Path], count: int) -> list[Path]:
    frames = sorted(_dedupe_paths(frames), key=lambda path: (_frame_number(path), path.as_posix()))
    if count <= 0 or len(frames) <= count:
        return frames
    if count == 1:
        return [frames[0]]
    indices = [round(index * (len(frames) - 1) / (count - 1)) for index in range(count)]
    return _dedupe_paths([frames[index] for index in indices])


def select_frames_for_question(
    frames: list[Path],
    question: str,
    max_frames_per_item: int | None = DEFAULT_FRAME_MAX_FRAMES_PER_ITEM,
) -> list[Path]:
    frames = sorted(_dedupe_paths(frames), key=lambda path: (_frame_number(path), path.as_posix()))
    if max_frames_per_item is None or max_frames_per_item == 0:
        return frames
    frame_limit = max(1, int(max_frames_per_item))
    wanted_numbers = set(_frame_reference_numbers(question))
    exact = [frame for frame in frames if _frame_number(frame) in wanted_numbers]
    exact = exact[:frame_limit]
    remaining_slots = frame_limit - len(exact)
    if remaining_slots <= 0:
        return exact
    exact_keys = {frame.as_posix() for frame in exact}
    remaining = [frame for frame in frames if frame.as_posix() not in exact_keys]
    return _dedupe_paths([*exact, *evenly_sample_frames(remaining, remaining_slots)])


def _pair_key_cache_segment(pair_key: str) -> Path:
    parts = Path(str(pair_key)).parts
    if parts and parts[0] == DEFAULT_FRAME_CACHE_ROOT.name:
        parts = parts[1:]
    if len(parts) > 1:
        parts = parts[:-1]
    return Path(*parts) if parts else Path()


def _cache_frame_pattern(modality: str) -> str:
    return "frame_*_depth.png" if modality == "depth" else "frame_*.png"


def resolve_frame_cache_candidates(
    item: dict[str, Any],
    frame_cache_root: Path | str = DEFAULT_FRAME_CACHE_ROOT,
) -> list[Path]:
    modality = str(item.get("modality", "")).strip().lower()
    cache_subdir = FRAME_CACHE_SUBDIRS.get(modality)
    if not cache_subdir:
        return []
    segment = _pair_key_cache_segment(str(item.get("pair_key", "")))
    cache_segment = Path(frame_cache_root) / cache_subdir / segment
    if not cache_segment.exists():
        return []

    pattern = _cache_frame_pattern(modality)
    leaf = Path(str(item.get("pair_key", ""))).name.lower()
    direct_dirs = [path for path in cache_segment.iterdir() if path.is_dir()]
    if modality == "depth":
        candidate_dirs = direct_dirs or [cache_segment]
    else:
        candidate_dirs = [
            path
            for path in direct_dirs
            if path.name.lower() == leaf
            or leaf in path.name.lower()
            or modality in path.name.lower().split("_")
        ]
        if not candidate_dirs:
            candidate_dirs = direct_dirs or [cache_segment]

    frames: list[Path] = []
    for candidate_dir in candidate_dirs:
        frames.extend(sorted(candidate_dir.rglob(pattern)))
    return sorted(_dedupe_paths(frames), key=lambda path: (_frame_number(path), path.as_posix()))


def resolve_frame_inputs_for_item(
    item: dict[str, Any],
    frame_cache_root: Path | str = DEFAULT_FRAME_CACHE_ROOT,
    max_frames_per_item: int | None = DEFAULT_FRAME_MAX_FRAMES_PER_ITEM,
) -> list[Path]:
    candidates = resolve_frame_cache_candidates(item, frame_cache_root=frame_cache_root)
    return select_frames_for_question(candidates, str(item.get("question", "")), max_frames_per_item)


def _segment_folder_from_pair_key(pair_key: str, dataset_root: Path | str = DEFAULT_FRAME_CACHE_ROOT) -> Path:
    parts = Path(str(pair_key)).parts
    dataset_root = Path(dataset_root)
    if parts and parts[0] == dataset_root.name:
        return dataset_root.joinpath(*parts[1:-1])
    return dataset_root / Path(*parts[:-1]) if len(parts) > 1 else dataset_root


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


def _is_matching_modality_video(path: Path, modality: str) -> bool:
    tokens = path.stem.lower().split("_")
    modality = str(modality).strip().lower()
    if modality == "rgb" and "with_audio" in path.stem.lower():
        return False
    return modality in tokens


def _video_preference_key(path: Path) -> tuple[int, str]:
    stem = path.stem.lower()
    side_rank = 2
    if "day" in stem or "with_light" in stem:
        side_rank = 0
    elif "night" in stem or "no_light" in stem or "cloudy_no_light" in stem:
        side_rank = 1
    return (side_rank, path.as_posix())


def resolve_video_input_for_item(
    item: dict[str, Any],
    dataset_root: Path | str = DEFAULT_FRAME_CACHE_ROOT,
) -> Path | None:
    modality = str(item.get("modality", "")).strip().lower()
    segment_folder = _segment_folder_from_pair_key(str(item.get("pair_key", "")), dataset_root=dataset_root)
    if not segment_folder.exists():
        return None
    candidates = [
        path
        for path in segment_folder.iterdir()
        if path.is_file() and _is_video_file(path) and _is_matching_modality_video(path, modality)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_video_preference_key)[0]


def build_model_prompt(item: dict[str, Any], for_qwen: bool = False) -> str:
    caption = _limit_text(
        item.get("caption", ""),
        DEFAULT_QWEN_MAX_CAPTION_CHARS if for_qwen else None,
    )
    question = _limit_text(
        item.get("question", ""),
        DEFAULT_QWEN_MAX_QUESTION_CHARS if for_qwen else None,
    )
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
            caption,
            "",
            "Question:",
            question,
        ]
    )


def build_frame_answer_prompt(item: dict[str, Any], frame_paths: list[Path]) -> str:
    frame_names = [path.name for path in frame_paths]
    return "\n".join(
        [
            "You are answering a video QA benchmark item using only the provided image frames.",
            "Do not assume access to captions, audio, hidden metadata, or outside knowledge.",
            "Return only a concise answer. Do not include explanation.",
            "",
            f"Modality: {item.get('modality', '')}",
            f"Section: {item.get('section', '')}",
            f"Pair key: {item.get('pair_key', '')}",
            f"Provided frames: {', '.join(frame_names)}",
            "",
            "Question:",
            str(item.get("question", "")).strip(),
        ]
    )


def _sample_video_frames_with_opencv(video_path: Path, sample_fps: float) -> tuple[list[Any], float]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for option 72 video decoding. Install opencv-python-headless.")
    if Image is None:
        raise RuntimeError("Pillow is required for option 72 video decoding. Install pillow.")

    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if raw_fps <= 0:
        raw_fps = 30.0
    if sample_fps and sample_fps > 0:
        frame_interval = max(1, int(round(raw_fps / float(sample_fps))))
    else:
        frame_interval = 1

    frames: list[Any] = []
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video file: {video_path}")
    return frames, raw_fps


def build_video_answer_prompt(item: dict[str, Any], video_path: Path) -> str:
    return "\n".join(
        [
            "You are answering a video QA benchmark item using only the provided video.",
            "Do not assume access to captions, audio, hidden metadata, or outside knowledge.",
            "Return only a concise answer. Do not include explanation.",
            "",
            f"Modality: {item.get('modality', '')}",
            f"Section: {item.get('section', '')}",
            f"Pair key: {item.get('pair_key', '')}",
            f"Provided video: {video_path.name}",
            "",
            "Question:",
            str(item.get("question", "")).strip(),
        ]
    )


def build_qwen_vl_frame_messages(item: dict[str, Any], frame_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": path.resolve().as_posix()}
        for path in frame_paths
    ]
    content.append({"type": "text", "text": build_frame_answer_prompt(item, frame_paths)})
    return [{"role": "user", "content": content}]


def build_internvl_frame_prompt(item: dict[str, Any], frame_paths: list[Path]) -> str:
    frame_markers = [f"Frame-{index}: <image>" for index, _ in enumerate(frame_paths, start=1)]
    return "\n".join([*frame_markers, build_frame_answer_prompt(item, frame_paths)])


def _internvl_target_ratios(max_num_tiles: int) -> list[tuple[int, int]]:
    ratios: set[tuple[int, int]] = set()
    for blocks in range(1, max_num_tiles + 1):
        for width_blocks in range(1, blocks + 1):
            for height_blocks in range(1, blocks + 1):
                if width_blocks * height_blocks <= max_num_tiles:
                    ratios.add((width_blocks, height_blocks))
    return sorted(ratios, key=lambda ratio: ratio[0] * ratio[1])


def _internvl_best_grid(width: int, height: int, max_num_tiles: int) -> tuple[int, int]:
    aspect_ratio = width / height
    best_ratio = (1, 1)
    best_diff = float("inf")
    image_area = width * height
    for ratio in _internvl_target_ratios(max_num_tiles):
        target_aspect_ratio = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect_ratio)
        if diff < best_diff:
            best_ratio = ratio
            best_diff = diff
        elif diff == best_diff:
            if image_area > 0.5 * DEFAULT_INTERNVL_IMAGE_SIZE * DEFAULT_INTERNVL_IMAGE_SIZE * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_image_to_tensor(image: Any, image_size: int) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install project requirements before running local InternVL 4B.")
    if Image is None:
        raise RuntimeError("InternVL 4B requires Pillow. Install project requirements first.")
    if image.mode != "RGB":
        image = image.convert("RGB")
    resized = image.resize((image_size, image_size), Image.BICUBIC)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes()))
    tensor = data.view(image_size, image_size, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def load_internvl_image_tiles(
    image_path: Path,
    image_size: int = DEFAULT_INTERNVL_IMAGE_SIZE,
    max_num_tiles: int = DEFAULT_INTERNVL_MAX_NUM_TILES,
) -> Any:
    if Image is None:
        raise RuntimeError("InternVL 4B requires Pillow. Install project requirements first.")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    grid_width, grid_height = _internvl_best_grid(width, height, max_num_tiles)
    target_width = grid_width * image_size
    target_height = grid_height * image_size
    resized = image.resize((target_width, target_height), Image.BICUBIC)
    tiles = []
    for row in range(grid_height):
        for col in range(grid_width):
            box = (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
            tiles.append(_internvl_image_to_tensor(resized.crop(box), image_size))
    if len(tiles) > 1 and len(tiles) < max_num_tiles:
        tiles.append(_internvl_image_to_tensor(image, image_size))
    return torch.stack(tiles)


def load_internvl_pixel_values(
    frame_paths: list[Path],
    image_size: int = DEFAULT_INTERNVL_IMAGE_SIZE,
    max_num_tiles: int = DEFAULT_INTERNVL_MAX_NUM_TILES,
) -> tuple[Any, list[int]]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install project requirements before running local InternVL 4B.")
    batches = [
        load_internvl_image_tiles(path, image_size=image_size, max_num_tiles=max_num_tiles)
        for path in frame_paths
    ]
    num_patches_list = [int(batch.shape[0]) for batch in batches]
    return torch.cat(batches, dim=0).to(torch.bfloat16), num_patches_list


def build_qwen_vl_video_messages(
    item: dict[str, Any],
    video_path: Path,
    video_fps: float | None = 1.0,
    video_frames: list[Any] | None = None,
    raw_fps: float | None = None,
) -> list[dict[str, Any]]:
    video_content: dict[str, Any] = {
        "type": "video",
        "video": video_frames if video_frames is not None else video_path.resolve().as_posix(),
    }
    if video_fps is not None and video_fps > 0:
        video_content["sample_fps"] = float(video_fps)
    if raw_fps is not None and raw_fps > 0:
        video_content["raw_fps"] = float(raw_fps)
    return [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": build_video_answer_prompt(item, video_path)},
            ],
        }
    ]


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


def _write_frame_answer_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qa_id",
        "modality",
        "section",
        "pair_key",
        "question",
        "provider",
        "model_name",
        "model_answer",
        "status",
        "reason",
        "frame_count",
        "frame_paths",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field, "") for field in fieldnames}
            if isinstance(csv_row.get("frame_paths"), list):
                csv_row["frame_paths"] = json.dumps(csv_row["frame_paths"], ensure_ascii=False)
            writer.writerow(csv_row)


def _write_video_answer_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qa_id",
        "modality",
        "section",
        "pair_key",
        "question",
        "provider",
        "model_name",
        "model_answer",
        "status",
        "reason",
        "video_path",
        "video_fps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _is_completed_frame_answer(result: dict[str, Any]) -> bool:
    reason = str(result.get("reason", ""))
    if reason.startswith("Frame answer call failed:") or reason.startswith("Video answer call failed:"):
        return False
    return result.get("status") == "answered" and bool(str(result.get("model_answer", "")).strip())


def _save_frame_answer_outputs(
    output_json: Path,
    output_csv: Path,
    results_by_id: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "results": results_by_id,
        "metadata": metadata,
    }
    _write_json(output_json, payload)
    _write_frame_answer_csv(output_csv, list(results_by_id.values()))


def _save_video_answer_outputs(
    output_json: Path,
    output_csv: Path,
    results_by_id: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "results": results_by_id,
        "metadata": metadata,
    }
    _write_json(output_json, payload)
    _write_video_answer_csv(output_csv, list(results_by_id.values()))


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
    if provider == "qwen":
        return QwenLocalCaptionAdapter(model_name=model_name)
    if provider in {"internvl"}:
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


def _frame_answer_row(
    item: dict[str, Any],
    provider: str,
    model_name: str,
    model_answer: str,
    frame_paths: list[Path],
    status: str = "answered",
    reason: str = "",
) -> dict[str, Any]:
    return {
        "qa_id": item["qa_id"],
        "modality": item["modality"],
        "section": item["section"],
        "pair_key": item["pair_key"],
        "question": item["question"],
        "provider": provider,
        "model_name": model_name,
        "model_answer": model_answer,
        "status": status,
        "reason": reason,
        "frame_count": len(frame_paths),
        "frame_paths": [path.as_posix() for path in frame_paths],
        "judge_enabled": False,
    }


def _video_answer_row(
    item: dict[str, Any],
    provider: str,
    model_name: str,
    model_answer: str,
    video_path: Path,
    video_fps: float,
    status: str = "answered",
    reason: str = "",
) -> dict[str, Any]:
    return {
        "qa_id": item["qa_id"],
        "modality": item["modality"],
        "section": item["section"],
        "pair_key": item["pair_key"],
        "question": item["question"],
        "provider": provider,
        "model_name": model_name,
        "model_answer": model_answer,
        "status": status,
        "reason": reason,
        "video_path": video_path.as_posix(),
        "video_fps": video_fps,
        "judge_enabled": False,
    }


def run_gemini_frame_answer_benchmark(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    max_items: int | None = 100,
    batch_size: int = 1,
    delay_between_batches: int = 0,
    resume: bool = True,
    api_key_list_path: Path | str = DEFAULT_GEMINI_API_KEY_LIST_PATH,
    enable_key_rotation: bool = True,
    frame_cache_root: Path | str = DEFAULT_FRAME_CACHE_ROOT,
    max_frames_per_item: int | None = DEFAULT_FRAME_MAX_FRAMES_PER_ITEM,
    adapter: GeminiFrameAnswerAdapter | None = None,
    adapter_factory: Callable[[str | None], GeminiFrameAnswerAdapter] | None = None,
) -> dict[str, Path]:
    """Generate Gemini answers from cached aligned-dataset frames without judging correctness."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    frame_cache_root = Path(frame_cache_root)
    output_json, output_csv = _frame_answer_output_paths(output_dir, model_name)
    batch_size = max(1, int(batch_size))

    api_keys = (
        load_api_keys(api_key_list_path, env_var_name="GEMINI_API_KEY", key_prefixes=("AIza",))
        if enable_key_rotation
        else []
    )
    key_rotation_enabled = bool(api_keys)
    active_key_index = 0
    exhausted_key_count = 0
    skipped_no_frames = 0

    def current_api_key() -> str | None:
        if not key_rotation_enabled:
            return None
        return api_keys[active_key_index]

    def build_adapter() -> GeminiFrameAnswerAdapter:
        key = current_api_key()
        if key is not None:
            os.environ["GEMINI_API_KEY"] = key
        if adapter_factory is not None:
            return adapter_factory(key)
        return GeminiFrameAnswerAdapter(model_name=model_name, api_key=key)

    items = load_valid_qa_items(input_path)
    results_by_id = _load_existing_results(output_json) if resume else {}
    pending = [
        item
        for item in items
        if not _is_completed_frame_answer(results_by_id.get(item["qa_id"], {}))
    ]
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]

    def make_metadata(stopped_reason: str | None = None) -> dict[str, Any]:
        answered_count = sum(1 for result in results_by_id.values() if _is_completed_frame_answer(result))
        metadata = {
            "benchmark_type": FRAME_ANSWER_BENCHMARK_TYPE,
            "input_path": input_path.as_posix(),
            "provider": "gemini",
            "model_name": model_name,
            "judge_enabled": False,
            "frame_cache_root": frame_cache_root.as_posix(),
            "max_frames_per_item": 0 if max_frames_per_item == 0 else max_frames_per_item,
            "resume": resume,
            "total_valid_items": len(items),
            "answered_items": answered_count,
            "pending_items": max(0, len(items) - answered_count),
            "skipped_no_frames": skipped_no_frames,
            "batch_size": batch_size,
            "key_rotation_enabled": key_rotation_enabled,
            "keys_available": len(api_keys),
            "exhausted_key_count": exhausted_key_count,
        }
        if stopped_reason:
            metadata["stopped_reason"] = stopped_reason
        return metadata

    print(
        f"Gemini frame-input answer benchmark resume scan: {len(results_by_id)} complete skipped, "
        f"{len(pending)} pending selected, {len(items)} valid total, model={model_name}."
    )

    if not pending:
        _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
        return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}

    if key_rotation_enabled:
        print(f"Gemini key rotation enabled: {len(api_keys)} key(s) loaded from {Path(api_key_list_path)}.")
        print(f"Using Gemini API {_masked_key_label(api_keys[active_key_index], active_key_index + 1, len(api_keys))}.")

    frame_adapter = adapter or build_adapter()

    def rotate_key_after_quota() -> bool:
        nonlocal active_key_index, exhausted_key_count, frame_adapter
        if not key_rotation_enabled or active_key_index + 1 >= len(api_keys):
            exhausted_key_count = len(api_keys) if key_rotation_enabled else exhausted_key_count
            return False
        active_key_index += 1
        exhausted_key_count = active_key_index
        print(
            f"Gemini quota/rate/key limit reached for key {exhausted_key_count}/{len(api_keys)}. "
            f"Switching to key {active_key_index + 1}/{len(api_keys)} and retrying current item."
        )
        _save_frame_answer_outputs(
            output_json,
            output_csv,
            results_by_id,
            make_metadata(stopped_reason="rotating_after_quota"),
        )
        frame_adapter = build_adapter()
        print(f"Gemini API key changed to {_masked_key_label(api_keys[active_key_index], active_key_index + 1, len(api_keys))}.")
        return True

    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            print(f"Running frame-answer batch {start // batch_size + 1}: {len(batch)} item(s)")
            for item in batch:
                frame_paths = resolve_frame_inputs_for_item(
                    item,
                    frame_cache_root=frame_cache_root,
                    max_frames_per_item=max_frames_per_item,
                )
                if not frame_paths:
                    skipped_no_frames += 1
                    print(f"Skipping {item['qa_id']}: no cached frames found.")
                    continue

                while True:
                    try:
                        model_answer = frame_adapter.answer(item, frame_paths)
                        results_by_id[item["qa_id"]] = _frame_answer_row(
                            item,
                            provider="gemini",
                            model_name=model_name,
                            model_answer=model_answer,
                            frame_paths=frame_paths,
                            status="answered" if model_answer else "failed",
                            reason="" if model_answer else "Frame answer call failed: empty model answer",
                        )
                        break
                    except Exception as exc:
                        if is_quota_error(exc):
                            if rotate_key_after_quota():
                                continue
                            print("\nGemini quota/rate/key limit reached and no more keys are available.")
                            print("Progress saved. Run option 70 again later to resume.")
                            _save_frame_answer_outputs(
                                output_json,
                                output_csv,
                                results_by_id,
                                make_metadata(stopped_reason="quota_or_rate_limit"),
                            )
                            return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}
                        results_by_id[item["qa_id"]] = _frame_answer_row(
                            item,
                            provider="gemini",
                            model_name=model_name,
                            model_answer="",
                            frame_paths=frame_paths,
                            status="failed",
                            reason=f"Frame answer call failed: {exc}",
                        )
                        break

                _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
                print(f"Checkpoint saved: {len(results_by_id)} frame answer item(s)")

            if delay_between_batches > 0 and start + batch_size < len(pending):
                time.sleep(delay_between_batches)
    except KeyboardInterrupt:
        print("\nFrame-answer benchmark cancelled by user. Progress saved.")
        _save_frame_answer_outputs(
            output_json,
            output_csv,
            results_by_id,
            make_metadata(stopped_reason="user_cancelled"),
        )
        return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}

    _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata(stopped_reason="completed"))
    return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}


def run_qwen_vl_frame_answer_benchmark(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_QWEN_VL_MODEL_NAME,
    max_items: int | None = 100,
    batch_size: int = 1,
    delay_between_batches: int = 0,
    resume: bool = True,
    frame_cache_root: Path | str = DEFAULT_FRAME_CACHE_ROOT,
    max_frames_per_item: int | None = DEFAULT_FRAME_MAX_FRAMES_PER_ITEM,
    adapter: QwenVLFrameAnswerAdapter | None = None,
) -> dict[str, Path]:
    """Generate local Qwen-VL answers from cached aligned-dataset frames without judging correctness."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    frame_cache_root = Path(frame_cache_root)
    output_json, output_csv = _frame_answer_output_paths(output_dir, model_name)
    batch_size = max(1, int(batch_size))
    if batch_size != 1:
        print("Local Qwen-VL frame benchmark forces batch size to 1 to reduce GPU memory pressure.")
        batch_size = 1

    print(f"Local Qwen-VL runtime: {QwenVLFrameAnswerAdapter.runtime_summary(model_name=model_name)}")

    items = load_valid_qa_items(input_path)
    results_by_id = _load_existing_results(output_json) if resume else {}
    pending = [
        item
        for item in items
        if not _is_completed_frame_answer(results_by_id.get(item["qa_id"], {}))
    ]
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]

    skipped_no_frames = 0

    def make_metadata(stopped_reason: str | None = None) -> dict[str, Any]:
        answered_count = sum(1 for result in results_by_id.values() if _is_completed_frame_answer(result))
        metadata = {
            "benchmark_type": FRAME_ANSWER_BENCHMARK_TYPE,
            "input_path": input_path.as_posix(),
            "provider": "qwen_vl",
            "model_name": model_name,
            "judge_enabled": False,
            "frame_cache_root": frame_cache_root.as_posix(),
            "max_frames_per_item": 0 if max_frames_per_item == 0 else max_frames_per_item,
            "resume": resume,
            "total_valid_items": len(items),
            "answered_items": answered_count,
            "pending_items": max(0, len(items) - answered_count),
            "skipped_no_frames": skipped_no_frames,
            "batch_size": batch_size,
            "key_rotation_enabled": False,
            "keys_available": 0,
            "exhausted_key_count": 0,
        }
        if stopped_reason:
            metadata["stopped_reason"] = stopped_reason
        return metadata

    print(
        f"Qwen-VL frame-input answer benchmark resume scan: {len(results_by_id)} complete skipped, "
        f"{len(pending)} pending selected, {len(items)} valid total, model={model_name}."
    )

    if not pending:
        _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
        return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}

    frame_adapter = adapter or QwenVLFrameAnswerAdapter(model_name=model_name)

    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            print(f"Running Qwen-VL frame-answer batch {start // batch_size + 1}: {len(batch)} item(s)")
            for item in batch:
                frame_paths = resolve_frame_inputs_for_item(
                    item,
                    frame_cache_root=frame_cache_root,
                    max_frames_per_item=max_frames_per_item,
                )
                if not frame_paths:
                    skipped_no_frames += 1
                    print(f"Skipping {item['qa_id']}: no cached frames found.")
                    continue

                try:
                    model_answer = frame_adapter.answer(item, frame_paths)
                    results_by_id[item["qa_id"]] = _frame_answer_row(
                        item,
                        provider="qwen_vl",
                        model_name=model_name,
                        model_answer=model_answer,
                        frame_paths=frame_paths,
                        status="answered" if model_answer else "failed",
                        reason="" if model_answer else "Frame answer call failed: empty model answer",
                    )
                except Exception as exc:
                    results_by_id[item["qa_id"]] = _frame_answer_row(
                        item,
                        provider="qwen_vl",
                        model_name=model_name,
                        model_answer="",
                        frame_paths=frame_paths,
                        status="failed",
                        reason=f"Frame answer call failed: {exc}",
                    )

                _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
                print(f"Checkpoint saved: {len(results_by_id)} frame answer item(s)")

            if delay_between_batches > 0 and start + batch_size < len(pending):
                time.sleep(delay_between_batches)
    except KeyboardInterrupt:
        print("\nQwen-VL frame-answer benchmark cancelled by user. Progress saved.")
        _save_frame_answer_outputs(
            output_json,
            output_csv,
            results_by_id,
            make_metadata(stopped_reason="user_cancelled"),
        )
        return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}

    _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata(stopped_reason="completed"))
    return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}


def run_internvl_frame_answer_benchmark(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_INTERNVL_MODEL_NAME,
    max_items: int | None = 100,
    batch_size: int = 1,
    delay_between_batches: int = 0,
    resume: bool = True,
    frame_cache_root: Path | str = DEFAULT_FRAME_CACHE_ROOT,
    max_frames_per_item: int | None = DEFAULT_FRAME_MAX_FRAMES_PER_ITEM,
    revision: str | None = DEFAULT_INTERNVL_REVISION,
    adapter: InternVLFrameAnswerAdapter | None = None,
) -> dict[str, Path]:
    """Generate local InternVL 4B answers from cached aligned-dataset frames without judging correctness."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    frame_cache_root = Path(frame_cache_root)
    output_json, output_csv = _frame_answer_output_paths(output_dir, model_name)
    batch_size = max(1, int(batch_size))
    if batch_size != 1:
        print("Local InternVL 4B frame benchmark forces batch size to 1 to reduce GPU memory pressure.")
        batch_size = 1

    revision = str(revision or "").strip() or None

    print(f"Local InternVL 4B runtime: {InternVLFrameAnswerAdapter.runtime_summary(model_name=model_name, revision=revision)}")

    items = load_valid_qa_items(input_path)
    results_by_id = _load_existing_results(output_json) if resume else {}
    pending = [
        item
        for item in items
        if not _is_completed_frame_answer(results_by_id.get(item["qa_id"], {}))
    ]
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]

    skipped_no_frames = 0

    def make_metadata(stopped_reason: str | None = None) -> dict[str, Any]:
        answered_count = sum(1 for result in results_by_id.values() if _is_completed_frame_answer(result))
        metadata = {
            "benchmark_type": FRAME_ANSWER_BENCHMARK_TYPE,
            "input_path": input_path.as_posix(),
            "provider": "internvl",
            "model_name": model_name,
            "revision": revision or "",
            "judge_enabled": False,
            "frame_cache_root": frame_cache_root.as_posix(),
            "max_frames_per_item": 0 if max_frames_per_item == 0 else max_frames_per_item,
            "resume": resume,
            "total_valid_items": len(items),
            "answered_items": answered_count,
            "pending_items": max(0, len(items) - answered_count),
            "skipped_no_frames": skipped_no_frames,
            "batch_size": batch_size,
            "key_rotation_enabled": False,
            "keys_available": 0,
            "exhausted_key_count": 0,
        }
        if stopped_reason:
            metadata["stopped_reason"] = stopped_reason
        return metadata

    print(
        f"InternVL 4B frame-input answer benchmark resume scan: {len(results_by_id)} complete skipped, "
        f"{len(pending)} pending selected, {len(items)} valid total, model={model_name}."
    )

    if not pending:
        _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
        return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}

    frame_adapter = adapter or InternVLFrameAnswerAdapter(model_name=model_name, revision=revision)

    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            print(f"Running InternVL 4B frame-answer batch {start // batch_size + 1}: {len(batch)} item(s)")
            for item in batch:
                frame_paths = resolve_frame_inputs_for_item(
                    item,
                    frame_cache_root=frame_cache_root,
                    max_frames_per_item=max_frames_per_item,
                )
                if not frame_paths:
                    skipped_no_frames += 1
                    print(f"Skipping {item['qa_id']}: no cached frames found.")
                    continue

                try:
                    model_answer = frame_adapter.answer(item, frame_paths)
                    results_by_id[item["qa_id"]] = _frame_answer_row(
                        item,
                        provider="internvl",
                        model_name=model_name,
                        model_answer=model_answer,
                        frame_paths=frame_paths,
                        status="answered" if model_answer else "failed",
                        reason="" if model_answer else "Frame answer call failed: empty model answer",
                    )
                except Exception as exc:
                    results_by_id[item["qa_id"]] = _frame_answer_row(
                        item,
                        provider="internvl",
                        model_name=model_name,
                        model_answer="",
                        frame_paths=frame_paths,
                        status="failed",
                        reason=f"Frame answer call failed: {exc}",
                    )

                _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
                print(f"Checkpoint saved: {len(results_by_id)} frame answer item(s)")

            if delay_between_batches > 0 and start + batch_size < len(pending):
                time.sleep(delay_between_batches)
    except KeyboardInterrupt:
        print("\nInternVL 4B frame-answer benchmark cancelled by user. Progress saved.")
        _save_frame_answer_outputs(
            output_json,
            output_csv,
            results_by_id,
            make_metadata(stopped_reason="user_cancelled"),
        )
        return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}

    _save_frame_answer_outputs(output_json, output_csv, results_by_id, make_metadata(stopped_reason="completed"))
    return {"frame_answers_json": output_json, "frame_answers_csv": output_csv}


def run_qwen_vl_video_answer_benchmark(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_QWEN_VL_MODEL_NAME,
    max_items: int | None = 100,
    batch_size: int = 1,
    delay_between_batches: int = 0,
    resume: bool = True,
    dataset_root: Path | str = DEFAULT_FRAME_CACHE_ROOT,
    video_fps: float = 1.0,
    adapter: QwenVLVideoAnswerAdapter | None = None,
) -> dict[str, Path]:
    """Generate local Qwen-VL answers from aligned-dataset video files without judging correctness."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    dataset_root = Path(dataset_root)
    output_json, output_csv = _video_answer_output_paths(output_dir, model_name)
    batch_size = max(1, int(batch_size))
    if batch_size != 1:
        print("Local Qwen-VL video benchmark forces batch size to 1 to reduce GPU memory pressure.")
        batch_size = 1
    video_fps = max(0.0, float(video_fps))

    print(f"Local Qwen-VL runtime: {QwenVLFrameAnswerAdapter.runtime_summary(model_name=model_name)}")

    items = load_valid_qa_items(input_path)
    results_by_id = _load_existing_results(output_json) if resume else {}
    pending = [
        item
        for item in items
        if not _is_completed_frame_answer(results_by_id.get(item["qa_id"], {}))
    ]
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]

    skipped_no_video = 0

    def make_metadata(stopped_reason: str | None = None) -> dict[str, Any]:
        answered_count = sum(1 for result in results_by_id.values() if _is_completed_frame_answer(result))
        metadata = {
            "benchmark_type": "video_input_answer_generation",
            "input_path": input_path.as_posix(),
            "provider": "qwen_vl",
            "model_name": model_name,
            "judge_enabled": False,
            "dataset_root": dataset_root.as_posix(),
            "video_fps": video_fps,
            "resume": resume,
            "total_valid_items": len(items),
            "answered_items": answered_count,
            "pending_items": max(0, len(items) - answered_count),
            "skipped_no_video": skipped_no_video,
            "batch_size": batch_size,
            "key_rotation_enabled": False,
            "keys_available": 0,
            "exhausted_key_count": 0,
        }
        if stopped_reason:
            metadata["stopped_reason"] = stopped_reason
        return metadata

    print(
        f"Qwen-VL video-input answer benchmark resume scan: {len(results_by_id)} complete skipped, "
        f"{len(pending)} pending selected, {len(items)} valid total, model={model_name}."
    )

    if not pending:
        _save_video_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
        return {"video_answers_json": output_json, "video_answers_csv": output_csv}

    video_adapter = adapter or QwenVLVideoAnswerAdapter(
        model_name=model_name,
        video_fps=video_fps,
    )

    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            print(f"Running Qwen-VL video-answer batch {start // batch_size + 1}: {len(batch)} item(s)")
            for item in batch:
                video_path = resolve_video_input_for_item(item, dataset_root=dataset_root)
                if video_path is None:
                    skipped_no_video += 1
                    print(f"Skipping {item['qa_id']}: no aligned video found.")
                    continue

                try:
                    model_answer = video_adapter.answer(item, video_path)
                    results_by_id[item["qa_id"]] = _video_answer_row(
                        item,
                        provider="qwen_vl",
                        model_name=model_name,
                        model_answer=model_answer,
                        video_path=video_path,
                        video_fps=video_fps,
                        status="answered" if model_answer else "failed",
                        reason="" if model_answer else "Video answer call failed: empty model answer",
                    )
                except Exception as exc:
                    results_by_id[item["qa_id"]] = _video_answer_row(
                        item,
                        provider="qwen_vl",
                        model_name=model_name,
                        model_answer="",
                        video_path=video_path,
                        video_fps=video_fps,
                        status="failed",
                        reason=f"Video answer call failed: {exc}",
                    )

                _save_video_answer_outputs(output_json, output_csv, results_by_id, make_metadata())
                print(f"Checkpoint saved: {len(results_by_id)} video answer item(s)")

            if delay_between_batches > 0 and start + batch_size < len(pending):
                time.sleep(delay_between_batches)
    except KeyboardInterrupt:
        print("\nQwen-VL video-answer benchmark cancelled by user. Progress saved.")
        _save_video_answer_outputs(
            output_json,
            output_csv,
            results_by_id,
            make_metadata(stopped_reason="user_cancelled"),
        )
        return {"video_answers_json": output_json, "video_answers_csv": output_csv}

    _save_video_answer_outputs(output_json, output_csv, results_by_id, make_metadata(stopped_reason="completed"))
    return {"video_answers_json": output_json, "video_answers_csv": output_csv}


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
    elif provider == "qwen":
        answer_key_label = "Qwen local"
        answer_env_var = ""
        answer_key_path = api_key_list_path
        answer_key_prefixes = ()
        if batch_size != 1:
            print("Local Qwen benchmark forces batch size to 1 to reduce GPU memory pressure.")
            batch_size = 1
        print(f"Local Qwen runtime: {QwenLocalCaptionAdapter.runtime_summary()}")
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
