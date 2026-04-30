"""Audio modality pipeline for annotation."""
from .pipeline import (
    build_audio_visual_prompt,
    build_hia_prompt,
    build_qna_prompt,
    generate_audiovisual_caption,
    generate_hia_caption,
    generate_qa_pairs,
    process_single_audio,
    process_single_audio_pair,
    run_parallel_pipeline,
    normalize_annotation_results,
)

__all__ = [
    "build_hia_prompt",
    "build_audio_visual_prompt",
    "build_qna_prompt",
    "generate_hia_caption",
    "generate_audiovisual_caption",
    "generate_qa_pairs",
    "process_single_audio",
    "process_single_audio_pair",
    "run_parallel_pipeline",
    "normalize_annotation_results",
]
