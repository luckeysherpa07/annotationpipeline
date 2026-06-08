"""Audio modality pipeline for annotation."""
from .pipeline import (
    DEMO_TIMESTAMPED_CAPTION,
    build_audio_visual_prompt,
    build_hia_prompt,
    build_qna_prompt,
    enrich_audio_annotations,
    generate_audiovisual_caption,
    generate_hia_caption,
    generate_qa_pairs,
    format_audio_annotations,
    process_single_audio,
    process_single_audio_pair,
    run_parallel_pipeline,
    normalize_annotation_results,
)

__all__ = [
    "DEMO_TIMESTAMPED_CAPTION",
    "build_hia_prompt",
    "build_audio_visual_prompt",
    "build_qna_prompt",
    "enrich_audio_annotations",
    "generate_hia_caption",
    "generate_audiovisual_caption",
    "generate_qa_pairs",
    "format_audio_annotations",
    "process_single_audio",
    "process_single_audio_pair",
    "run_parallel_pipeline",
    "normalize_annotation_results",
]
