"""RGB modality pipeline for annotation."""
from .pipeline import (
    run_parallel_pipeline,
    process_single_pair_batch,
    normalize_annotation_results,
)

__all__ = [
    "run_parallel_pipeline",
    "process_single_pair_batch",
    "normalize_annotation_results",
]
