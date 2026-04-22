"""Event modality pipeline for annotation."""
from .pipeline import (
    run_event_parallel_pipeline,
    process_event_pair_batch,
    normalize_event_results,
)

__all__ = [
    "run_event_parallel_pipeline",
    "process_event_pair_batch",
    "normalize_event_results",
]
