"""IR modality pipeline for annotation."""
from .pipeline import (
    run_ir_parallel_pipeline,
    process_ir_pair_batch,
    normalize_ir_results,
)

__all__ = [
    "run_ir_parallel_pipeline",
    "process_ir_pair_batch",
    "normalize_ir_results",
]
