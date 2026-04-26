"""
Annotation pipeline wrapper module.

This module provides a unified interface to the annotation pipeline.
It re-exports the main functions from the pipeline submodule.
"""

from .pipeline.main import run, run_audio, run_depth, run_event, run_ir, run_marigold_depth_qa
from .fusion import run_late_fusion


def run_marigold_depth_estimation(*args, **kwargs):
    from .pipeline.modalities.marigold import run_marigold_depth_estimation as _run

    return _run(*args, **kwargs)


__all__ = [
    "run",
    "run_audio",
    "run_event",
    "run_depth",
    "run_ir",
    "run_marigold_depth_estimation",
    "run_marigold_depth_qa",
    "run_late_fusion",
]
