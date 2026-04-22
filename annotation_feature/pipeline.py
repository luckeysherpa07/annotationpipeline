"""
Annotation pipeline wrapper module.

This module provides a unified interface to the annotation pipeline.
It re-exports the main functions from the pipeline submodule.
"""
from .pipeline.main import run, run_event, run_depth

__all__ = ["run", "run_event", "run_depth"]

