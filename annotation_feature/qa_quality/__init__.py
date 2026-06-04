"""Quality checks for generated QA files."""

__all__ = ["evaluate_aligned_qa"]


def evaluate_aligned_qa(*args, **kwargs):
    from .aligned_evaluator import evaluate_aligned_qa as _evaluate

    return _evaluate(*args, **kwargs)
