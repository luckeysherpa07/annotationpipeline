"""Quality checks for generated QA files."""

__all__ = ["evaluate_aligned_qa", "run_aligned_qa_llm_evaluation"]


def evaluate_aligned_qa(*args, **kwargs):
    from .aligned_evaluator import evaluate_aligned_qa as _evaluate

    return _evaluate(*args, **kwargs)


def run_aligned_qa_llm_evaluation(*args, **kwargs):
    from .llm_evaluator import run_aligned_qa_llm_evaluation as _run

    return _run(*args, **kwargs)
