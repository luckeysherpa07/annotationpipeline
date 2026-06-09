"""Quality checks for generated QA files."""

__all__ = [
    "clean_aligned_qa_dataset",
    "evaluate_aligned_qa",
    "run_aligned_qa_benchmark",
    "run_aligned_qa_llm_evaluation",
]


def clean_aligned_qa_dataset(*args, **kwargs):
    from .cleaner import clean_aligned_qa_dataset as _clean

    return _clean(*args, **kwargs)


def evaluate_aligned_qa(*args, **kwargs):
    from .aligned_evaluator import evaluate_aligned_qa as _evaluate

    return _evaluate(*args, **kwargs)


def run_aligned_qa_benchmark(*args, **kwargs):
    from .benchmark import run_aligned_qa_benchmark as _run

    return _run(*args, **kwargs)


def run_aligned_qa_llm_evaluation(*args, **kwargs):
    from .llm_evaluator import run_aligned_qa_llm_evaluation as _run

    return _run(*args, **kwargs)
