"""Quality checks for generated QA files."""

__all__ = [
    "clean_aligned_qa_dataset",
    "evaluate_aligned_qa",
    "run_gemini_frame_answer_benchmark",
    "run_internvl_frame_answer_benchmark",
    "run_molmo2_frame_answer_benchmark",
    "run_qwen_vl_frame_answer_benchmark",
    "run_qwen_vl_video_answer_benchmark",
    "run_aligned_qa_benchmark",
    "run_aligned_qa_llm_evaluation",
    "load_evaluation_records",
    "score_records",
    "write_evaluation_outputs",
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


def run_gemini_frame_answer_benchmark(*args, **kwargs):
    from .benchmark import run_gemini_frame_answer_benchmark as _run

    return _run(*args, **kwargs)


def run_internvl_frame_answer_benchmark(*args, **kwargs):
    from .benchmark import run_internvl_frame_answer_benchmark as _run

    return _run(*args, **kwargs)


def run_molmo2_frame_answer_benchmark(*args, **kwargs):
    from .benchmark import run_molmo2_frame_answer_benchmark as _run

    return _run(*args, **kwargs)


def run_qwen_vl_frame_answer_benchmark(*args, **kwargs):
    from .benchmark import run_qwen_vl_frame_answer_benchmark as _run

    return _run(*args, **kwargs)


def run_qwen_vl_video_answer_benchmark(*args, **kwargs):
    from .benchmark import run_qwen_vl_video_answer_benchmark as _run

    return _run(*args, **kwargs)


def run_aligned_qa_llm_evaluation(*args, **kwargs):
    from .llm_evaluator import run_aligned_qa_llm_evaluation as _run

    return _run(*args, **kwargs)


def load_evaluation_records(*args, **kwargs):
    from .result_loader import load_evaluation_records as _load

    return _load(*args, **kwargs)


def score_records(*args, **kwargs):
    from .evaluation_report import score_records as _score

    return _score(*args, **kwargs)


def write_evaluation_outputs(*args, **kwargs):
    from .evaluation_report import write_evaluation_outputs as _write

    return _write(*args, **kwargs)
