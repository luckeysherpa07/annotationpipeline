"""Aligned QA quality menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.cli.actions.aligned_choices import (
    ALIGNED_QA_FRAME_ANSWER_BENCHMARK,
    ALIGNED_QA_QUALITY_BENCHMARK,
    ALIGNED_QA_QUALITY_CLEAN,
    ALIGNED_QA_QUALITY_EVALUATE,
    ALIGNED_QA_QUALITY_GPT_BENCHMARK,
    ALIGNED_QA_QUALITY_LLM_EVAL,
    ALIGNED_QA_QUALITY_QWEN_BENCHMARK,
)
from annotation_feature.qa_quality import (
    clean_aligned_qa_dataset,
    evaluate_aligned_qa,
    run_gemini_frame_answer_benchmark,
    run_aligned_qa_benchmark,
    run_aligned_qa_llm_evaluation,
)


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_aligned_qa_quality_actions(
    confirm: Callable[[str], bool],
    output_dir: Path | str = "outputs",
) -> dict[str, MenuAction]:
    """Build menu choices for aligned QA quality evaluation."""

    def run_evaluation() -> None:
        _print_header("Running: evaluate aligned QA quality")
        print("Reads qa_pairs/aligned/*.json.")
        print("Writes aligned QA quality report, item CSV, split-item exports, and cleaned QA exports.")
        print("This is rule-based and does not call Gemini.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            outputs = evaluate_aligned_qa(output_dir=output_dir)
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    def run_llm_evaluation() -> None:
        _print_header("Running: LLM-assisted aligned QA evaluation")
        print("Reads outputs/aligned_qa_cleaned_items.json.")
        print("Writes outputs/aligned_qa_llm_eval_results.json and outputs/aligned_qa_llm_eval_items.csv.")
        print("This calls Gemini and supports resume/checkpoint.")
        print("Default max items is 1000 for quota-controlled full evaluation; enter 0 to run all remaining items.")
        print("-" * 60)
        raw_limit = input("Max items to evaluate this run? (default 1000, 0 = all): ").strip()
        if not raw_limit:
            max_items = 1000
        else:
            try:
                parsed_limit = int(raw_limit)
            except ValueError:
                print("Invalid max items value.")
                return
            max_items = None if parsed_limit == 0 else max(0, parsed_limit)

        raw_batch_size = input("Batch size? (default 50): ").strip()
        if not raw_batch_size:
            batch_size = 50
        else:
            try:
                batch_size = max(1, int(raw_batch_size))
            except ValueError:
                print("Invalid batch size value.")
                return

        if confirm("Continue? (yes/no): "):
            outputs = run_aligned_qa_llm_evaluation(
                batch_size=batch_size,
                max_items=max_items,
            )
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    def run_cleaner() -> None:
        input_path = Path(output_dir) / "aligned_qa_llm_eval_results.json"
        output_path = Path(output_dir) / "aligned_qa_valid_items.json"
        _print_header("Running: clean aligned QA dataset")
        print(f"Reads {input_path}.")
        print(f"Writes {output_path}.")
        print("Keeps only pass/low-risk/caption-supported single QA items.")
        print("This is rule-based and does not call Gemini.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            result = clean_aligned_qa_dataset(input_path=input_path, output_path=output_path)
            summary = result["summary"]
            print(
                "Cleaned aligned QA dataset: "
                f"{summary['total_valid']} valid, {summary['total_removed']} removed, "
                f"{summary['total_input']} input."
            )
            print(f"valid_json: {output_path}")
        else:
            print("Cancelled.")

    def run_benchmark() -> None:
        input_path = Path(output_dir) / "aligned_qa_valid_items.json"
        benchmark_output_dir = Path(output_dir) / "benchmarks"
        _print_header("Running: aligned QA caption-only benchmark")
        print(f"Reads {input_path}.")
        print(f"Writes benchmark JSON/CSV files under {benchmark_output_dir}.")
        print("Caption-only mode: tested models receive modality, section, caption, and question.")
        print("Gemini is used as the answer judge.")
        print("Providers available now: gemini. Reserved: chatgpt/openai, qwen, internvl.")
        print("-" * 60)

        provider = input("Model provider? (default gemini): ").strip().lower() or "gemini"
        model_name = input("Model name? (default gemini-3.1-flash-lite): ").strip() or "gemini-3.1-flash-lite"

        raw_limit = input("Max items to benchmark this run? (default 100, 0 = all): ").strip()
        if not raw_limit:
            max_items = 100
        else:
            try:
                parsed_limit = int(raw_limit)
            except ValueError:
                print("Invalid max items value.")
                return
            max_items = None if parsed_limit == 0 else max(0, parsed_limit)

        raw_batch_size = input("Batch size? (default 5): ").strip()
        if not raw_batch_size:
            batch_size = 5
        else:
            try:
                batch_size = max(1, int(raw_batch_size))
            except ValueError:
                print("Invalid batch size value.")
                return

        raw_delay = input("Delay between batches? (default 30 seconds): ").strip()
        if not raw_delay:
            delay_between_batches = 30
        else:
            try:
                delay_between_batches = max(0, int(raw_delay))
            except ValueError:
                print("Invalid delay value.")
                return

        if confirm("Continue? (yes/no): "):
            try:
                outputs = run_aligned_qa_benchmark(
                    input_path=input_path,
                    output_dir=benchmark_output_dir,
                    provider=provider,
                    model_name=model_name,
                    max_items=max_items,
                    batch_size=batch_size,
                    delay_between_batches=delay_between_batches,
                )
            except NotImplementedError as exc:
                print(str(exc))
                return
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    def run_gpt_benchmark() -> None:
        input_path = Path(output_dir) / "aligned_qa_valid_items.json"
        benchmark_output_dir = Path(output_dir) / "benchmarks"
        _print_header("Running: ChatGPT/OpenAI aligned QA caption-only benchmark")
        print(f"Reads {input_path}.")
        print(f"Writes benchmark JSON/CSV files under {benchmark_output_dir}.")
        print("Caption-only mode: OpenAI receives modality, section, caption, and question.")
        print("Gemini is used as the answer judge for comparability.")
        print("OpenAI keys are read from api_key_list/openai_api_key_list.")
        print("-" * 60)

        model_name = input("OpenAI model name? (default gpt-5.4-mini): ").strip() or "gpt-5.4-mini"

        raw_limit = input("Max items to benchmark this run? (default 100, 0 = all): ").strip()
        if not raw_limit:
            max_items = 100
        else:
            try:
                parsed_limit = int(raw_limit)
            except ValueError:
                print("Invalid max items value.")
                return
            max_items = None if parsed_limit == 0 else max(0, parsed_limit)

        raw_batch_size = input("Batch size? (default 5): ").strip()
        if not raw_batch_size:
            batch_size = 5
        else:
            try:
                batch_size = max(1, int(raw_batch_size))
            except ValueError:
                print("Invalid batch size value.")
                return

        raw_delay = input("Delay between batches? (default 30 seconds): ").strip()
        if not raw_delay:
            delay_between_batches = 30
        else:
            try:
                delay_between_batches = max(0, int(raw_delay))
            except ValueError:
                print("Invalid delay value.")
                return

        if confirm("Continue? (yes/no): "):
            outputs = run_aligned_qa_benchmark(
                input_path=input_path,
                output_dir=benchmark_output_dir,
                provider="openai",
                model_name=model_name,
                max_items=max_items,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches,
            )
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    def run_qwen_benchmark() -> None:
        input_path = Path(output_dir) / "aligned_qa_valid_items.json"
        benchmark_output_dir = Path(output_dir) / "benchmarks"
        _print_header("Running: local Qwen 8B 4-bit aligned QA caption-only benchmark")
        print(f"Reads {input_path}.")
        print(f"Writes benchmark JSON/CSV files under {benchmark_output_dir}.")
        print("Caption-only mode: local Qwen receives modality, section, caption, and question.")
        print("Gemini is used as the answer judge for comparability.")
        print("Local Qwen uses Transformers + bitsandbytes 4-bit NF4 and requires CUDA; CPU fallback is disabled.")
        print("-" * 60)

        model_name = input("Qwen model name? (default Qwen/Qwen3-8B): ").strip() or "Qwen/Qwen3-8B"

        raw_limit = input("Max items to benchmark this run? (default 20, 0 = all): ").strip()
        if not raw_limit:
            max_items = 20
        else:
            try:
                parsed_limit = int(raw_limit)
            except ValueError:
                print("Invalid max items value.")
                return
            max_items = None if parsed_limit == 0 else max(0, parsed_limit)

        raw_batch_size = input("Batch size? (default 1): ").strip()
        if not raw_batch_size:
            batch_size = 1
        else:
            try:
                batch_size = max(1, int(raw_batch_size))
            except ValueError:
                print("Invalid batch size value.")
                return

        raw_delay = input("Delay between batches? (default 0 seconds): ").strip()
        if not raw_delay:
            delay_between_batches = 0
        else:
            try:
                delay_between_batches = max(0, int(raw_delay))
            except ValueError:
                print("Invalid delay value.")
                return

        if confirm("Continue? (yes/no): "):
            try:
                outputs = run_aligned_qa_benchmark(
                    input_path=input_path,
                    output_dir=benchmark_output_dir,
                    provider="qwen",
                    model_name=model_name,
                    max_items=max_items,
                    batch_size=batch_size,
                    delay_between_batches=delay_between_batches,
                )
            except (ImportError, RuntimeError, NotImplementedError) as exc:
                print(str(exc))
                return
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    def run_frame_answer_benchmark() -> None:
        input_path = Path(output_dir) / "aligned_qa_valid_items.json"
        benchmark_output_dir = Path(output_dir) / "benchmarks"
        _print_header("Running: Gemini frame-input aligned QA answer benchmark")
        print(f"Reads {input_path}.")
        print("Reads cached frames from aligned_dataset/.frames_cache*.")
        print(f"Writes answer-only JSON/CSV files under {benchmark_output_dir}.")
        print("No judge runs in this option; correctness scoring can run later.")
        print("-" * 60)

        model_name = input("Gemini model name? (default gemini-3.1-flash-lite): ").strip() or "gemini-3.1-flash-lite"

        raw_limit = input("Max items to answer this run? (default 100, 0 = all): ").strip()
        if not raw_limit:
            max_items = 100
        else:
            try:
                parsed_limit = int(raw_limit)
            except ValueError:
                print("Invalid max items value.")
                return
            max_items = None if parsed_limit == 0 else max(0, parsed_limit)

        raw_max_frames = input("Max frames per item? (default 6, 0 = all): ").strip()
        if not raw_max_frames:
            max_frames_per_item = 6
        else:
            try:
                parsed_frames = int(raw_max_frames)
            except ValueError:
                print("Invalid max frames value.")
                return
            max_frames_per_item = 0 if parsed_frames == 0 else max(1, parsed_frames)

        raw_batch_size = input("Batch size? (default 1): ").strip()
        if not raw_batch_size:
            batch_size = 1
        else:
            try:
                batch_size = max(1, int(raw_batch_size))
            except ValueError:
                print("Invalid batch size value.")
                return

        raw_delay = input("Delay between batches? (default 0 seconds): ").strip()
        if not raw_delay:
            delay_between_batches = 0
        else:
            try:
                delay_between_batches = max(0, int(raw_delay))
            except ValueError:
                print("Invalid delay value.")
                return

        if confirm("Continue? (yes/no): "):
            outputs = run_gemini_frame_answer_benchmark(
                input_path=input_path,
                output_dir=benchmark_output_dir,
                model_name=model_name,
                max_items=max_items,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches,
                max_frames_per_item=max_frames_per_item,
            )
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    quality_action = MenuAction(
        action_id="aligned.qa_quality.evaluate",
        title="Evaluate aligned QA quality",
        section="ALIGNED QA QUALITY",
        handler=run_evaluation,
    )
    llm_action = MenuAction(
        action_id="aligned.qa_quality.llm_eval",
        title="Run LLM-assisted aligned QA evaluation",
        section="ALIGNED QA QUALITY",
        handler=run_llm_evaluation,
    )
    clean_action = MenuAction(
        action_id="aligned.qa_quality.clean",
        title="Clean aligned QA dataset",
        section="ALIGNED QA QUALITY",
        handler=run_cleaner,
    )
    benchmark_action = MenuAction(
        action_id="aligned.qa_quality.benchmark",
        title="Run aligned QA benchmark",
        section="BENCHMARK EVALUATION",
        handler=run_benchmark,
    )
    gpt_benchmark_action = MenuAction(
        action_id="aligned.qa_quality.gpt_benchmark",
        title="Run ChatGPT/OpenAI aligned QA benchmark",
        section="BENCHMARK EVALUATION",
        handler=run_gpt_benchmark,
    )
    qwen_benchmark_action = MenuAction(
        action_id="aligned.qa_quality.qwen_benchmark",
        title="Run local Qwen 8B aligned QA benchmark",
        section="BENCHMARK EVALUATION",
        handler=run_qwen_benchmark,
    )
    frame_answer_benchmark_action = MenuAction(
        action_id="aligned.qa_quality.frame_answer_benchmark",
        title="Run Gemini frame-input aligned QA answer benchmark",
        section="FRAME INPUT ANSWER BENCHMARK",
        handler=run_frame_answer_benchmark,
    )
    return {
        ALIGNED_QA_QUALITY_EVALUATE: quality_action,
        ALIGNED_QA_QUALITY_LLM_EVAL: llm_action,
        ALIGNED_QA_QUALITY_CLEAN: clean_action,
        ALIGNED_QA_QUALITY_BENCHMARK: benchmark_action,
        ALIGNED_QA_QUALITY_GPT_BENCHMARK: gpt_benchmark_action,
        ALIGNED_QA_QUALITY_QWEN_BENCHMARK: qwen_benchmark_action,
        ALIGNED_QA_FRAME_ANSWER_BENCHMARK: frame_answer_benchmark_action,
        "aligned.qa_quality.evaluate": quality_action,
        "aligned.qa_quality.llm_eval": llm_action,
        "aligned.qa_quality.clean": clean_action,
        "aligned.qa_quality.benchmark": benchmark_action,
        "aligned.qa_quality.gpt_benchmark": gpt_benchmark_action,
        "aligned.qa_quality.qwen_benchmark": qwen_benchmark_action,
        "aligned.qa_quality.frame_answer_benchmark": frame_answer_benchmark_action,
    }
