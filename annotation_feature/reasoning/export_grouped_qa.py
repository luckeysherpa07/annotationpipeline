"""Script entrypoint for exporting grouped Q/A pairs."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_feature.reasoning import run_export_grouped_qa


if __name__ == "__main__":
    run_export_grouped_qa(
        input_path="grouped_evidence.json",
        output_path="grouped_qa_pairs.json",
    )
