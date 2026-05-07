"""Script entrypoint for grouping normalized evidence units."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_feature.reasoning import run_group_evidence


if __name__ == "__main__":
    run_group_evidence(
        input_path="normalized_evidence_units.json",
        output_path="grouped_evidence.json",
    )
