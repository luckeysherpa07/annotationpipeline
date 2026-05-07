"""Script entrypoint for normalizing modality QA results."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_feature.reasoning import normalize_all_modalities


if __name__ == "__main__":
    normalize_all_modalities(
        rgb_path="rgb_qa_results.json",
        event_path="event_qa_results.json",
        depth_path="depth_qa_results.json",
        ir_path="ir_qa_results.json",
        audio_path="audio_qa_results.json",
        output_path="normalized_evidence_units.json",
    )
