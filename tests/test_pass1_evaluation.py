import csv
import json
from pathlib import Path

import pytest

from scripts.run_pass1_evaluation import (
    BASELINE_PROMPT_VERSION,
    aggregate_metrics,
    extract_run_metrics,
    load_manifest,
    next_run_path,
    write_reports,
)


def _write_manifest(path: Path, **overrides):
    manifest = {
        "evaluation_set": "pass1_dev_v1",
        "prompt_version": BASELINE_PROMPT_VERSION,
        "runs_per_sample": 2,
        "pipeline": {"generation_mode": "gemini"},
        "samples": [
            {
                "sample_id": "bike_night",
                "input": "outputs/temp_bike_input.json",
                "tags": ["rgb_event", "night"],
                "pipeline": {"pairs": "rgb+event", "limit": 1},
            }
        ],
        **overrides,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _evidence():
    return {
        "global_scene": {
            "physical_entities": [
                {"entity_id": "entity_001"},
                {"entity_id": "entity_002"},
            ]
        },
        "video1_analysis": {
            "information_atoms": [{"atom_id": "v1_atom_001"}],
            "uncertain_observations": [{"uncertainty_id": "v1_unc_001"}],
            "missing_key_attributes": [],
        },
        "video2_analysis": {
            "information_atoms": [
                {"atom_id": "v2_atom_001"},
                {"atom_id": "v2_atom_002"},
            ],
            "uncertain_observations": [],
            "missing_key_attributes": [{"entity_id": "entity_001"}],
        },
    }


def _record(*, evidence, first_attempt, warnings=None, final_error=None, retry_history=None):
    return {
        "status": "generated" if evidence is not None else "failed",
        "evidence": evidence,
        "final_error_category": final_error,
        "validation_warnings": warnings or [],
        "diagnostics": {
            "api_calls": 2,
            "validation_attempts": 2,
            "transport_retries": 1,
            "first_validation_attempt_success": first_attempt,
            "retry_history": retry_history or [],
        },
    }


def _metrics(payload, run_id="run_001"):
    return extract_run_metrics(
        payload,
        sample_id="bike_night",
        run_id=run_id,
        evaluation_set="pass1_dev_v1",
        prompt_version=BASELINE_PROMPT_VERSION,
        raw_output_path=f"outputs/pass1_evaluation/pass1_dev_v1/bike_night/{run_id}.json",
    )


def test_manifest_parsing(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path / "manifest.json"))

    assert manifest["evaluation_set"] == "pass1_dev_v1"
    assert manifest["prompt_version"] == BASELINE_PROMPT_VERSION
    assert manifest["runs_per_sample"] == 2
    assert manifest["samples"][0]["sample_id"] == "bike_night"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prompt_version": "unknown"}, "freezes"),
        ({"runs_per_sample": 0}, "positive integer"),
        ({"pipeline": {"unknown_option": True}}, "unsupported keys"),
        ({"samples": []}, "non-empty list"),
    ],
)
def test_manifest_rejects_invalid_contract(tmp_path, overrides, message):
    path = _write_manifest(tmp_path / "manifest.json", **overrides)

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def test_successful_output_metric_extraction():
    payload = {
        "metadata": {"model_name": "gemini-test", "input": "input.json"},
        "items": [
            _record(
                evidence=_evidence(),
                first_attempt=False,
                warnings=["warning one", "warning two"],
                retry_history=[{"type": "validation", "category": "generic_sensor_theory"}],
            )
        ],
        "skipped": [],
    }

    metrics = _metrics(payload)

    assert metrics["status"] == "success"
    assert metrics["completed_item_count"] == 1
    assert metrics["first_validation_attempt_success"] is False
    assert metrics["validation_attempt_count"] == 2
    assert metrics["api_call_count"] == 2
    assert metrics["transport_retry_count"] == 1
    assert metrics["validation_warning_count"] == 2
    assert metrics["physical_entity_count"] == 2
    assert metrics["video1_atom_count"] == 1
    assert metrics["video2_atom_count"] == 2
    assert metrics["video1_uncertainty_count"] == 1
    assert metrics["video2_missing_target_count"] == 1


def test_failed_output_metric_extraction():
    payload = {
        "metadata": {},
        "items": [],
        "skipped": [
            _record(
                evidence=None,
                first_attempt=False,
                final_error="shared_global_source_attribute_leakage",
            )
        ],
    }

    metrics = _metrics(payload)

    assert metrics["status"] == "failed"
    assert metrics["completed_item_count"] == 0
    assert metrics["skipped_item_count"] == 1
    assert metrics["final_error_category"] == "shared_global_source_attribute_leakage"
    assert metrics["final_error_categories"] == ["shared_global_source_attribute_leakage"]


def test_null_evidence_does_not_crash_metric_extraction():
    payload = {
        "metadata": {},
        "items": [{"evidence": None, "validation_warnings": []}],
        "skipped": [],
    }

    metrics = _metrics(payload)

    assert metrics["status"] == "success"
    assert metrics["physical_entity_count"] == 0
    assert metrics["video1_atom_count"] == 0
    assert metrics["video2_missing_target_count"] == 0


def test_aggregate_statistic_computation(tmp_path):
    success = _metrics({
        "metadata": {},
        "items": [_record(evidence=_evidence(), first_attempt=True, warnings=["warning"])],
        "skipped": [],
    })
    failure = _metrics({
        "metadata": {},
        "items": [],
        "skipped": [_record(evidence=None, first_attempt=False, final_error="parse_error")],
    }, run_id="run_002")

    summary = aggregate_metrics([success, failure], "pass1_dev_v1", BASELINE_PROMPT_VERSION)

    assert summary["total_runs"] == 2
    assert summary["successful_runs"] == 1
    assert summary["failed_runs"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["first_attempt_success_rate"] == 0.5
    assert summary["mean_validation_attempts"] == 2
    assert summary["mean_api_calls"] == 2
    assert summary["error_category_counts"] == {"parse_error": 1}
    assert summary["warning_count"] == 1
    assert summary["per_sample"]["bike_night"]["total_runs"] == 2

    write_reports(tmp_path, [success, failure], "pass1_dev_v1", BASELINE_PROMPT_VERSION)
    assert json.loads((tmp_path / "summary.json").read_text())["total_runs"] == 2
    with open(tmp_path / "summary.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_unique_run_filenames(tmp_path):
    run_id1, output1, lock1 = next_run_path(tmp_path)
    run_id2, output2, lock2 = next_run_path(tmp_path)

    assert run_id1 == "run_001"
    assert run_id2 == "run_002"
    assert output1 != output2
    assert lock1.exists()
    assert lock2.exists()


def test_retry_error_categories_preserve_order_and_duplicates():
    payload = {
        "metadata": {},
        "items": [],
        "skipped": [
            _record(
                evidence=None,
                first_attempt=False,
                final_error="parse_error",
                retry_history=[
                    {"type": "validation", "category": "parse_error"},
                    {"type": "validation", "category": "invalid_reference"},
                    {"type": "validation", "category": "parse_error"},
                ],
            )
        ],
    }

    metrics = _metrics(payload)

    assert metrics["retry_error_categories"] == [
        "parse_error",
        "invalid_reference",
        "parse_error",
    ]
