from pathlib import Path

import pytest

from annotation_feature.aligned_caption_pass1_prompt import (
    build_modality_physical_guidance,
    build_pass1_system_prompt,
    build_pass1_user_prompt,
)


class DummyTask:
    segment_id = "segment"
    side = "night"
    composite_frames = [Path("frame_000000.png")]
    modality1 = "rgb"
    modality2 = "event"


@pytest.mark.parametrize(
    ("modality1", "modality2", "present", "absent"),
    [
        ("rgb", "event", ("RGB", "EVENT"), ("DEPTH", "IR")),
        ("rgb", "depth", ("RGB", "DEPTH"), ("EVENT", "IR")),
        ("depth", "ir", ("DEPTH", "IR"), ("RGB", "EVENT")),
    ],
)
def test_only_active_modality_guidance_is_injected(modality1, modality2, present, absent):
    prompt = build_pass1_user_prompt(DummyTask(), modality1, modality2)
    for name in present:
        assert prompt.count(f"### ACTIVE MODALITY GUIDANCE: {name}") == 1
    for name in absent:
        assert f"### ACTIVE MODALITY GUIDANCE: {name}" not in prompt


def test_duplicate_modality_is_deduplicated_but_sources_remain_explicit():
    prompt = build_pass1_user_prompt(DummyTask(), "rgb", "rgb")
    assert prompt.count("### ACTIVE MODALITY GUIDANCE: RGB") == 1
    assert "Source 1 modality: rgb" in prompt
    assert "Source 2 modality: rgb" in prompt


def test_unknown_modality_fails_strictly():
    with pytest.raises(ValueError, match="Unsupported modality"):
        build_modality_physical_guidance("rgb", "lidar")


def test_pair_guidance_is_pair_specific():
    assert "### ACTIVE PAIR GUIDANCE: RGB + EVENT" in build_modality_physical_guidance("rgb", "event")
    assert "### ACTIVE PAIR GUIDANCE: RGB + EVENT" not in build_modality_physical_guidance("rgb", "depth")


def test_common_guidance_is_stable_and_not_duplicated_in_dynamic_blocks():
    system_prompt = build_pass1_system_prompt()
    assert system_prompt.count("PHYSICAL-WORLD SUBJECT TEST") == 1
    assert "PHYSICAL-WORLD SUBJECT TEST" not in build_modality_physical_guidance("rgb", "event")
