#!/usr/bin/env python
from annotation_feature.fusion import (
    CaptionEvidence,
    _assign_evidences_to_sections,
    _tokenize,
)


def make_evidence(modality: str, annotation_key: str, sentence: str, *, support_score: float = 0.35, reliability: float = 0.85) -> CaptionEvidence:
    return CaptionEvidence(
        modality=modality,
        source_key=f"sample_{modality}",
        annotation_key=annotation_key,
        caption=sentence,
        sentence=sentence,
        normalized_tokens=_tokenize(sentence),
        support_score=support_score,
        modality_reliability=reliability,
    )


evidences = [
    make_evidence("rgb", "scene_overview_main", "A kitchen counter is visible in the room."),
    make_evidence("rgb", "spatial_reasoning_layout", "The cutting board is to the right of the bowl."),
    make_evidence("event", "event_dynamic_action", "Hands peel the carrot and move it across the board."),
    make_evidence("audio", "audio_sound_description", "A chopping sound is heard in the background."),
]

selected = _assign_evidences_to_sections(evidences, min_score=0.2, relax_filters=False)

assert selected["scene_overview"][0].sentence == "A kitchen counter is visible in the room."
assert selected["visible_objects_and_layout"][0].sentence == "The cutting board is to the right of the bowl."
assert selected["motion_and_event_cues"][0].sentence == "Hands peel the carrot and move it across the board."
assert selected["audio_cues"][0].sentence == "A chopping sound is heard in the background."

print("section assignment routing OK")
