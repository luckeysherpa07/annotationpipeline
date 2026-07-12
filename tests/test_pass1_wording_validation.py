import pytest
from annotation_feature.aligned_caption_validation import (
    _validate_physical_world_wording,
    _validate_no_generic_sensor_explanation,
    CaptionValidationError
)

def test_representation_wording_fails():
    with pytest.raises(CaptionValidationError, match="representation-oriented"):
        _validate_physical_world_wording("The vehicle is rendered as a clear silhouette.", "fact")
    
    with pytest.raises(CaptionValidationError, match="representation-oriented"):
        _validate_physical_world_wording("The vehicle is represented by dense boundaries.", "fact")
        
    with pytest.raises(CaptionValidationError, match="representation-oriented"):
        _validate_physical_world_wording("The lamppost is highlighted as a vertical response pattern.", "fact")

def test_representation_wording_passes_normal_physical():
    # These should NOT raise an exception
    _validate_physical_world_wording("The vehicle has a clear silhouette.", "fact")
    _validate_physical_world_wording("The wall boundary is visible.", "fact")
    _validate_physical_world_wording("The headlights highlight the wet asphalt.", "fact")
    _validate_physical_world_wording("The painted boundary is visible.", "fact")
    _validate_physical_world_wording("The wall seam is clearly resolved.", "fact")
    _validate_physical_world_wording("The vehicle has a rectangular silhouette.", "fact")

def test_generic_theory_fails():
    with pytest.raises(CaptionValidationError, match="generic sensor-theory"):
        _validate_no_generic_sensor_explanation("The event sensor registers only temporal contrast changes.", "why_missing")
        
    warnings = []
    _validate_no_generic_sensor_explanation("Static regions generate no event boundaries.", "sensor_limitations", hard_fail=False, warnings=warnings)
    assert len(warnings) == 1
    assert "generic sensor-theory" in warnings[0]

def test_generic_theory_passes_segment_specific():
    warnings = []
    _validate_no_generic_sensor_explanation(
        "The vehicle body in frames 000120–000360 contains no stable internal surface detail sufficient to distinguish paint color or texture.",
        "why_missing"
    )
    _validate_no_generic_sensor_explanation(
        "Large dark background regions in frames 000000–000360 contain few distinguishable internal boundaries, leaving their surface structure unresolved.",
        "sensor_limitations", hard_fail=False, warnings=warnings
    )
    assert len(warnings) == 0
