"""Shared schema constants and small helpers for aligned multimodal captions."""

from __future__ import annotations

import re
from typing import Any

class CaptionParseError(Exception):
    """Raised when the response cannot be parsed as valid JSON or lacks the expected top-level structure."""
    pass

class CaptionValidationError(Exception):
    """Raised when the parsed JSON fails semantic schema validation."""
    pass

CAPTION_REQUIRED_TOP_LEVEL_FIELDS = {
    "global_scene",
    "video1_analysis",
    "video2_analysis",
    "cross_modal_evidence_links",
    "information_gain",
    "reasoning_events",
    "ambiguity_events",
    "qa_relevant_details",
    "rejected_observations",
}
ALLOWED_MISSING_ATTRIBUTE_TYPES = {
    "existence",
    "semantic_identity",
    "physical_cause",
    "surface_attribute",
    "state_attribute",
    "motion_state",
    "spatial_relation",
    "temporal_relation",
    "count",
    "fine_grained_category",
}
ALLOWED_GAIN_RATINGS = {"low", "medium", "high"}
ALLOWED_CROSS_MODAL_CONTRIBUTION_DIRECTIONS = {
    "video1_improves_video2",
    "video2_improves_video1",
    "mutual_complementarity",
    "confirmation_only",
}
ALLOWED_GAIN_TYPES = {
    "semantic_emergence",
    "disambiguation",
    "complementarity",
    "confirmation",
    "confidence_gain",
}
ALLOWED_QA_REASONING_PATTERNS = {
    "cross_modal_disambiguation", "temporal_integration", "occlusion_reasoning", 
    "interaction_reasoning", "spatial_transition", "hypothesis_elimination", 
    "multi_hop_composition", "joint_fusion"
}
ALLOWED_REASONING_EVENT_TYPES = {
    "confirmation",
    "cross_modal_complementarity",
    "unidirectional_disambiguation",
    "temporal_change",
    "interaction",
    "occlusion_change",
    "spatial_transition",
    "joint_fusion",
}
ALLOWED_AMBIGUITY_DIRECTIONS = {"video1_resolves_video2", "video2_resolves_video1"}
FORBIDDEN_SENSOR_QUALITY_TERMS = (
    "modality",
    "rgb",
    "infrared",
    "ir",
    "thermal camera",
    "thermal image",
    "thermal frame",
    "thermal modality",
    "event camera",
    "event stream",
    "event sensor",
    "event frame",
    "event modality",
    "depth camera",
    "depth sensor",
    "depth map",
    "depth frame",
    "depth modality",
    "edge map",
    "edge-based",
    "edge-like",
    "heat signature",
    "heat map",
    "blurry",
    "noisy",
    "pixel",
    "pixels",
    "grayscale",
    "greyscale",
    "monochrome",
    "overexposed",
    "saturated",
)
FORBIDDEN_SENSOR_QUALITY_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in FORBIDDEN_SENSOR_QUALITY_TERMS)
    + r")\b",
    re.I,
)
FORBIDDEN_SENSOR_QUALITY_MESSAGE = (
    "EXACT BLOCKLIST: modality, rgb, infrared, ir, thermal camera/image/frame/modality, "
    "event camera/stream/sensor/frame/modality, depth camera/sensor/map/frame/modality, "
    "edge map/edge-based/edge-like, heat signature/map, blurry, noisy, pixel/pixels, "
    "grayscale/greyscale, monochrome, overexposed, saturated. REMOVE THESE TERMS!"
)

FORBIDDEN_GLOBAL_SCENE_TERMS = (
    "camera",
    "lens",
    "viewpoint",
    "frame",
)
FORBIDDEN_GLOBAL_SCENE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in FORBIDDEN_GLOBAL_SCENE_TERMS) + r")\b",
    re.I,
)
FORBIDDEN_GLOBAL_SCENE_MESSAGE = (
    "EXACT BLOCKLIST for global_scene: camera, lens, viewpoint, frame. REMOVE THESE TERMS!"
)

FORBIDDEN_INFERENTIAL_TERMS = (
    "corresponding",
    "suggesting",
    "indicating",
    "implying",
    "consequently",
    "representing",
    "associated",
)
FORBIDDEN_INFERENTIAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in FORBIDDEN_INFERENTIAL_TERMS) + r")\b",
    re.I,
)
FORBIDDEN_INFERENTIAL_MESSAGE = (
    "EXACT BLOCKLIST (inferential terms): corresponding, suggesting, indicating, implying, "
    "consequently, representing, associated. REMOVE THESE TERMS!"
)

MIN_DETAILED_CAPTION_WORDS = 30
MIN_SCENE_SUMMARY_WORDS = 20
MIN_FRAME_DETAIL_WORDS = 8
MIN_UNCERTAIN_OBSERVATION_HYPOTHESES = 1
MIN_AMBIGUITY_EVENT_HYPOTHESES = 2
GENERIC_SENSOR_EXPLANATION_PATTERNS = (
    re.compile(r"\bevent cameras?\s+(capture|detect|record|respond)", re.I),
    re.compile(r"\b(depth|rgb|infrared|ir)\s+(camera|sensor)s?\s+(capture|detect|record|measure)", re.I),
    re.compile(r"\bthis modality\s+(captures|detects|records|measures)", re.I),
    re.compile(r"\bdesigned to\s+(capture|detect|record|measure)", re.I),
    re.compile(r"\b(inability|unable)\s+to\s+(capture|detect|record)", re.I),
    re.compile(r"\b(loss|lack)\s+of\s+(color|absolute|illumination)", re.I),
    re.compile(r"\bzero\s+response\s+on", re.I),
    re.compile(r"\bhigh\s+sensitivity\s+to", re.I),
    re.compile(r"\black\s+of\s+signal\s+response", re.I),
)

# ─── Section 1: Structured capability constraints (machine-readable) ───
MODALITY_CAPABILITIES = {
    "rgb":   {"color": "direct",       "visual_category": "direct",      "thermal": "not_direct", "structure_edge": "direct",      "depth": "conditional"},
    "event": {"color": "not_direct",   "visual_category": "conditional", "thermal": "not_direct", "structure_edge": "direct",      "depth": "not_direct"},
    "ir":    {"color": "not_direct",   "visual_category": "conditional", "thermal": "direct",     "structure_edge": "conditional", "depth": "not_direct"},
    "depth": {"color": "not_direct",   "visual_category": "conditional", "thermal": "not_direct", "structure_edge": "conditional", "depth": "direct"},
}

# ─── Section 2: Per-modality exclusive observation cues (LLM prompt fragments) ───
# These describe what each modality is EXCLUSIVELY able to observe and what it
# cannot observe. Used to generate the CROSS-MODAL DIFFERENTIATION block in prompts.
MODALITY_EXCLUSIVE_CUES: dict[str, dict[str, str]] = {
    "rgb": {
        "exclusive":    "surface color of objects, texture details (e.g. cobblestone vs asphalt), specular highlights on glass, overexposed regions where texture is lost, surface reflectance patterns",
        "not_visible":  "motion onset/offset edge dynamics, high-frequency structural outlines invisible in static frames, regions of zero activation on uniform surfaces",
    },
    "event": {
        "exclusive":    "edge-onset of moving objects, contour density of high-frequency texture borders, zero-activation zones (smooth low-contrast surfaces like painted car doors), motion direction implied by event polarity gradients",
        "not_visible":  "absolute surface color, spectral contrast, texture details on uniform or slowly-moving surfaces",
    },
    "ir": {
        "exclusive":    "heat signatures, emissivity contrast, temperature gradients, active cooling/heating sources, thermal radiation from engine components",
        "not_visible":  "visible surface color, fine texture details, spectral highlights",
    },
    "depth": {
        "exclusive":    "absolute metric distance, 3D surface geometry, foreground/background segmentation boundaries, depth discontinuities between adjacent surfaces",
        "not_visible":  "surface color, texture, thermal information, fine visual detail of surfaces",
    },
}

def _describe_capability_pair(attr: str, mod1: str, state1: str, mod2: str, state2: str) -> str:
    if state1 == "direct" and state2 == "direct":
        return f"- {attr}: Both videos may provide direct evidence for this cue. Final claims must still be grounded in the supplied frames."
    elif state1 == "direct" and state2 == "conditional":
        return f"- {attr}: Video 1 ({mod1}) may provide direct evidence. Video 2 ({mod2}) may contribute only when the supplied frames visibly support the cue."
    elif state1 == "conditional" and state2 == "direct":
        return f"- {attr}: Video 2 ({mod2}) may provide direct evidence. Video 1 ({mod1}) may contribute only when the supplied frames visibly support the cue."
    elif state1 == "direct" and state2 == "not_direct":
        return f"- {attr}: Video 1 ({mod1}) may provide direct evidence. Do not infer this cue from Video 2 ({mod2}) alone."
    elif state1 == "not_direct" and state2 == "direct":
        return f"- {attr}: Video 2 ({mod2}) may provide direct evidence. Do not infer this cue from Video 1 ({mod1}) alone."
    elif state1 == "conditional" and state2 == "conditional":
        return f"- {attr}: Either video may contribute only when the supplied frames visibly support the cue. Do not assume the cue from modality type alone."
    elif state1 == "conditional" and state2 == "not_direct":
        return f"- {attr}: Video 1 ({mod1}) may provide conditional evidence when visibly supported. Do not infer this cue from Video 2 ({mod2}) alone."
    elif state1 == "not_direct" and state2 == "conditional":
        return f"- {attr}: Video 2 ({mod2}) may provide conditional evidence when visibly supported. Do not infer this cue from Video 1 ({mod1}) alone."
    else:
        return f"- {attr}: This cue is not directly supported by either modality according to the capability prior. Do not claim it unless the supplied frames provide exceptional directly observable evidence."

def build_modality_constraint_block(mod1: str, mod2: str) -> str:
    default_caps = {"color": "conditional", "visual_category": "conditional", "thermal": "conditional", "structure_edge": "conditional", "depth": "conditional"}
    h = MODALITY_CAPABILITIES.get(mod1, default_caps)
    v = MODALITY_CAPABILITIES.get(mod2, default_caps)
    
    lines = []
    for attr, cap_name in [
        ("color/paint", "color"), 
        ("visually discernible vehicle types/categories", "visual_category"), 
        ("thermal/heat", "thermal"), 
        ("structural edges/motion boundaries", "structure_edge"), 
        ("metric depth/distance", "depth")
    ]:
        h_can = h[cap_name]
        v_can = v[cap_name]
        lines.append(_describe_capability_pair(attr, mod1, h_can, mod2, v_can))
    
    capability_block = "MODALITY CAPABILITY CONSTRAINTS:\n" + "\n".join(lines)
    
    default_cues = {"exclusive": "general visual observations", "not_visible": "unknown"}
    cues1 = MODALITY_EXCLUSIVE_CUES.get(mod1, default_cues)
    cues2 = MODALITY_EXCLUSIVE_CUES.get(mod2, default_cues)
    
    differentiation_block = "\n".join([
        f"CROSS-MODAL DIFFERENTIATION GUIDE ({mod1} + {mod2}):",
        f"- {mod1}-EXCLUSIVE atoms should capture: {cues1['exclusive']}.",
        f"- {mod2}-EXCLUSIVE atoms should capture: {cues2['exclusive']}.",
        "- SHARED atoms are allowed ONLY for large-scale physical events observable through both modalities (e.g. coarse object presence, global scene change). Do NOT duplicate fine-grained perceptual details across modalities."
    ])
    
    return capability_block + "\n\n" + differentiation_block

def _enum_line(name: str, values: set[str]) -> str:
    return f"- {name}: {', '.join(sorted(values))}"

def _normalize_license_plates(data: Any) -> Any:
    if isinstance(data, str):
        pattern = r'\b([A-Z]{1,3})[\s\-]+([A-Z]{1,2})[\s\-]*(\d{1,4})\b'
        return re.sub(pattern, r'\1-\2 \3', data)
    elif isinstance(data, list):
        return [_normalize_license_plates(item) for item in data]
    elif isinstance(data, dict):
        return {k: _normalize_license_plates(v) for k, v in data.items()}
    return data


