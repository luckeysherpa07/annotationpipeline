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

FORBIDDEN_MECHANISM_PATTERNS = (
    re.compile(r"\bactivations?\b", re.I),
    re.compile(r"\bedge[\s-]?onset\b", re.I),
    re.compile(r"\bresponse\s+maps?\b", re.I),
    re.compile(r"\bzero[\s-]?activation\b", re.I),
    re.compile(r"\b(?:dense\s+)?edge\s+activity\b", re.I),
    re.compile(r"\bevent\s+activity\b", re.I),
    re.compile(r"\bclusters?\s+of\s+(?:edge\s+)?activations?\b", re.I),
    re.compile(r"\bgenerated\s+activations?\b", re.I),
    re.compile(r"\bdata\s+represents?\s+only\b", re.I),
    re.compile(r"\b(?:is|are|were)\s+(?:rendered|represented|captured|encoded|resolved|visualized|depicted|highlighted)\s+(?:as|by)(?:\s+\w+){0,5}\s+(?:silhouettes?|boundaries|boundary|patterns?|signals?|responses?|events?|clusters?|pixels?|maps?|traces?|contours?)\b", re.I),
)
FORBIDDEN_MECHANISM_MESSAGE = (
    "Rewrite to describe the physical world (e.g., 'parked vehicle outlines remain distinct' instead of 'edge activations define the car'). Do not use mechanism-oriented or representation-oriented wording."
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
    re.compile(r"\b(?:the|this)\s+(?:event|ir|depth|rgb|thermal)?\s*(?:sensor|camera|modality)\s+(?:registers|records|captures|detects|measures|cannot|does not)\b", re.I),
    re.compile(r"\b(?:the|this)\s+(?:sensing|imaging)\s+process\b", re.I),
    re.compile(r"\bonly\s+(?:temporal|intensity|contrast)\s+changes\b", re.I),
    re.compile(r"\bstatic(?:\s+background)?\s+regions\s+generate\s+no\b", re.I),
    re.compile(r"\bmovement\s+generates\b", re.I),
    re.compile(r"\b(?:the|this)\s+modality\s+does\s+not\s+provide\b", re.I),
    re.compile(r"\b(?:inability|unable)\s+to\s+(?:capture|detect|record)\b", re.I),
    re.compile(r"\b(?:loss|lack)\s+of\s+(?:color|absolute|illumination)\b", re.I),
    re.compile(r"\bzero\s+response\s+on\b", re.I),
    re.compile(r"\bhigh\s+sensitivity\s+to\b", re.I),
    re.compile(r"\black\s+of\s+signal\s+response\b", re.I),
)

# ─── Section 1: Structured capability constraints (machine-readable) ───
MODALITY_CAPABILITIES = {
    "rgb":   {"color": "direct",       "visual_category": "direct",      "thermal": "not_direct", "structure_edge": "direct",      "depth": "conditional"},
    "event": {"color": "not_direct",   "visual_category": "conditional", "thermal": "not_direct", "structure_edge": "direct",      "depth": "not_direct"},
    "ir":    {"color": "not_direct",   "visual_category": "conditional", "thermal": "not_direct", "structure_edge": "conditional", "depth": "not_direct"},
    "depth": {"color": "not_direct",   "visual_category": "conditional", "thermal": "not_direct", "structure_edge": "conditional", "depth": "direct"},
}

# ─── Section 2: Per-modality exclusive observation cues (LLM prompt fragments) ───
# These describe what each modality is EXCLUSIVELY able to observe and what it
# cannot observe. Used to generate the CROSS-MODAL DIFFERENTIATION block in prompts.
MODALITY_EXCLUSIVE_CUES: dict[str, dict[str, str]] = {
    "rgb": {
        "exclusive":    "visible color, surface texture, paint and material appearance, visible lighting effects",
        "not_visible":  "short-lived spatial changes around moving entities, physical outlines distinguishable under poor illumination, relative temperature differences, absolute metric distance",
    },
    "event": {
        "exclusive":    "changes in physical object boundaries across sampled times, changes in object position, extent, or silhouette, physical outlines that remain distinguishable under poor illumination, short-lived spatial changes around moving entities",
        "not_visible":  "visible color, surface texture, paint and material appearance, stationary objects without lighting changes",
    },
    "ir": {
        "exclusive":    "relative infrared appearance, one region being brighter or darker in infrared, coarse object or structural contrast",
        "not_visible":  "visible color, paint and material appearance, fine texture details, exact temperature, overheating, thermal physical causes",
    },
    "depth": {
        "exclusive":    "one entity standing in front of or behind another, relative or metric distance when directly supported, physical surface geometry and spatial separation",
        "not_visible":  "visible color, paint and material appearance, thermal differences, fine surface texture",
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




def normalize_modality_name(modality: str) -> str:
    if not modality:
        return ""
    n = modality.casefold().strip()
    if n in ("rgb", "video", "visual", "optical"):
        return "rgb"
    if n in ("event", "dvs", "event camera"):
        return "event"
    if n in ("ir", "infrared", "thermal"):
        return "ir"
    if n in ("depth", "disparity", "range"):
        return "depth"
    return n

SUPPORTED_MODALITIES = tuple(MODALITY_CAPABILITIES.keys())
