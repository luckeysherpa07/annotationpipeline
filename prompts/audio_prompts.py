################################################################################
#                           AUDIO PROMPTS LIBRARY
################################################################################
# This module contains specialized prompts for comprehensive audio-visual
# analysis of egocentric videos. Prompts guide AI models to extract and
# structure information about:
#   • Human-object interactions in video
#   • Detailed audio-visual caption generation
#   • Question-Answer pair extraction from captions
################################################################################

# ============================================================================
# HUMAN_INTERACTION_ANNOTATION_PROMPT
# ============================================================================
# Prompt for identifying and annotating human-object interactions in video,
# including interaction types, involved objects, and temporal boundaries.
# ============================================================================

HUMAN_INTERACTION_ANNOTATION_PROMPT = {
    "caption_prompt": (
        "Identify all interactions between humans and objects in the video, "
        "including the type of interaction, the objects involved, and the "
        "temporal boundaries of each interaction."
    ),
}


# ============================================================================
# AUDIO_VISUAL_CAPTION_GENERATION_PROMPT
# ============================================================================
# Detailed prompt for generating comprehensive audio-visual captions that
# combine auditory signals with visual context from egocentric videos.
# ============================================================================

AUDIO_VISUAL_CAPTION_GENERATION_PROMPT = """You are an expert audio-visual analyst specializing in egocentric videos.
You will receive two inputs:
  (1) Human Interaction Annotations (HIA)
  (2) The corresponding video segment

Your task is to generate detailed and unambiguous captions describing all
meaningful sound events by combining auditory signals with visually grounded
human-object interactions.
---
CORE EXECUTION LOGIC:

Your process MUST begin by identifying all meaningful sounds, including speech
and environmental sounds. For each sound, you must:
  1. Locate its position in time
  2. Link it to relevant visual actions indicated by the HIA (if available)
  3. Describe the event in detail using the tasks listed in Your Goal below

Each description must integrate:
  • Audio perception
  • Visual grounding
  • Temporal alignment
  • Causal reasoning
---
YOUR GOAL: DETAILED SOUND ANALYSIS

Focus on all sounds (human speech and environmental sounds). Address the
following seven tasks as much as possible:

1. SOUND CHARACTERISTICS
   Describe intrinsic acoustic properties: volume, texture, timbre
   Example: "a sharp clink", "a soft rustle", "a loud bang"

2. COUNTING
   Number of distinct sound events or repetitions

3. TEMPORAL ATTRIBUTE
   Occurrence timing, duration, onset/offset characteristics, volume variation
   Example: "continuous sound with gradually decreasing volume over 5 seconds"

4. SPATIAL LOCATION
   Relative location of source to camera
   Example: "knock from the left, behind, 1m"
           "distant siren from the right, 10m"
           "nearby footstep from behind, 0.5m"

5. SOUND SOURCE IDENTIFICATION
   What object or action (visual event) created the sound?
   Example: "cup hitting table", "birds chirping", "people talking",
            "hand places mug on counter with loud thud"

6. INFERENTIAL CAUSALITY
   Why a sound/speech event occurred. Focus on underlying reason, motivation,
   context. Leverage chronological context of surrounding events.
   Example: "User closed window to block loud shouting from outside"

7. CROSS-MODAL REASONING
   How audio events influence visual understanding and vice versa
   • Audio-Guided Visual Reasoning: Use audio cues to ground what is/will
     happen visually
   • Visual-Guided Audio Reasoning: Use visual cues to ground what is/will
     happen in audio
---
PRINCIPLES: FILTERING AND COHERENCE

✓ CORE RULE: Output must be strictly chronological and non-redundant

✓ FOCUS ON:
  • Interactions and human actions
  • Clear cause-effect audio-visual relationships

✗ MUST IGNORE:
  • Silent segments
  • Idle scenes (e.g., just standing)
  • Repetitive actions without significant meaning

✗ DO NOT:
  • Narrate per-second; only describe meaningful audio events
  • Hallucinate visual or audio details

✓ TERMINATION: Stop generating once you have described all meaningful
  sound events.
---
FINAL OUTPUT: CHRONOLOGICAL, EXCEEDINGLY DETAILED AUDIOVISUAL CAPTION

Your final output must be exceedingly detailed and form a coherent caption
combining audio and visual reasoning.

STRUCTURE:
  Use chronological timeline format: [start time – end time] event description
  Use seconds as timestamps (e.g., 00:12 - 00:15)

FORMAT:
  Each line MUST represent one distinct, non-overlapping audio event

SPEECH HANDLING:
  If an event includes speech/dialogue, embed the transcribed text using
  quotation marks or clear label
  Example: "The user says: ..."

STYLE:
  • Be concise but highly informative
  • Avoid redundancy and trivial details:
    - "standing", "breathing", "background wall", "nothing happens"
  • Prioritize meaningful sound events and their significance"""


# ============================================================================
# QNA_GENERATION_PROMPT
# ============================================================================
# Prompt for extracting high-quality Question–Answer pairs from audio-visual
# captions, focusing on seven core sound-centric analysis tasks.
# ============================================================================

QNA_GENERATION_PROMPT = """You are an expert AI for sound-centric information extraction. Read the
detailed audiovisual caption and generate high-quality Question–Answer (QA)
pairs that cover the seven core sound-centric tasks.

BASIC PRINCIPLE:
  Base every question and answer solely on information contained in the
  caption or on inferences that naturally follow from its context. Use the
  accompanying video frames for double validation: confirm that each QA pair
  is fully supported by visual evidence, correcting or discarding any detail
  that cannot be verified.

✓ Ensure all QA pairs remain factual, precise, and hallucination-free
✗ DO NOT introduce content beyond what is grounded in caption or video frames
---
CORE OBJECTIVE:

Every question you generate must target one of the following seven core tasks.
---
SEVEN CORE TASKS (Question Focus):

1. SOUND CHARACTERISTICS
   Ask about the sound's acoustic qualities: volume, texture, timbre.
   Examples:
   • "What is the intensity and volume of distant traffic noise?"
   • "Was the 'tap' at 01:10 sharp, muffled, dull, or resonant?"

2. COUNTING
   Ask about count, number of repetitions, or frequency of sound events.
   Examples:
   • "How many types of environmental sounds (excluding speech) were
     simultaneously present at 00:45?"
   • "How many times did the 'clink' sound occur?"
   • "How many distinct, quick footsteps were recorded between 02:00
     and 02:05?"

3. TEMPORAL ATTRIBUTE
   Ask about sound's timing, duration, or volume fluctuation.
   Examples:
   • "When did the 'clink' sound start and how long did it last?"
   • "What is the duration and volume change of the high-frequency
     whining sound heard between 01:30 and 01:35?"

4. SPATIAL LOCATION
   Ask about sound's precise location (Direction & Distance) relative to
   camera.
   Examples:
   • "What was the spatial location (direction and estimated distance in
     meters) of the faint rustling sound at 00:15?"
   • "Did the close-range speech originate from the left side, right side,
     or directly in front of the camera?"

5. SOUND SOURCE IDENTIFICATION
   Ask what specific object, person, or action generated the sound.
   Examples:
   • "What object generated the 'clink' sound at 00:15?"
   • "What was the source of the high-pitched metallic scraping sound
     at 00:15?"

6. INFERENTIAL CAUSALITY
   Ask why a sound/speech event occurred. Focus on underlying reason,
   motivation, or context by leveraging chronological context.
   Examples:
   • "What was the likely reason for the car's sudden horn honk at 01:10?"
   • "Why did the speaker laugh at 00:45?"
   • "Based on preceding events, why did the mom say 'I'm so proud'?"

7. CROSS-MODAL REASONING
   Ask how audio events influence visual understanding or vice versa,
   requiring cross-modal inference.

   Audio-Guided Visual Reasoning (Sound to Visual):
   Use audio cues to ground and interpret what is/will happen visually.
   Examples:
   • "After the loud crash sound at 00:22, what object most likely fell?"
   • "When the barking sound occurs, where is the dog located in scene?"

   Visual-Guided Audio Reasoning (Visual to Sound):
   Use visual cues to ground and interpret what is/will happen in audio.
   Examples:
   • "When the man slammed the door at 00:45, what sound followed?"
   • "After the woman claps her hands, what sound follows and what does
     it suggest about the environment?"
---
EXECUTION PRINCIPLES:

✓ QUALITY FIRST
  Emphasis on QA quality, not absolute quantity. Generate only meaningful
  questions with clear answers within the text.

✓ STRICTLY GROUNDED
  All QA pairs must be inferred from the provided caption text.
  ✗ DO NOT hallucinate or invent details not present in text.

✓ FILTER IRRELEVANCE
  Skip lines with no specific details related to the 7 tasks.
  Example: "The user is walking" with no sound description → SKIP

✓ NO DUPLICATION
  Avoid generating multiple, repetitive questions about the same event detail.

✓ VALIDATE AND PRUNE
  After generating QA pairs, re-check each question and answer against
  caption and video frames. Discard immediately if:
  • Contains incorrect information
  • Lacks proper support
  • Contains ambiguous content
---
OUTPUT FORMAT (JSON):

Return your answer as a single JSON list. Each QA pair is an object within
this list.

JSON STRUCTURE:
{
  "timestamp": "HH:MM - HH:MM",
  "context": "[HH:MM] Description of relevant events from caption",
  "question_type": "One of the 7 task types",
  "question": "Clear, specific question grounded in caption",
  "answer": "Precise answer based on caption/video content"
}

JSON EXAMPLE 1:
{
  "timestamp": "00:55 - 01:00",
  "context": "[00:55] A man outside the window shouts loudly. [00:58] The user
    closes the window with a definitive thud.",
  "question_type": "Inferential Causality",
  "question": "Based on the preceding event, what was the inferred reason for
    the user closing the window at 00:58?",
  "answer": "The user closed the window to block out the loud shouting man
    outside."
}

JSON EXAMPLE 2:
{
  "timestamp": "00:20 - 00:25",
  "context": "[00:20] The user's hand is seen dropping a key fob. [00:21] A loud,
    metallic clang is heard from the floor, 0.5m directly below the camera.",
  "question_type": "Sound Characteristics",
  "question": "What was the quality and volume of the sound made by the
    dropped key fob at 00:21?",
  "answer": "The sound was a 'loud, metallic clang'."
}"""