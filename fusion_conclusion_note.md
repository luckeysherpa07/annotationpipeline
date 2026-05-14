# Temporary Conclusion on Caption Fusion

## 1. Current Findings

Current conclusion:

- The main issue is not simple duplication.
- The current fusion pipeline is sample-specific enough in grouping, but the actual fusion logic is too template-driven.
- The fused output can suffer from three problems:
  - missing information because sentence selection is keyword- and score-filtered too aggressively
  - incomplete final captions because only the first sentence of each section is kept
  - wrong or irrelevant content because `cross_modal_details` contains hard-coded template sentences that are not generated from the current sample

What this means for the QA ground-truth step:

- The current fusion stage is not reliably producing a holistic caption from all modalities.
- This can contaminate downstream QA ground truth, because the fused caption may omit important modality evidence or include mismatched cross-modal content.

## 2. How Q/A is Generated in Fusion

### Data Flow: From Modality Captions to Fused Q/A

**Original modality data structure** (e.g., rgb_qa_results.json):
- Each modality file contains per-sample annotations
- Each annotation includes:
  - `caption`: A descriptive sentence or paragraph
  - `question`: A manually crafted question (high quality)
  - `answer`: A manually crafted answer (high quality)
- Example from rgb_qa_results.json:
  ```json
  "object_recognition": {
    "caption": "To the right of the cutting board sits an empty white bowl.",
    "question": "What electronic device is situated...",
    "answer": "A grey laptop."
  }
  ```

**Fusion process – Q/A generation steps:**

1. **Extract captions only** (NOT the original Q/A pairs):
   - `_extract_caption_evidence()` parses all modality JSON files
   - Extracts only the `caption` field from each annotation_key
   - Discards the original high-quality `question` and `answer` fields
   - Creates `CaptionEvidence` objects with: `sentence`, `modality`, `annotation_key`, `support_score`, `modality_reliability`

2. **Score and filter sentences** by section:
   - `_select_sentences()` ranks sentences using: base modality weight + reliability + support score + category bonus + keyword bonus
   - Filters by keyword gates (e.g., scene_overview and audio_cues require keyword_hits > 0 unless relaxed)
   - Filters by annotation_key type hints (e.g., visible_objects_and_layout prefers spatial, count, object annotations)
   - Returns top-k sentences per section (DEFAULT: 2-3, RELAXED: 4-6)

3. **Generate NEW Q/A pairs** from filtered captions (NOT reusing original Q/A):
   - `_generate_field_qa()` takes each selected caption sentence
   - Calls section-specific extractors:
     - `_extract_scene_answer(caption)` → matches regex patterns like "view of the [kitchen]", returns generic question "What setting is described in the scene overview?"
     - `_extract_layout_qa(caption)` → matches "To the [right] of the [board] is [bowl]", generates "What is to the right of the board?" and extracts answer "bowl"
     - `_extract_motion_qa(caption)` → matches motion patterns, generates motion-related Q/A
     - `_extract_audio_qa(caption)` → matches audio patterns, generates audio-related Q/A
     - `_extract_cross_modal_qa(caption)` → matches cross-modal patterns
     - `_extract_final_qa(caption)` → generates unified caption Q/A

4. **Trim answers** to max 6 words:
   - `_trim_answer(text, max_words=6)` forcibly truncates all answers
   - Removes leading articles (a, an, the)
   - Splits on separators (", followed by", ", and", ",", " and ", " while ", " as ", " because ", " which ")
   - Takes only first 6 words

### Why This Causes Problems

| Issue | Root Cause | Example |
|-------|-----------|---------|
| **Answer truncation** | `_trim_answer(max_words=6)` hard limit | "To the right of the cutting board sits an empty white bowl." → "To the right of the cutting" |
| **Question-answer mismatch** | Regex extractors are section-generic, may not match current caption context | Caption about spatial layout selected as scene_overview → generic question "What setting?" asked on layout info |
| **Lost semantic richness** | Original modality Q/A pairs (manually crafted) are discarded; new ones are template-generated | Original: "Where is the white bowl?" Answer: "To the right." → Fused: "What is to the right of the cutting board?" Answer: "To the right of the cutting" |
| **Coverage loss** | Some captions don't match any regex pattern → fall back to generic fallback Q/A | "The power strip has 3 sockets" doesn't match specific outlet regex → becomes generic "What object is explicitly mentioned?" |

### Comparison: Original vs. Fused Q/A Quality

**Original Q/A (from modality files):**
```json
{
  "annotation_key": "object_recognition",
  "caption": "To the right of the cutting board sits an empty white bowl.",
  "question": "What electronic device is situated on the black stovetop?",
  "answer": "A grey laptop."
}
```

**Fused Q/A (after fusion pipeline):**
```json
{
  "source_modality": "rgb",
  "annotation_key": "object_recognition",
  "caption": "To the right of the cutting board sits an empty white bowl.",
  "question": "What is to the right of the cutting board?",
  "answer": "To the right of the cutting",
  "fusion_score": 2.75,
  "is_outlier": false
}
```

**Issues in fused version:**
- Question changed from specific to generic pattern
- Answer truncated to 6 words (loses "board sits an empty white bowl")
- Original semantic meaning ("What electronic device?") is lost

## 3. Recommended Fix Approaches

### Option A: Preserve Original Modality Q/A
- Instead of extracting captions and regenerating Q/A, keep mappings to original Q/A pairs
- Modify `_extract_caption_evidence()` to store `original_qa` field
- In `_build_section_evidence_qas()`, reuse the original question and answer
- Advantage: No quality loss, preserves expert-crafted Q/A
- Disadvantage: Requires linking sentences back to original annotations; fusion question may not align with section semantics

### Option B: Improve Answer Extraction
- Modify `_trim_answer()` to extract complete noun phrases instead of word-count truncation
- Use spaCy or NLTK for phrase boundary detection
- Or, adjust regex extractors to be more context-aware
- Advantage: Keeps current pipeline, improves answer quality
- Disadvantage: May still not handle all edge cases; extractors remain template-driven

### Option C: Generate Q/A from Fused Caption
- After synthesizing the fused section caption (combine multiple selected sentences coherently)
- Generate ONE high-quality Q/A pair from the fused caption, rather than per-sentence Q/A
- Use a more robust NLP approach (seq2seq QA generation, or better regex patterns)
- Advantage: Aligns question with actual fused content
- Disadvantage: Reduces QA count (one per section instead of multiple)

## 4. Recommended Next Steps

### Immediate Actions (Priority Order)

1. **Fix answer truncation** (Quick Win)
   - Change `_trim_answer(max_words=6)` to extract complete noun phrases
   - Example fix:
     ```python
     def _trim_answer(text: str) -> str:
         # Instead of cutting at 6 words, cut at sentence boundaries
         for sep in [", followed by", ", and", ",", " and ", " while "]:
             if sep in text:
                 text = text.split(sep, 1)[0]
         return text.strip(" .,:;")
     ```
   - Expected impact: "To the right of the cutting" → "To the right of the cutting board"

2. **Preserve original modality Q/A** (Medium Effort, High Impact)
   - Store `original_qa` in `CaptionEvidence` dataclass
   - Reuse when answer extraction fails or when answer is too short
   - Fallback chain: extracted Q/A → original modality Q/A → generic fallback
   - Expected impact: Better answer quality, preserves expert-crafted questions

3. **Improve section filtering logic** (Medium Effort)
   - Don't allow layout sentences into scene_overview (verb gates? topic classification?)
   - Better heuristics for annotation_key matching
   - Expected impact: Fewer mismatched Q/A pairs like "What setting?" asked on layout data

4. **Remove hard-coded cross-modal templates and make sample-driven**
   - Generate cross_modal_details from actual evidence that supports claims across modalities
   - Currently contains templates like "RGB and IR cues agree on..." that may not reflect current sample

5. **Replace first-sentence-only final caption** with robust sample summary
   - Synthesize multiple selected evidence into coherent final caption
   - Generate single high-quality Q/A from synthesized caption

6. **Add manual inspection / QA quality score**
   - Script to flag low-quality Q/A (truncated answers, generic questions, mismatched section)
   - Human review loop for uncertain cases

### Strategic Approach

- The current fusion is **sentence-selection + per-sentence Q/A generation**
- Alternative: **sentence-selection + section synthesis + unified Q/A generation**
- Papers to read later: late fusion architectures, multimodal QA generation, answer extraction techniques

Related files:

- `annotation_feature/fusion.py` — main fusion pipeline
- `rgb_qa_results.json`, `ir_qa_results.json`, etc. — original modality Q/A pairs (discard current extraction logic)

## 5. Current Filtering Rules Explanation

- `scene_overview` and `audio_cues` these 2 sections require keyword matches so that each section keeps only sentences that really look like scene-level or audio-level evidence.
- `visible_objects_and_layout` also checks the annotation key for `spatial`, `count`, or `object` because that section is meant to capture objects, positions, and quantities, not generic descriptions.
- `motion_and_event_cues` checks for `dynamic`, `action`, `scene_sequence`, or `event_*` so that movement and temporal change stay separate from static layout.

Why these rules exist:

- They act as a coarse type filter before scoring, so each section stays focused and does not mix unrelated content.

What happens if we relax them:

- Coverage goes up and more QA content is kept.
- Noise also increases, because more weakly related sentences can enter the fusion.
- Sections become less cleanly separated, so manual review and deduplication become more important.
- In practice this is a tradeoff: relaxing the rules helps when the current fusion is too sparse, but too much relaxation can make the final caption redundant or messy.

## 6. Current Implementation Status

**Completed:**
- ✓ Relaxed filtering mode (fewer keyword gates, lower score threshold)
- ✓ Per-QA metadata (is_outlier, drop_reason, fusion_score, support_score, modality_reliability)
- ✓ Top-k QA per section (not single QA per section)
- ✓ Diagnostics output (dropped examples tracking)
- ✓ JSON/CSV export scripts

**Known Issues:**
- ✗ Answer truncation (6-word limit breaks semantic completeness)
- ✗ Template-driven Q/A extraction (question-answer mismatch)
- ✗ Lost original modality Q/A (discarded during extraction)
- ✗ Hard-coded cross-modal templates (not sample-driven)

## 7. Pragmatic Path Forward

**Reality check:** The current fusion works for coverage but trades Q/A quality for quantity. If the goal is **QA ground truth covering all modalities with acceptable quality**:

1. Keep current fusion architecture (it works for multi-modality alignment)
2. **Priority 1:** Fix answer truncation → (1-2 hours coding, high ROI)
3. **Priority 2:** Preserve original modality Q/A as fallback → (2-3 hours, high ROI)
4. **Priority 3:** Manual spot-check on 5-10 samples to identify systematic Q/A issues
5. **Priority 4:** If still noisy after fixes, implement per-section sample synthesis (longer timeline)

**Do NOT block on:**
- Finding the perfect late fusion algorithm
- Reading papers on multimodal QA before fixing basics
- Implementing seq2seq QA generation prematurely

Use papers to justify final design choices, not as prerequisite for iteration.
