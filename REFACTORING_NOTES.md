# Pipeline Refactoring Summary

## Overview
The pipeline has been refactored with a clearer folder structure and consistent naming conventions. The three modalities (RGB, Event, Depth) are now organized in dedicated subdirectories under a `modalities/` folder, and shared utilities are in a `shared/` folder.

## New Structure

```
annotation_feature/
├── pipeline.py                          # Wrapper/entry point (imports from pipeline/)
├── pipeline/
│   ├── __init__.py
│   ├── main.py                          # Main orchestrator
│   ├── client.py                        # Gemini API client
│   ├── utils.py                         # Shared utilities
│   ├── modalities/                      # Modality-specific pipelines
│   │   ├── __init__.py
│   │   ├── rgb/
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py              # RGB QA pipeline
│   │   ├── event/
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py              # Event QA pipeline
│   │   └── depth/
│   │       ├── __init__.py
│   │       └── pipeline.py              # Depth QA pipeline
│   └── shared/                          # Shared utilities across modalities
│       ├── __init__.py
│       └── caption_generator.py         # Caption generation functions
```

## Changes Made

### File Reorganization
1. **qa_pipeline.py** → **modalities/rgb/pipeline.py**
   - Handles RGB video annotation
   - Uses RGB_PROMPTS for caption, question, and answer generation
   - Named `pipeline.py` for consistency within modality directories

2. **event_pipeline.py** → **modalities/event/pipeline.py**
   - Handles event-based annotation
   - Uses EVENT_PROMPTS
   - Named `pipeline.py` for consistency

3. **depth_pipeline.py** → **modalities/depth/pipeline.py**
   - Handles depth-based annotation
   - Uses DEPTH_PROMPTS
   - Named `pipeline.py` for consistency

4. **caption_pipeline.py** → **shared/caption_generator.py**
   - Moved to shared folder as it's used across modalities
   - More descriptive name: `caption_generator.py`
   - Contains: `get_caption_from_gemini()`, `get_question_from_gemini()`, `get_answer_from_gemini()`

### Import Updates
- **annotation_feature/pipeline.py**: Updated to import from `pipeline.main` module
- **annotation_feature/pipeline/main.py**: Updated imports to use new modality paths:
  ```python
  from .modalities.rgb import run_parallel_pipeline
  from .modalities.event import run_event_parallel_pipeline
  from .modalities.depth import run_depth_parallel_pipeline
  ```

### Code Organization
Each modality now has consistent structure:
- `build_*_mega_prompt()` - Creates modality-specific prompts
- `parse_json_response()` - Parses API responses (duplicated for clarity within each modality)
- `normalize_*_results()` - Ensures data consistency
- `call_gemini_with_retry()` - API retry logic
- `process_*_pair_batch()` - Processes a single video pair
- `run_*_parallel_pipeline()` - Main entry point for parallel processing

## Naming Conventions

### Module Names
- **modalities**: Directory for modality-specific code
- **rgb, event, depth**: Lowercase modality directories
- **pipeline.py**: Consistent name for modality entry points
- **shared**: Directory for cross-modality utilities
- **caption_generator.py**: Descriptive name for caption utilities

### Function Names
- `run_parallel_pipeline()` - RGB modality main function
- `run_event_parallel_pipeline()` - Event modality main function
- `run_depth_parallel_pipeline()` - Depth modality main function
- All follow the pattern `run_*_parallel_pipeline()` for consistency

### Prompt Building Functions
- `build_rgb_mega_prompt()` - RGB specific
- `build_event_mega_prompt()` - Event specific
- `build_depth_mega_prompt()` - Depth specific
- All follow the pattern `build_*_mega_prompt()`

## Benefits of This Refactoring

1. **Clear Separation of Concerns**: Each modality is in its own directory
2. **Easy to Extend**: Adding a new modality (e.g., `audio/`) is straightforward
3. **Better Discoverability**: Code organization reflects the conceptual structure
4. **Consistent Naming**: All modalities follow the same pattern
5. **Reduced Code Duplication**: Shared functions are in one place
6. **Improved Maintainability**: Related code is grouped together
7. **Clearer Dependencies**: Import statements clearly show modality relationships

## Usage

### Running the pipeline
The API remains the same:
```python
from annotation_feature.pipeline import run, run_event, run_depth

# Run RGB pipeline
run(test_mode=True)

# Run Event pipeline
run_event(test_mode=True)

# Run Depth pipeline
run_depth(test_mode=True)
```

### Importing specific modalities
```python
from annotation_feature.pipeline.modalities.rgb import run_parallel_pipeline
from annotation_feature.pipeline.shared import get_caption_from_gemini
```

## Files Not Moved
The following files remain at the pipeline level as they are shared across modalities:
- **client.py** - Gemini API client initialization
- **utils.py** - Utility functions (frame encoding, image parts building, pair key generation)
- **main.py** - Main orchestrator that calls all three modality pipelines

## Notes

- All imports have been verified and there are no circular dependencies
- Each modality has its own `__init__.py` for clean imports
- The wrapper file `annotation_feature/pipeline.py` ensures backward compatibility
- Documentation strings have been added/updated in all refactored files
