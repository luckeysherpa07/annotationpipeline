# Two-pass cross-modal caption experiment (archived)

## Status

This experiment was archived on 2026-07-13. The implementation remains on the
`experiment/two-pass-caption` branch and is not intended to be merged into
`main` in its current form.

## Why it was paused

The current model's ability to interpret video frames from the `event` modality
is not yet sufficiently understood. In the RGB/event comparison used during
development, it is therefore unclear whether an incorrect or unstable result
comes from the two-pass caption design, the prompt and validation rules, or a
more fundamental limitation in the model's event-frame perception.

Continuing prompt-level optimization before measuring that capability would
make the results difficult to interpret, so development is paused rather than
treating the current behavior as a pipeline failure.

## Archived implementation

The branch includes:

- the two-pass cross-modal caption pipeline;
- pass-1 prompt and semantic validation iterations;
- RGB/event frame sampling and evidence handling;
- representative successful and failed Gemini outputs.

The archive head before this note is commit `b0e764a` (`prompt_v1_2`).

## Representative JSON results

The latest successful pass-1 result is:

- `outputs/test_pass1_bike_night_18_19.json`
- model: `gemini-3.5-flash`
- input pair: night RGB and event frames from `ride_a_bike_split/Seg1`
- result: 1 completed item, 0 skipped items, 2 Gemini calls
- introduced in commit `e7e671a` (`prompt_opt_v1`)

The chronologically later result is:

- `outputs/test_pass1_bike_night_19_15.json`
- result: 0 completed items, 1 skipped item, 3 Gemini calls
- failure: the shared motion attribute `parked` did not have entity-bound atom
  support in both analyses
- introduced in commit `b0e764a` (`prompt_v1_2`)

Both JSON files are intentionally retained: the first is the latest successful
example, while the second documents the latest validation behavior. Other
untracked `test_*`, `temp_*`, recovery, reject, and diff files are local working
artifacts and are not part of this archive.

## Conditions for resuming

Resume this work after a focused event-modality capability check can answer:

1. whether the candidate model reliably recognizes objects, states, and motion
   in event-frame representations;
2. whether RGB/event disagreement reflects complementary evidence or model
   perception failure;
3. which event-frame visualization, sampling strategy, and model provide a
   stable baseline;
4. whether the semantic validator is rejecting genuinely unsupported claims or
   otherwise usable captions.

Once those questions are answered, continue from this branch or its archive
tag and evaluate pipeline changes against the retained JSON baselines.
