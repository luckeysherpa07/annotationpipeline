# Pass 1 Evaluation

`pass1_dev_manifest.json` defines the fixed development set and generation configuration for the `pass1_prompt_v1.1` baseline.

Run the evaluation from the repository root:

```bash
.venv/bin/python scripts/run_pass1_evaluation.py
```

Each invocation creates the configured number of new, non-overwriting `run_NNN.json` files under `outputs/pass1_evaluation/pass1_dev_v1/`. The runner updates `summary.json` and `summary.csv` with all runs currently present in that evaluation directory.

Use `pass1_manual_scores.csv` for human semantic review. Every score column uses:

- `0`: poor
- `1`: partial
- `2`: acceptable

Leave semantic score cells empty until a reviewer has inspected the corresponding raw run. `reviewer_notes` may contain free-form text.
