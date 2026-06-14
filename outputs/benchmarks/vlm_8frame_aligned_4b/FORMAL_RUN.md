# Current Formal 4B Eight-Frame Run

This directory is the canonical output location for the current-machine 4B
eight-frame benchmark.

Environment:

```text
Runner: scripts/run_vlm_4b_aligned_frame_benchmark_cu130.py
PyTorch: 2.12.0+cu130
CUDA runtime: 13.0
Transformers: 4.57.1
GPU: NVIDIA GeForce RTX 4090
Frame manifest SHA-256:
ce1b15ad21ec8e429b71a7b71e5e2ab4d08453ec74ce76409efde3d9082ce8b3
```

The initial files contain four verified preflight items per model. Formal
execution resumes these files in place and expands each model result to all
5,465 QA items. A result is complete only when its metadata reports:

```text
attempted_items: 5465
status_counts: {"answered": 5465}
```

The runner checkpoints the complete JSON and CSV every 25 new answers and
forces a final checkpoint on completion, exceptions, or `Ctrl+C`. The
checkpoint interval is recorded in result metadata.

Results from the other CUDA 12.6 / RTX 4080 machine are archived separately:

```text
outputs/benchmarks/vlm_8frame_aligned_4b_other_machine_cu126
```
