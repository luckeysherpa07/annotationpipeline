# Archived 4B Eight-Frame Results

These results were generated on a different machine and are retained for
provenance only. They are not the current formal results for this repository.

Recorded environment:

```text
Operating system: Ubuntu 22.04.5 LTS
Kernel: Linux 6.8.0-111-generic x86_64
Python: 3.10.12
Virtual environment: .venv-cu126
PyTorch: 2.11.0+cu126
CUDA runtime: 12.6
Transformers: 5.12.0
bitsandbytes: 0.49.2
NVIDIA driver: 535.309.01
GPU: NVIDIA GeForce RTX 4080, 16376 MiB
```

The archived directory contains completed InternVL3.5-4B and Molmo2-4B
outputs. The Molmo2 result is known to require further investigation and must
not replace the current-machine formal run.

Current formal results are written to:

```text
outputs/benchmarks/vlm_8frame_aligned_4b
```

using:

```text
scripts/run_vlm_4b_aligned_frame_benchmark_cu130.py
```
