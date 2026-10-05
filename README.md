# CS6180 Homework 1

Repository: https://github.com/Gingobell/cs6180-hw1

Written answers: [report.md](report.md). Q1–Q3 and Q4 implementation details are
included. Formal Q4 training has started, but completed formal results and
comparisons have not yet been collected. `results/smoke/` contains only tiny
pipeline checks; those losses must not be interpreted as formal comparisons.

## Setup and reproduction

Use Python 3.10+ with `requirements-hpc.txt`; `requirements.txt` records the
local verification environment. Formal runs use PyTorch 2.5.1, CUDA 12.1,
a Tesla V100-SXM2-32GB, and float32. From the repository root:

```sh
python scripts/prepare_data.py
python scripts/verify_variants.py
python scripts/run_all.py --profile formal --device cuda:0 --dtype float32 --out results/formal/run1
```

Do not start a duplicate of an active experiment. Existing run directories are
never overwritten. Each variant uses 5,000 optimizer updates and seed 1337.
Model variants: baseline, RMSNorm, SwiGLU, NoPE, RoPE and GQA (6Q/3KV).
Training and evaluation sampling are controlled independently. The runner saves
configs, environment versions, data/code hashes, JSONL losses, summaries and
checkpoints; the comparison script generates CSV and loss plots.

Upstream nanoGPT commit: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`.
`model.py`, `train.py`, `configurator.py`, `config/baseline.py` and the original
data preparation script retain upstream code and its MIT license. Changes are
implemented in `model_variants.py` and the new runner/scripts.

## Source provenance

The originally uploaded package is preserved locally. Two files changed locally
after that upload; their uploaded versions are under `reproducibility/uploaded-run1/`:

- `run_experiment.py`: the current version normalizes bare `cuda` to `cuda:0`.
  The running experiment explicitly uses `cuda:0` with the uploaded version.
- `scripts/check_gpu.py`: the current version checks native BF16 capability.
  The original recommended BF16 via an emulation-inclusive query; the actual
  formal command explicitly uses float32 instead.

Remote code hashes must still be checked against the collected run configs.
The model implementation itself is unchanged. The archived files are provenance
copies, not standalone entry points; restore them at their original relative
paths in a separate copy to reproduce the uploaded source exactly.

## Figures and checks

- `python scripts/verify_q1.py`: convolution gradient verification.
- `python scripts/plot_rope.py`: Q2 curve, all 65,537 points and verification.
- `python scripts/draw_gec_architecture.py`: Q3 SVG/PNG/PDF architecture diagram.
- `python scripts/plot_experiments.py RUN_DIRS... --out OUTPUT_DIR`: recorded losses.

The final written PDF and completed formal results will be added after verification.
