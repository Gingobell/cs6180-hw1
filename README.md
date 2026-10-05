# CS6180 Homework 1

Q4 implementations: baseline, RMSNorm, SwiGLU, NoPE, RoPE, and GQA.

```sh
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/run_all.py --profile formal --device cuda:0 --dtype float32 --out results/formal/run1
```

Each run uses seed 1337 and 5,000 optimizer updates. Loss logs and comparison plots are saved under the output directory.

Q2 and Q3 figures:

```sh
python scripts/plot_rope.py
python scripts/draw_gec_architecture.py
```

Based on [nanoGPT](https://github.com/karpathy/nanoGPT), commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. See `LICENSE`.
