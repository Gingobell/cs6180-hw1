# CS6180 Homework 1

This repository contains the Q4 code and results, plus the scripts used to draw the Q2 and Q3 figures.

## Run the experiments

```sh
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/run_all.py --profile formal --device cuda:0 --dtype float32 --out results/formal/run2
```

The script runs the baseline, RMSNorm, SwiGLU, NoPE, RoPE, and GQA in that order. Each model trains for 5,000 updates with seed 1337.

The submitted logs are in `results/formal/run1/`. The `comparison/` folder contains the loss curves and result table. Use a new output folder when rerunning the experiments.

## Draw the figures

```sh
python scripts/plot_rope.py
python scripts/draw_gec_architecture.py
```

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. See `LICENSE`.
