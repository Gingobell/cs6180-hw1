"""Reproduce HW1 Q2 using the lecture's scaled, pre-softmax score."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = 128
BASE = 10000.0


def score(delta):
    omega = BASE ** (-2 * np.arange(D // 2, dtype=np.float64) / D)
    return np.cos(np.asarray(delta)[..., None] * omega).mean(axis=-1) / np.sqrt(D)


def rotated_vector(position):
    angles = position * BASE ** (-2 * np.arange(D // 2) / D)
    # Each original pair is (1, 1) / sqrt(D).
    return np.stack((np.cos(angles) - np.sin(angles),
                     np.sin(angles) + np.cos(angles)), axis=-1).ravel() / np.sqrt(D)


def main():
    distances = np.arange(65537)
    values = score(distances)
    pairs = [(0, 0), (0, 1), (17, 29), (1234, 1234), (8192, 32768), (65536, 0)]
    errors = [abs(rotated_vector(i) @ rotated_vector(j) / np.sqrt(D) - score(i - j))
              for i, j in pairs]
    assert max(errors) < 1e-12
    assert np.isclose(values[0], 1 / np.sqrt(D), atol=1e-15, rtol=0)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values)) <= 1 / np.sqrt(D) + 1e-15
    assert np.allclose(score(-distances), values, atol=1e-15, rtol=0)

    figures = ROOT / "results" / "figures"
    metrics = ROOT / "results" / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    np.savetxt(metrics / "rope_attention.csv", np.column_stack((distances, values)),
               delimiter=",", header="distance,scaled_attention_score", comments="",
               fmt=["%d", "%.17g"])

    with plt.rc_context({"font.size": 11, "path.simplify": False}):
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        ax.plot(distances, values, color="#2463A6", linewidth=0.45)
        ax.axhline(0, color="#555555", linewidth=0.65)
        ax.set(xlabel=r"Relative distance $|i-j|$",
               ylabel=r"Scaled attention score $A_{i,j}$",
               title=r"RoPE positional score ($d=128$, base $10{,}000$)",
               xlim=(0, 65536))
        ticks = np.arange(0, 65537, 8192)
        ax.set_xticks(ticks, [f"{x:,}" for x in ticks])
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(figures / "rope_attention.png", dpi=200)
        fig.savefig(figures / "rope_attention.pdf")
        plt.close(fig)

    record = {"dimension": D, "frequency_base": BASE,
              "score_convention": "dot product divided by sqrt(d), before softmax",
              "distance_min": int(distances[0]), "distance_max": int(distances[-1]),
              "sample_count": len(distances), "score_at_zero": float(values[0]),
              "direct_rotation_position_pairs": pairs,
              "max_direct_rotation_error": float(max(errors)),
              "even_symmetry_finite_values_and_bound_checks": "passed"}
    (ROOT / "results" / "q2_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
