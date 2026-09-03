import glob
import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATHS = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
WINDOW_SIZE = 20000
FULL_DATASET_NAME = "output1500"
N_BINS = 50

datasets = []
for dataset_path in DATASET_PATHS:
    name = os.path.splitext(os.path.basename(dataset_path))[0]

    df = pd.read_csv(dataset_path, header=None, sep=',', names=['X', 'Y'])
    projected_full = df[['X', 'Y']].to_numpy()

    if name == FULL_DATASET_NAME:
        projected = projected_full
    else:
        projected = projected_full[-WINDOW_SIZE:]

    scores = np.linalg.norm(projected, axis=1)
    datasets.append((name, projected, scores))

xlim = (min(p[:, 0].min() for _, p, _ in datasets), max(p[:, 0].max() for _, p, _ in datasets))
ylim = (min(p[:, 1].min() for _, p, _ in datasets), max(p[:, 1].max() for _, p, _ in datasets))
radius_lim = (0.0, max(s.max() for _, _, s in datasets))
bin_edges = np.linspace(radius_lim[0], radius_lim[1], N_BINS + 1)

for name, projected, scores in datasets:
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 7))

    ax_scatter.scatter(projected[:, 0], projected[:, 1], s=8,
                        c=colormaps["turbo"](scores_norm), alpha=0.3)
    ax_scatter.set_xlim(xlim)
    ax_scatter.set_ylim(ylim)
    ax_scatter.set_aspect("equal")
    ax_scatter.axis("off")
    n_value = "".join(ch for ch in name if ch.isdigit())
    ax_scatter.set_title(f"n = {n_value}", fontsize=24)

    ax_hist.hist(scores, bins=bin_edges, color="steelblue")
    ax_hist.set_xlim(radius_lim)
    ax_hist.set_xlabel("radius")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("radius histogram", fontsize=16)

    plt.tight_layout()
    plt.savefig(png_path)
    plt.show()
    plt.close(fig)
