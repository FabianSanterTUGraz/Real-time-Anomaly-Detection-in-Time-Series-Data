import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output", "output.txt")
DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "input.csv")
DATASET_ROOT = os.path.join(SCRIPT_DIR, "..",
                            "ieee-phm-2012-data-challenge-dataset",
                            "Full_Test_Set")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_accel(f):
    df = pd.read_csv(f, header=None)
    return df.iloc[:, 4 if df.shape[1] >= 6 else 0].to_numpy()


def plot_output(png_path, title, fingerprints, n_cols=10):
    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 3.0))
    axes = np.atleast_1d(axes).flatten()
    for i, (idx, seg) in enumerate(fingerprints):
        r = np.linalg.norm(seg - seg.mean(0), axis=1)
        rn = (r - r.min()) / (r.max() - r.min() + 1e-12)
        axes[i].scatter(seg[:, 0], seg[:, 1], s=2,
                        c=colormaps["turbo"](rn), alpha=0.7)
        axes[i].set_aspect('equal')
        axes[i].axis('off')
        axes[i].set_title(f"#{idx}", fontsize=8)
    for j in range(len(fingerprints), len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(png_path, dpi=100)
    plt.show()
    plt.close(fig)


for bearing_dir in sorted(Path(DATASET_ROOT).iterdir()):
    if not bearing_dir.is_dir():
        continue
    bearing_name = bearing_dir.name
    csv_files = sorted([f for f in bearing_dir.iterdir()
                        if f.name.startswith("acc_") and f.name.endswith(".csv")])
    if not csv_files:
        continue

    target_dir = os.path.join(RESULTS_DIR, bearing_name)
    os.makedirs(target_dir, exist_ok=True)

    for d in [300]:
        n_files = len(csv_files)
        n_views = min(10, n_files)
        indices = (np.array([0]) if n_views == 1 else
                   np.unique(np.linspace(0, n_files - 1, n_views).round().astype(int)))
        indices[0], indices[-1] = 0, n_files - 1

        fingerprints = []
        for file_idx in indices:
            sig = load_accel(csv_files[file_idx])
            pd.Series(sig).to_csv(DATA_PATH, index=False, header=False)
            subprocess.run(["./anomaly_detection.exe", str(d), "1", "2560", "combined_output"],
                           check=True, cwd=SCRIPT_DIR)
            df = pd.read_csv(OUTPUT_PATH, header=None, sep=',', names=['X', 'Y'])
            fingerprints.append((file_idx, df[['X', 'Y']].to_numpy()))
            print(f"[{bearing_name}] {csv_files[file_idx].name}")

        plot_output(os.path.join(target_dir, f"dynamic_{bearing_name}_d{d}.png"),
                    f"{bearing_name} | d = {d}", fingerprints)