import os
import subprocess
import glob

import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.txt")

BASE_DIR = r"C:\Users\39320\workspace\Real-time-Anomaly-Detection-in-Time-Series-Data\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_output(png_path, title):
    df = pd.read_csv(OUTPUT_PATH, header=None, sep=',', names=['X', 'Y'])
    projected = df[['X', 'Y']].to_numpy()

    scores = np.linalg.norm(projected, axis=1)
    scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(projected[:, 0], projected[:, 1], s=8, c=colormaps["turbo"](scores_norm), alpha=0.6)
    ax.axis("off")
    ax.set_title(title, fontsize=30)

    plt.tight_layout()
    plt.savefig(png_path)
    plt.show()
    plt.close(fig)


d = 13

for laspi_path in glob.glob(os.path.join(BASE_DIR, "**", "*.csv"), recursive=True):
    if "__MACOSX" in laspi_path:
        continue

    fileName = os.path.splitext(os.path.basename(laspi_path))[0]

    DATA_PATH = os.path.join(SCRIPT_DIR, "Data", os.path.basename(laspi_path))
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    df = pd.read_csv(laspi_path, header=None)
    values = df[0]
    values.to_csv(DATA_PATH, header=False, index=False)

    # Spiegelung der Ordnerstruktur im results-Verzeichnis[cite: 2]
    rel_dir = os.path.dirname(os.path.relpath(laspi_path, BASE_DIR))
    current_results_dir = os.path.join(RESULTS_DIR, rel_dir)
    os.makedirs(current_results_dir, exist_ok=True)

    for tau in [1]:
        for windowSize in [8000]:

            subprocess.run(["./anomaly_detection.exe", str(d), str(tau), str(windowSize), str(fileName)])
            png_path = os.path.join(current_results_dir, f"dynamic_{fileName}_tau_{tau}_w_{windowSize}_d{d}.png")
            plot_output(png_path, f"w = {windowSize} d = {d} tau = {tau}")