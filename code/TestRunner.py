#Wichtig wenn d zu groß ist dann im c++ file ein buffer overflow

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.txt")

LASPI_DATA_PATH = r"C:\Users\39320\workspace\Real-time-Anomaly-Detection-in-Time-Series-Data\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox\Healthy_motor\25hz_0%_1490rpm\acc_00001.csv"
DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "input.csv")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "benchmarkCode", "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)


def plot_output(png_path, title):
    df = pd.read_csv(OUTPUT_PATH, header=None, sep=',', names=['X', 'Y'])
    projected = df[['X', 'Y']].to_numpy()

    scores = np.linalg.norm(projected, axis=1)
    scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(projected[:, 0], projected[:, 1], s=8, c=colormaps["turbo"](scores_norm), alpha=0.6)
    ax.axis("off")
    ax.set_title(title, fontsize=18)

    plt.tight_layout()
    plt.savefig(png_path)
    plt.show()
    plt.close(fig)


tau = 1

root_path = Path(LASPI_DATA_PATH).parents[2]

for csv_file in root_path.glob("**/*.csv"):
    if csv_file.name.startswith("._"):
        continue

    fault_type_name = csv_file.parent.parent.name
    condition_name = csv_file.parent.name
    fileName = csv_file.stem

    shutil.copyfile(csv_file, DATA_PATH)
    for d in [12,25,35]:
        for windowSize in [8000]:
            subprocess.run(["./anomaly_detection.exe", str(d), str(tau), str(windowSize), str("input")])
            png_path = os.path.join(RESULTS_DIR,
                                    f"dynamic_{fault_type_name}_{condition_name}_{fileName}_tau_{tau}_w_{windowSize}_d{d}.png")
            plot_title = f"{fault_type_name}\n{condition_name}\nw = {windowSize} | d = {d} | tau = {tau}"
            plot_output(png_path, plot_title)