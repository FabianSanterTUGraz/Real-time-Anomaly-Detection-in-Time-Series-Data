import os
import subprocess

import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.txt")

LASPI_DATA_PATH = r"C:\Users\Abuscom\Desktop\Real-time-Anomaly-Detection-in-Time-Series-Data\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox\Bearing_inner_race_fault\35hz_0%_2091rpm\acc_00001.csv"
DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "acc_00001.csv")

FAULT_TYPE_NAME = os.path.basename(os.path.dirname(os.path.dirname(LASPI_DATA_PATH)))
CONDITION_NAME = os.path.basename(os.path.dirname(LASPI_DATA_PATH))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "benchmarkCode", "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

df = pd.read_csv(LASPI_DATA_PATH, header=None)
values = df[0]
values.to_csv(DATA_PATH, header=False, index=False)


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

DATASETS = {
    "Dataset01":"acc_00001",
}

d = 13

for tau in [25,35]:
    for windowSize in [8000]:
        for label, fileName in DATASETS.items():
            subprocess.run(["./anomaly_detection.exe", str(d), str(tau), str(windowSize), str(fileName)])
            png_path = os.path.join(RESULTS_DIR, f"dynamic_{FAULT_TYPE_NAME}_{CONDITION_NAME}_tau_{tau}_w_{windowSize}_d{d}.png")
            plot_output(png_path, f"w = {windowSize} d = {d} tau = {tau}")
