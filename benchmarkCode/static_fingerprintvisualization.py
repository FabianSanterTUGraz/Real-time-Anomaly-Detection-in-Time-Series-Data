# Required Python libraries:
 #-numpy
 #-matplotlib
 #-scikit-learn
 # Install via: pip3 install numpy matplotlib scikit-learn

import os

import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LASPI_DATA_PATH = r"C:\Users\Abuscom\Desktop\Real-time-Anomaly-Detection-in-Time-Series-Data\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox\Healthy_motor\45hz_0%_2691rpm\acc_00001.csv"
#LASPI_DATA_PATH = r"C:\Users\Abuscom\Desktop\Real-time-Anomaly-Detection-in-Time-Series-Data\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox\Gear_half_broken_tooth\35hz_50%_2084rpm\acc_00001.csv"
FAULT_TYPE_NAME = os.path.basename(os.path.dirname(os.path.dirname(LASPI_DATA_PATH)))
CONDITION_NAME = os.path.basename(os.path.dirname(LASPI_DATA_PATH))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
BENCHMARK_RESULTS_PATH = os.path.join(RESULTS_DIR, "benchmarkResults.txt")
STATIC_RESULT_PLOT_PATH = os.path.join(RESULTS_DIR, f"static_{FAULT_TYPE_NAME}_{CONDITION_NAME}.png")
AXIS_LIMIT = 4.0

os.makedirs(RESULTS_DIR, exist_ok=True)

def time_delay_embedding(values, d, tau=1, stride=1):
    values = np.asarray(values)
    windows = []
    index = 0
    while index <= len(values)- d:
        window = []
        for i in range(index, index + d * tau, tau):  #the d is the size of the window if tau bigger 1 than window bigger and more values
            if i >= len(values):
                return np.array(windows)
            window.append(values[i])
        windows.append(window)
        index += stride
    return np.array(windows)

def plot_embedding(ax, tde, title="offline-TDE", benchmark_results_path=BENCHMARK_RESULTS_PATH):
    pca = PCA(n_components=2,svd_solver="full")
    projected = pca.fit_transform(tde)

    # sklearn hat keinen Schalter fuer die Vorzeichen-Konvention (svd_flip ist fest verdrahtet).
    # Die C++/online-Variante (subspaceIteration) verankert ihr Vorzeichen an den festen
    # Startvektoren e0=[1,0,...] fuer PC1 und e1=[0,1,...] fuer PC2. Gleiche Konvention hier
    # erzwingen, damit offline- und online-Plot dasselbe Vorzeichen zeigen.
    for i in range(2):
        if pca.components_[i][i] < 0:
            projected[:, i] *= -1

    scores = []
    for point in projected:
        scores.append(np.linalg.norm(point))
    scores = np.array(scores)
    scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    ax.scatter(projected[:, 0], projected[:, 1], s=8, c=colormaps["turbo"](scores_norm), alpha=0.6)
    ax.axis("off")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=30)


df = pd.read_csv(LASPI_DATA_PATH, header=None)
values = df[0].to_numpy()

# Input: values.npy as a 1D numpy array, ideally a vibration
fig, ax = plt.subplots(figsize=(7, 7))
tde = time_delay_embedding(values, d=25, tau=13, stride=1)
plot_embedding(ax, tde)

plt.tight_layout()
plt.savefig(STATIC_RESULT_PLOT_PATH)
plt.show()
