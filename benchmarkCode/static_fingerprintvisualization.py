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
IMS_DIR = os.path.join(SCRIPT_DIR, "..", "IMS", "2nd_test", "2nd_test")
IMS_COLUMN = 0  # Bearing 3 (inner race defect), Ch 5 = 0-indexed column 4
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
BENCHMARK_RESULTS_PATH = os.path.join(RESULTS_DIR, "benchmarkResults.txt")
N_PLOTS = 10  # files plotted, evenly spaced over the whole run
N_LAST = 5  # additionally the last N files of the run

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
    limit = np.abs(projected).max() * 1.05  # per-plot axis: IMS amplitudes change a lot over the run
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=30)


names = sorted(os.listdir(IMS_DIR))

# N_PLOTS files evenly spaced over the run (first and last included) plus the last N_LAST files, each plotted individually
indices = np.linspace(0, len(names) - 1, N_PLOTS).round().astype(int).tolist()
indices += [i for i in range(len(names) - N_LAST, len(names)) if i not in indices]
for i in indices:
    name = names[i]
    df = pd.read_csv(os.path.join(IMS_DIR, name), header=None, sep="	")
    values = df[IMS_COLUMN].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    tde = time_delay_embedding(values, d=2500, tau=1, stride=1)
    plot_embedding(ax, tde, title=name)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"static_IMS_1st_test_{name}.png"))
    plt.show()
