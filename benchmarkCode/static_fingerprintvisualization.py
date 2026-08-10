# Required Python libraries:
 #-numpy
 #-matplotlib
 #-scikit-learn
 # Install via: pip3 install numpy matplotlib scikit-learn

import os

import numpy as np
from matplotlib.pyplot import colormaps
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "code", "Data", "20.csv")
BENCHMARK_RESULTS_PATH = os.path.join(SCRIPT_DIR, "data", "benchmarkResults.txt")
STATIC_RESULT_PLOT_PATH = os.path.join(SCRIPT_DIR, "data", "StaticResult.png")

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

def plot_embedding(ax, tde, title="TDE", benchmark_results_path=BENCHMARK_RESULTS_PATH):
    pca = PCA(n_components=2)
    projected = pca.fit_transform(tde)

    cov_matrix = np.cov(tde, rowvar=False)

    print("Covariance Matrix Shape:", cov_matrix.shape)
    print("Top-left 3x3 corner:\n", cov_matrix[:3, :3])

    np.savetxt(benchmark_results_path, projected, delimiter=",", fmt="%.6f")

    scores = []
    for point in projected:
        scores.append(np.linalg.norm(point))
    scores = np.array(scores)
    scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    print(projected)
    ax.scatter(projected[:, 0], projected[:, 1], s=8, c=colormaps["turbo"](scores_norm))
    ax.axis("off")
    ax.set_title(title, fontsize=30)


values = np.loadtxt(DATA_PATH, skiprows=1)

# Input: values.npy as a 1D numpy array, ideally a vibration
plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(nrows=1, ncols=1)
tde = time_delay_embedding(values, d=13, tau=15, stride=1)
plot_embedding(ax, tde)

plt.savefig(STATIC_RESULT_PLOT_PATH)

plt.show()
