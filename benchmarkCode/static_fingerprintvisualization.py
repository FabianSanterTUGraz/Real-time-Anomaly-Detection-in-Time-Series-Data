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
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "code", "Data", "11 - m1_mechanically_imbalanced_electrically_50_ohm_fault_half_speed.csv")
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

def plot_embedding(ax, tde, title="offline-TDE", benchmark_results_path=BENCHMARK_RESULTS_PATH):
    pca = PCA(n_components=2,svd_solver="full")
    projected = pca.fit_transform(tde)

    cov_matrix = np.cov(tde, rowvar=False)

    print("Covariance Matrix Shape:", cov_matrix.shape)
    print("Top-left 3x3 corner:\n", cov_matrix[:3, :3])

    # NEU: vollständige Eigenwerte berechnen und ausgeben
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues_sorted_desc = np.sort(eigenvalues)[::-1]
    print("\nAll eigenvalues (descending):", eigenvalues_sorted_desc)
    print("Ratio λ2/λ1:", eigenvalues_sorted_desc[1] / eigenvalues_sorted_desc[0])
    if len(eigenvalues_sorted_desc) > 2:
        print("Ratio λ3/λ2:", eigenvalues_sorted_desc[2] / eigenvalues_sorted_desc[1])

    np.savetxt(benchmark_results_path, projected, delimiter=",", fmt="%.6f")

    scores = []
    for point in projected:
        scores.append(np.linalg.norm(point))
    scores = np.array(scores)
    scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    print(projected)
    ax.scatter(projected[:, 0], projected[:, 1], s=8, c=colormaps["turbo"](scores_norm), alpha=0.3)
    ax.axis("off")
    ax.set_title(title, fontsize=30)


values = np.loadtxt(DATA_PATH, skiprows=1)
print(f"Loaded {len(values)} samples -> use this exact number as windowSize for the dynamic run")

# Input: values.npy as a 1D numpy array, ideally a vibration
fig, ax = plt.subplots(figsize=(7, 7))
tde = time_delay_embedding(values, d=15, tau=1, stride=1)
plot_embedding(ax, tde)

plt.tight_layout()
plt.savefig(STATIC_RESULT_PLOT_PATH)
plt.show()
