import os

import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
OUTPUT_PATH = r"C:\Users\39320\Desktop\Real-time-Anomaly-Detection-in-Time-Series-Data\code\output\output.txt"
DELTA_PATH = r"C:\Users\39320\Desktop\Real-time-Anomaly-Detection-in-Time-Series-Data\benchmarkCode\data\deltaOutput.txt"
BENCHMARK_RESULTS_PATH = os.path.join(REPO_ROOT, "benchmarkCode", "data", "benchmarkResults.txt")
DYNAMIC_RESULT_PLOT_PATH = os.path.join(REPO_ROOT, "benchmarkCode", "data", "DynamicResult.png")

df = pd.read_csv(OUTPUT_PATH, nrows=111563, header=None, sep=',', names=['X', 'Y'])

print("First few rows of extracted coordinates:")
print(df.head())

projected = df[['X', 'Y']].to_numpy()

#sign correction to visually get the same result as the static version which uses sklearn sign correction
benchmark = np.loadtxt(BENCHMARK_RESULTS_PATH, delimiter=",")
rows = min(len(projected), len(benchmark))
sign = np.sign(np.sum(projected[:rows] * benchmark[:rows], axis=0))
sign[sign == 0] = 1
projected = projected * sign
print(f"Sign correction applied per column (x, y): {sign}")

scores = np.linalg.norm(projected, axis=1)
scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# --- 2D VISUAL FINGERPRINT PLOT ---
plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(projected[:, 0], projected[:, 1], s=8, c=colormaps["turbo"](scores_norm))
ax.axis("off")
ax.set_title("TDE", fontsize=30)

# Save the fingerprint layout graph
plt.savefig(DYNAMIC_RESULT_PLOT_PATH)
plt.show()