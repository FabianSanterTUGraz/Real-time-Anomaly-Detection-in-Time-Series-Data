import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIGURATION: Set your file path here
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, "..", "code", "Data", "04 - m1_mechanically_imbalanced_half_speed.csv")
STATIC_RESULT_PLOT_PATH = os.path.join(SCRIPT_DIR, "data", "TimeSeries.png")


def plot_single_column_time_series(file_path, skiprows=1, max_rows=25000):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    # 1. READ SINGLE-COLUMN DATA
    # skiprows=1 skips the header line; max_rows sets how many points to plot
    data = np.loadtxt(file_path, skiprows=skiprows, max_rows=max_rows)
    data = np.asarray(data).flatten()  # Ensure 1D array shape

    print(f"Loaded {len(data)} samples from '{os.path.basename(file_path)}'")

    # 2. PLOT TIME SERIES
    fig, ax = plt.subplots(figsize=(12, 5))

    # X-axis will automatically be the sample index (0, 1, 2, ..., N-1)
    ax.plot(data, color="#1f77b4", linewidth=1.0)

    ax.set_xlabel("Sample Index", fontsize=11, labelpad=8)
    ax.set_ylabel("Amplitude", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig(STATIC_RESULT_PLOT_PATH)

    plt.show()


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    plot_single_column_time_series(FILE_PATH, skiprows=1, max_rows=25000)