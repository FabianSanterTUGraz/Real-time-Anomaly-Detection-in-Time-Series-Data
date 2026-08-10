# Required Python libraries:
 #-numpy
 # Install via: pip3 install numpy

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "code", "output", "output.txt")
BENCHMARK_RESULTS_PATH = os.path.join(SCRIPT_DIR, "data", "benchmarkResults.txt")
DELTA_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "deltaOutput.txt")


def compute_delta(output_path=OUTPUT_PATH, benchmark_path=BENCHMARK_RESULTS_PATH, delta_path=DELTA_OUTPUT_PATH):
    output = np.loadtxt(output_path, delimiter=",")
    benchmark = np.loadtxt(benchmark_path, delimiter=",")

    rows = min(len(output), len(benchmark))
    if len(output) != len(benchmark):
        print(f"Warning: output.txt has {len(output)} rows, benchmarkResults.txt has {len(benchmark)} rows. "
              f"Comparing only the first {rows} rows.")

    output = output[:rows].copy()
    benchmark = benchmark[:rows]

    # PCA principal components are only defined up to sign: the streaming
    # subspace iteration and sklearn's batch SVD can converge to the same
    # axis but pointing in opposite directions. Align each column's sign to
    # the benchmark (via correlation) before diffing, otherwise a converged
    # axis shows up as a huge delta of ~2x instead of ~0.
    sign = np.sign(np.sum(output * benchmark, axis=0))
    sign[sign == 0] = 1
    output *= sign
    print(f"Sign correction applied per column (x, y): {sign}")

    delta = output - benchmark

    np.savetxt(delta_path, delta, delimiter=",", fmt="%.6f")

    abs_delta = np.abs(delta)
    bench_scale = np.abs(benchmark).mean(axis=0)
    print(f"Compared {rows} rows.")
    print(f"Mean absolute delta (x, y): {abs_delta.mean(axis=0)}")
    print(f"Max absolute delta (x, y):  {abs_delta.max(axis=0)}")
    print(f"Mean relative delta (x, y): {abs_delta.mean(axis=0) / bench_scale}")
    print(f"Delta written to: {delta_path}")

    return delta


if __name__ == "__main__":
    compute_delta()
