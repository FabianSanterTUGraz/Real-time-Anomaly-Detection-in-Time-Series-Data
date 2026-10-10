import os, re, time, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps

# ============================================================
# Parameters
# ============================================================
d = 2500
tau = 1
window_size = 20480
SNAPSHOT_LEN = 20480            # samples per original IMS file

TEST_SET = "2nd_test"
BEARING = 1                     # bearing number (1-4), NOT the column index
COMBINED_NAME = "0000.00.00.00.00.00"

N_PLOTS = 2500                    # windows plotted, evenly spaced over the whole run
N_LAST = 5                      # additionally the last N windows of the run
MAX_SAMPLES = 20480 * 10              # e.g. 20480 * 20 for a quick test run; None = whole file
RUN_CPP = True                  # False: only re-plot the stored output of the last run

SET_CHANNELS = {
    "1st_test": {1: 0, 2: 2, 3: 4, 4: 6},
    "2nd_test": {1: 0, 2: 1, 3: 2, 4: 3},
    "3rd_test": {1: 0, 2: 1, 3: 2, 4: 3},
}
FAILURE_NOTE = {
    "1st_test": {3: "inner race defect", 4: "roller element defect"},
    "2nd_test": {1: "outer race failure"},
    "3rd_test": {3: "outer race failure"},
}
FILENAME_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")

BASE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(BASE, "anomaly_detection.exe")
INPUT = os.path.join(BASE, "Data", "input.csv")
OUTPUT = os.path.join(BASE, "output", "output.txt")
IMS_ROOT = os.path.join(BASE, "..", "IMS")
RESULTS = os.path.join(BASE, "..", "results")
for p in (RESULTS, os.path.dirname(INPUT), os.path.dirname(OUTPUT)):
    os.makedirs(p, exist_ok=True)

TAG = f"{TEST_SET}_bearing{BEARING}_d{d}_tau{tau}_w{window_size}"
STORED_OUTPUT = os.path.join(os.path.dirname(OUTPUT), f"combined_{TAG}.txt")
OFFSET = (d - 1) * tau


def find_combined_file():
    """The combined file sits one level above the snapshot folder: IMS/<set>/0000.00.00.00.00.00"""
    for name in (COMBINED_NAME, COMBINED_NAME + ".txt"):
        path = os.path.join(IMS_ROOT, TEST_SET, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{COMBINED_NAME} nicht gefunden in {os.path.join(IMS_ROOT, TEST_SET)}")


def snapshot_names():
    """Timestamps of the original files, used as plot titles."""
    for folder in (os.path.join(IMS_ROOT, TEST_SET, TEST_SET), os.path.join(IMS_ROOT, TEST_SET)):
        if os.path.isdir(folder):
            names = sorted(f.name for f in Path(folder).iterdir() if FILENAME_RE.match(f.name))
            if names:
                return names
    return []


# ============================================================
# 1) one C++ run over the whole file
# ============================================================
def run_cpp():
    if not os.path.isfile(EXE):
        raise FileNotFoundError(f"C++-Programm nicht gefunden: {EXE} - erst bauen (Release, -O2)")
    combined = find_combined_file()
    col = SET_CHANNELS[TEST_SET][BEARING]

    print("Lese", combined, "...", flush=True)
    sig = pd.read_csv(combined, header=None, sep="\t", usecols=[col], dtype=np.float64)[col]
    if MAX_SAMPLES:
        sig = sig.iloc[:MAX_SAMPLES]
    n_samples = len(sig)
    sig.to_csv(INPUT, index=False, header=False)
    del sig
    print(f"{n_samples:,} Samples ({n_samples / SNAPSHOT_LEN:.0f} Snapshots) nach {INPUT} geschrieben", flush=True)

    print(f"Starte C++ (d={d}, tau={tau}, window={window_size}) ... das kann dauern", flush=True)
    t0 = time.perf_counter()
    subprocess.run([EXE, str(d), str(tau), str(window_size), "input"],
                   cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    runtime = time.perf_counter() - t0
    print(f"C++ fertig nach {runtime / 60:.1f} min "
          f"({n_samples / runtime:,.0f} Samples/s, Echtzeit = 20,000 Samples/s)", flush=True)
    shutil.copyfile(OUTPUT, STORED_OUTPUT)  # keep it, so RUN_CPP = False can re-plot without re-running


# ============================================================
# 2) split the output into one point cloud per window
# ============================================================
def has_separators(path, max_lines=4 * window_size):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.strip() == "---":
                return True
            if i > max_lines:
                break
    return False


def count_windows(path, window_mode):
    if window_mode:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip() == "---")
    with open(path, "rb") as fh:
        n_rows = sum(1 for _ in fh)
    return (n_rows + OFFSET) // SNAPSHOT_LEN


def read_windows(path, wanted, window_mode):
    """Returns {window index: (n, 2) array} for the wanted windows only (streams the file)."""
    wanted = set(wanted)
    segs = {i: [] for i in wanted}
    seg, row, bad = 0, 0, 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if window_mode:
                if line == "---":
                    seg += 1
                    continue
                target = seg
            else:
                k = row          # row index must count EVERY line, also broken ones
                row += 1
                target = k // SNAPSHOT_LEN if (k % SNAPSHOT_LEN) < SNAPSHOT_LEN - OFFSET else None
            if target not in wanted:
                continue
            try:
                x, y = line.split(",")
                x, y = float(x), float(y)
            except ValueError:
                bad += 1
                continue
            if np.isfinite(x) and np.isfinite(y):
                segs[target].append((x, y))
            else:
                bad += 1
    if bad:
        print(f"  ({bad} fehlerhafte/NaN-Zeilen übersprungen)")
    return {i: np.array(v) for i, v in segs.items() if v}


# ============================================================
# 3) plotting (same look as TestRunner.py)
# ============================================================
def plot_embedding(ax, fp, title):
    scores = np.linalg.norm(fp, axis=1)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    ax.scatter(fp[:, 0], fp[:, 1], s=8, c=colormaps["turbo"](scores_norm), alpha=0.6)
    ax.axis("off")
    limit = np.abs(fp).max() * 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=30)


def main():
    if RUN_CPP:
        run_cpp()
    elif not os.path.isfile(STORED_OUTPUT):
        raise FileNotFoundError(f"Kein gespeicherter Output {STORED_OUTPUT} - RUN_CPP = True setzen")

    window_mode = has_separators(STORED_OUTPUT)
    n_win = count_windows(STORED_OUTPUT, window_mode)
    print(f"Output-Format: {'WINDOW (neu projiziert pro Fenster)' if window_mode else 'ARRIVAL (Projektion bei Ankunft)'}, "
          f"{n_win} Fenster")
    if n_win == 0:
        print("Keine vollständigen Fenster im Output.")
        return

    names = snapshot_names()
    if len(names) < n_win:
        names = [f"window {i}" for i in range(n_win)]
    note = FAILURE_NOTE.get(TEST_SET, {}).get(BEARING, "no failure recorded")

    # same selection as TestRunner: N_PLOTS evenly spaced + last N_LAST
    indices = np.linspace(0, n_win - 1, min(N_PLOTS, n_win)).round().astype(int).tolist()
    indices += [i for i in range(max(0, n_win - N_LAST), n_win) if i not in indices]

    print("Lese Fenster", indices, "...", flush=True)
    windows = read_windows(STORED_OUTPUT, indices, window_mode)

    mode = "window" if window_mode else "arrival"
    for i in indices:
        if i not in windows:
            print("kein Fingerprint fuer Fenster", i)
            continue
        name = names[i]
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_embedding(ax, windows[i], title=name)
        plt.tight_layout()
        png = os.path.join(RESULTS, f"dynamic_{mode}_IMS_{TEST_SET}_bearing{BEARING}_{i:04d}_{name}.png")
        plt.savefig(png)
        plt.show()
        plt.close(fig)
        print(f"gespeichert: {os.path.basename(png)} ({note})")


if __name__ == "__main__":
    main()