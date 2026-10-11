"""
ims_each_file.py  (lives next to TestRunner.py in code/)

For ONE IMS test set (folder) and the selected bearing(s), snapshot files are fed one by one
into the C++ program and every fingerprint is plotted as its own figure, in chronological order.
Which files: every FILE_STEP-th file (0, FILE_STEP, 2*FILE_STEP, ...) plus always the last N_LAST.

IMS layout: IMS/<set>/<set>/<timestamp>  (or IMS/<set>/<timestamp>), each file
20480 rows, tab-separated, no header, one column per channel.
"""
import os, re, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps

# ============================================================
# Parameters
# ============================================================
d = 1200                 # ~ one shaft revolution (2000 RPM at 20 kHz)
tau = 1
window_size = 20480     # = one snapshot file

SET = "2nd_test"        # one folder: "1st_test", "2nd_test" or "3rd_test"
BEARINGS = [1]          # bearing NUMBERS 1-4 (not column indices!): [1], [3, 4], 4 or None = all

FILE_STEP = 100         # plot every FILE_STEP-th file (1 = every file)
N_LAST = 5              # additionally always the last N_LAST files (0 = none)

SHOW_PLOTS = True       # show every figure (with many files: False and look at the saved PNGs)
SAVE_PLOTS = True       # save to results/IMS_<set>_bearing<n>/

SET_CHANNELS = {        # bearing -> 0-based column (IMS readme); set 1: x-channel of each bearing
    "1st_test": {1: 0, 2: 2, 3: 4, 4: 6},
    "2nd_test": {1: 0, 2: 1, 3: 2, 4: 3},
    "3rd_test": {1: 0, 2: 1, 3: 2, 4: 3},
}
FAILURE_NOTE = {
    "1st_test": {3: "inner race defect", 4: "roller element defect"},
    "2nd_test": {1: "outer race failure"},
    "3rd_test": {3: "outer race failure"},
}

BASE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(BASE, "anomaly_detection.exe")
INPUT = os.path.join(BASE, "Data", "input.csv")
OUTPUT = os.path.join(BASE, "output", "output.txt")
IMS_ROOT = os.path.join(BASE, "..", "IMS")
RESULTS = os.path.join(BASE, "..", "results")
for p in (RESULTS, os.path.dirname(INPUT), os.path.dirname(OUTPUT)):
    os.makedirs(p, exist_ok=True)

FILENAME_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


# ============================================================
# Configuration checks
# ============================================================
def selected_bearings(set_name):
    """BEARINGS as a list; checks that every bearing exists for this set (clear error instead of KeyError)."""
    if BEARINGS is None:
        bearings = sorted(SET_CHANNELS[set_name])
    elif isinstance(BEARINGS, (list, tuple)):
        bearings = list(BEARINGS)
    else:
        bearings = [BEARINGS]
    valid = SET_CHANNELS[set_name]
    for b in bearings:
        if b not in valid:
            mapping = ", ".join(f"bearing {k} -> column {v}" for k, v in valid.items())
            raise ValueError(f"BEARINGS contains {b!r}, but {set_name} only has bearings 1-4 ({mapping}). "
                             f"Use bearing numbers, not column indices.")
    return bearings


def check_config():
    if SET not in SET_CHANNELS:
        raise ValueError(f"unknown set {SET!r}, choose from {list(SET_CHANNELS)}")
    selected_bearings(SET)
    if FILE_STEP < 1 or N_LAST < 0:
        raise ValueError("FILE_STEP must be >= 1 and N_LAST >= 0")


# ============================================================
# Files and C++ run
# ============================================================
def resolve_test_dir(set_name):
    """Extracted .rar archives nest a duplicate folder (e.g. 2nd_test/2nd_test)."""
    for folder in (os.path.join(IMS_ROOT, set_name, set_name), os.path.join(IMS_ROOT, set_name)):
        if os.path.isdir(folder) and list_snapshot_files(folder):
            return folder
    return None


def list_snapshot_files(folder):
    """Timestamp-named snapshot files, chronological; the combined file 0000.00.00.00.00.00 is skipped."""
    return sorted(f for f in Path(folder).iterdir()
                  if f.is_file() and FILENAME_RE.match(f.name) and not f.name.startswith("0000"))


def read_output():
    """Reads x,y lines of the FIRST block (everything before the first '---').
    Works for both driver variants (one line per sample, or window blocks ending with '---').
    Broken lines (e.g. -nan(ind) from MSVC) are skipped."""
    rows, bad = [], 0
    with open(OUTPUT, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line == "---":
                break
            if not line:
                continue
            try:
                x, y = (float(v) for v in line.split(","))
            except ValueError:
                bad += 1
                continue
            if np.isfinite(x) and np.isfinite(y):
                rows.append((x, y))
            else:
                bad += 1
    if bad:
        print(f"      ({bad} fehlerhafte/NaN-Zeilen übersprungen)")
    return np.array(rows) if rows else None


def fingerprint(snapshot_file, col):
    """One IMS snapshot file -> one C++ run -> fingerprint (L, 2) or None."""
    df = pd.read_csv(snapshot_file, header=None, sep="\t", usecols=[col])
    df[col].to_csv(INPUT, index=False, header=False)
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)                      # never plot a stale output of a previous run
    subprocess.run([EXE, str(d), str(tau), str(window_size), "input"],
                   cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    if not os.path.exists(OUTPUT):
        print("      C++ hat keine output.txt geschrieben")
        return None
    return read_output()


# ============================================================
# Plotting (one figure per file)
# ============================================================
def plot_fingerprint(fp, title, png_path):
    r = np.linalg.norm(fp - fp.mean(0), axis=1)
    rn = (r - r.min()) / (r.max() - r.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(fp[:, 0], fp[:, 1], s=8, c=colormaps["turbo"](rn), alpha=0.6)
    limit = np.abs(fp).max() * 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    if png_path:
        plt.savefig(png_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def main():
    check_config()
    if not os.path.isfile(EXE):
        raise FileNotFoundError(f"C++-Programm nicht gefunden: {EXE} - erst bauen (mingw32-make)")

    test_dir = resolve_test_dir(SET)
    if test_dir is None:
        raise FileNotFoundError(f"{SET}: keine Snapshot-Dateien unter {IMS_ROOT} gefunden")
    files = list_snapshot_files(test_dir)
    indices = sorted(set(range(0, len(files), FILE_STEP)) |
                     set(range(max(0, len(files) - N_LAST), len(files))))
    print(f"{SET}: {len(files)} Dateien, davon {len(indices)} ausgewählt "
          f"(jede {FILE_STEP}. + die letzten {N_LAST}): {indices}")

    for bearing in selected_bearings(SET):
        col = SET_CHANNELS[SET][bearing]
        note = FAILURE_NOTE.get(SET, {}).get(bearing, "no failure recorded")
        out_dir = os.path.join(RESULTS, f"IMS_{SET}_bearing{bearing}")
        if SAVE_PLOTS:
            os.makedirs(out_dir, exist_ok=True)
        print(f"  bearing {bearing} (column {col}, {note}):", flush=True)

        t_start = time.perf_counter()
        for count, i in enumerate(indices, 1):
            name = files[i].name
            t0 = time.perf_counter()
            fp = fingerprint(files[i], col)
            if fp is None:
                print(f"    [{count}/{len(indices)}] #{i} {name}: kein Fingerprint")
                continue
            png = os.path.join(out_dir, f"{i:04d}_{name}.png") if SAVE_PLOTS else None
            plot_fingerprint(fp, f"#{i}  {name}", png)
            eta = (time.perf_counter() - t_start) / count * (len(indices) - count)
            print(f"    [{count}/{len(indices)}] #{i} {name}: {time.perf_counter() - t0:.1f} s "
                  f"(noch ca. {eta / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()