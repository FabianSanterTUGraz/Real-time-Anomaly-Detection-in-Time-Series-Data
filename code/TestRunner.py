"""
ims_start_end_blocks.py  (lives next to TestRunner.py in code/)

For every selected IMS test set and bearing: the first N_PER_BLOCK and the last
N_PER_BLOCK snapshot files are fed one by one into the C++ program, and the
fingerprints are plotted side by side ("start  ~>  end").

IMS layout: IMS/<set>/<set>/<timestamp>  (or IMS/<set>/<timestamp>), each file
20480 rows, tab-separated, no header, one column per channel.
"""
import os, re, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.pyplot import colormaps

# ============================================================
# Parameters
# ============================================================
d = 600                 # ~ one shaft revolution (2000 RPM at 20 kHz)
tau = 1
window_size = 20480     # = one snapshot file
N_PER_BLOCK = 5         # files at the start / at the end of the run

SETS = ["1st_test", "2nd_test", "3rd_test"]
BEARINGS = None         # None = all 4 bearings, or e.g. [3, 4]

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
    Broken lines (e.g. -nan(ind) from MSVC) are skipped instead of crashing np.loadtxt."""
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
        print(f"    ({bad} fehlerhafte/NaN-Zeilen übersprungen)")
    return np.array(rows) if rows else None


def fingerprint(snapshot_file, col):
    """One IMS snapshot file -> one C++ run -> fingerprint (L, 2) or None."""
    df = pd.read_csv(snapshot_file, header=None, sep="\t", usecols=[col])
    df[col].to_csv(INPUT, index=False, header=False)
    subprocess.run([EXE, str(d), str(tau), str(window_size), "input"],
                   cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    return read_output()


def short_time(name):
    """'2004.02.12.10.32.39' -> '02-12 10:32'"""
    p = name.split(".")
    return f"{p[1]}-{p[2]} {p[3]}:{p[4]}"


def plot_blocks(png_path, title, block1, block2, block1_title, block2_title):
    n1, n2 = len(block1), len(block2)
    fig = plt.figure(figsize=(2.5 * (n1 + n2 + 1), 3.5))
    gs = gridspec.GridSpec(2, n1 + 1 + n2, height_ratios=[1, 12], hspace=0.05, wspace=0.05)

    ax = fig.add_subplot(gs[0, :n1]); ax.axis('off')
    ax.set_title(block1_title, fontsize=11, color='#2a7a2a')
    ax = fig.add_subplot(gs[0, n1 + 1:]); ax.axis('off')
    ax.set_title(block2_title, fontsize=11, color='#a02020')

    def draw(ax, label, seg):
        r = np.linalg.norm(seg - seg.mean(0), axis=1)
        rn = (r - r.min()) / (r.max() - r.min() + 1e-12)
        ax.scatter(seg[:, 0], seg[:, 1], s=2, c=colormaps["turbo"](rn), alpha=0.7)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(label, fontsize=7)

    for i, (label, seg) in enumerate(block1):
        draw(fig.add_subplot(gs[1, i]), label, seg)
    for i, (label, seg) in enumerate(block2):
        draw(fig.add_subplot(gs[1, n1 + 1 + i]), label, seg)

    ax = fig.add_subplot(gs[1, n1]); ax.axis('off')
    ax.text(0.5, 0.5, "⟿", ha='center', va='center', fontsize=48,
            color='#444444', transform=ax.transAxes)

    fig.suptitle(title, fontsize=13, y=0.98)
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    if not os.path.isfile(EXE):
        raise FileNotFoundError(f"C++-Programm nicht gefunden: {EXE} - erst bauen (Release, -O2)")

    for set_name in SETS:
        test_dir = resolve_test_dir(set_name)
        if test_dir is None:
            print(f"{set_name}: keine Snapshot-Dateien unter {IMS_ROOT} gefunden - übersprungen")
            continue
        files = list_snapshot_files(test_dir)
        if len(files) < 2 * N_PER_BLOCK:
            print(f"{set_name}: nur {len(files)} Dateien - übersprungen")
            continue
        first = list(range(N_PER_BLOCK))
        last = list(range(len(files) - N_PER_BLOCK, len(files)))
        print(f"{set_name}: {len(files)} Dateien ({files[0].name} ... {files[-1].name})")

        for bearing in (BEARINGS or sorted(SET_CHANNELS[set_name])):
            col = SET_CHANNELS[set_name][bearing]
            note = FAILURE_NOTE.get(set_name, {}).get(bearing, "no failure recorded")
            t0 = time.perf_counter()

            blocks = []
            for idxs in (first, last):
                b = []
                for i in idxs:
                    fp = fingerprint(files[i], col)
                    if fp is None:
                        print("  kein Fingerprint:", set_name, f"bearing {bearing}", files[i].name)
                    else:
                        b.append((f"#{i}  {short_time(files[i].name)}", fp))
                blocks.append(b)
            if not blocks[0] or not blocks[1]:
                continue

            png = os.path.join(RESULTS, f"IMS_{set_name}_bearing{bearing}_start_end_d{d}.png")
            plot_blocks(png,
                        f"IMS {set_name} | bearing {bearing} ({note}) | d = {d} | tau = {tau}",
                        blocks[0], blocks[1],
                        f"Anfang (Dateien #{first[0]} – #{first[-1]})",
                        f"Ende (Dateien #{last[0]} – #{last[-1]})")
            print(f"  bearing {bearing}: gespeichert {os.path.basename(png)} "
                  f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()