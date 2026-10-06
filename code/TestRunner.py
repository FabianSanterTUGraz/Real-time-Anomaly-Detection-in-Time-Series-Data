import os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.pyplot import colormaps

d = 100
tau = 1
window_size = 2560
ACC_COL =  4
N_PER_BLOCK = 5   # Dateien am Anfang / am Ende

BASE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(BASE, "anomaly_detection.exe")
INPUT = os.path.join(BASE, "Data", "input.csv")
OUTPUT = os.path.join(BASE, "output", "output.txt")
DATASET = os.path.join(BASE, "..", "ieee-phm-2012-data-challenge-dataset", "Full_Test_Set")
RESULTS = os.path.join(BASE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def fingerprint(acc_file):
    """Eine acc-Datei -> ein C++-Lauf -> Fingerprint (L, 2) oder None."""
    df = pd.read_csv(acc_file, header=None, sep=None, engine="python")
    df[ACC_COL].to_csv(INPUT, index=False, header=False)
    subprocess.run([EXE, str(d), str(tau), str(window_size), "input"],
                   cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    block = open(OUTPUT).read().split("---")[0].strip()
    return np.loadtxt(block.splitlines(), delimiter=",") if block else None


def plot_blocks(png_path, title, block1, block2, block1_title, block2_title):
    n1, n2 = len(block1), len(block2)
    fig = plt.figure(figsize=(2.5 * (n1 + n2 + 1), 3.3))
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


for bearing in sorted(Path(DATASET).iterdir()):
    files = sorted(bearing.glob("acc_*.csv")) if bearing.is_dir() else []
    if len(files) < 2 * N_PER_BLOCK:
        continue
    first = list(range(N_PER_BLOCK))
    last = list(range(len(files) - N_PER_BLOCK, len(files)))

    blocks = []
    for idxs in (first, last):
        b = []
        for i in idxs:
            fp = fingerprint(files[i])
            if fp is None:
                print("kein Fingerprint:", bearing.name, files[i].name)
            else:
                b.append((f"#{i}", fp))
        blocks.append(b)
    if not blocks[0] or not blocks[1]:
        continue

    plot_blocks(os.path.join(RESULTS, f"{bearing.name}.png"),
                f"{bearing.name} | d = {d} | tau = {tau}",
                blocks[0], blocks[1],
                f"Anfang (Dateien #{first[0]} – #{first[-1]})",
                f"Ende (Dateien #{last[0]} – #{last[-1]})")