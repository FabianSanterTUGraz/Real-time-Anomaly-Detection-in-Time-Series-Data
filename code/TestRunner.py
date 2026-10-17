"""
ims_fingerprints.py  (lives next to TestRunner.py in code/)

Plots fingerprints of selected IMS files as one strip like Fig. 6 of Rakuschek et al.:
N_FIRST consecutive files starting at FIRST_START (healthy phase, clean rings), a zigzag,
and the last N_LAST files of the run (failure). The date of each file is shown above it.
Same selection as static_batch.py, so both strips can be compared directly.

MODE = "files"    -> RESTART mode: every selected ORIGINAL snapshot file is fed on its own
                     into a fresh C++ run (like TestRunner.py). Nothing is reused.
MODE = "combined" -> STREAM mode: the WHOLE combined file 0000.00.00.00.00.00 (all files of the
                     set, made by combine.py with N_FILES = None) is fed into ONE C++ run. The
                     sliding window moves through the entire stream and mean/scatter/basis are
                     reused across file boundaries. Afterwards the fingerprints of the selected
                     files are cut out of that one output.
MODE = "test"     -> STREAM mode on the short test file 1111.11.11.11.11.11 (TEST_N_FILES
                     consecutive files starting at TEST_FIRST_INDEX, made by combine.py). All
                     fingerprints of the test file are plotted. TEST_FIRST_INDEX must be the same
                     value as FIRST_INDEX in combine.py, otherwise the dates are wrong.

Output format of the C++ driver is detected automatically:
  WINDOW  : after every windowSize samples a block of x,y lines, then "---".
            Block i = window that ends with the last sample of file i (= exactly file i).
  ARRIVAL : one x,y line per TDE vector, projected at arrival with the basis of that moment.
            Line k = vector ending at sample k + (d-1)*tau; vectors spanning two files are dropped.

Both modes use the same selection and the same plot settings and save into the same folder
(suffix _files / _stream / _test), so the images of both modes can be laid on top of each other.

Timing: every C++ run is timed. Two numbers are reported:
  wall : whole subprocess (process start + reading input.csv + core + writing output.txt)
  core : only the processNewDataPoint calls, IF the C++ driver prints a line "core_ms=<value>"
         (see the snippet in the README / chat). Without that line only wall time is shown.
Every run is appended to results/timing.csv (one row per C++ run), tagged with VERSION_LABEL.
Run-time estimate: before a run the expected duration is computed from earlier rows in timing.csv
(same VERSION_LABEL, d, tau, window); during the run the remaining time is updated.
Note: in RESTART mode every run is exactly one file = only the FILLING phase (window never slides).
The sliding-phase speed (O(d) lag ring) is only visible in STREAM mode.
"""
import os, re, csv, time, shutil, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps

# ============================================================
# Parameters
# ============================================================
MODE = "test"                   # test -> original , combined -> ganze setup combined , files -> über jede einzeln
RUN_CPP = True                  # combined/test only: False = re-plot the stored output of the last run

d = 1500
tau = 1
window_size = 20480             # = one original file
SNAPSHOT_LEN = 20480            # samples per original IMS file

TEST_SET = "2nd_test"           # "1st_test", "2nd_test" or "3rd_test"
BEARING = 1                     # bearing NUMBER 1-4 (not the column index)

FIRST_START = 200               # index of the first file of the left block (healthy phase, clean rings)
N_FIRST = 6                     # consecutive files in the left block
N_LAST = 4                      # last files of the run in the right block (failure)

SHOW_PLOTS = True               # show the strip figure (False: only save the PNG)
SAVE_SINGLE_PLOTS = True        # additionally one PNG per file (for overlaying with other versions)

VERSION_LABEL = "lagring"       # free text written to results/timing.csv, e.g. "rank1", "lagring", "lagring_O3"
SAMPLE_RATE = 20000             # Hz, real time = SAMPLE_RATE samples per second

COMBINED_NAME = "0000.00.00.00.00.00"   # all files of the set
TEST_NAME = "1111.11.11.11.11.11"       # short test file
TEST_FIRST_INDEX = 200                  # = FIRST_INDEX in combine.py
TEST_N_FILES = 10                       # = N_FILES in combine.py
STREAM_NAME = {"combined": COMBINED_NAME, "test": TEST_NAME}
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
OUT_DIR = os.path.join(RESULTS, f"compare_IMS_{TEST_SET}_bearing{BEARING}_d{d}")
STORED_OUTPUT = os.path.join(os.path.dirname(OUTPUT), f"{MODE}_{TEST_SET}_bearing{BEARING}_d{d}_tau{tau}_w{window_size}.txt")
OFFSET = (d - 1) * tau
TIMING_CSV = os.path.join(RESULTS, "timing.csv")
CORE_RE = re.compile(r"core_ms\s*=\s*([0-9.eE+-]+)")
for p in (OUT_DIR, os.path.dirname(INPUT), os.path.dirname(OUTPUT)):
    os.makedirs(p, exist_ok=True)


# ============================================================
# Files
# ============================================================
def snapshot_files():
    """Original snapshot files, chronological (the combined files 0000... and 1111... are skipped)."""
    for folder in (os.path.join(IMS_ROOT, TEST_SET, TEST_SET), os.path.join(IMS_ROOT, TEST_SET)):
        if os.path.isdir(folder):
            files = sorted(f for f in Path(folder).iterdir()
                           if f.is_file() and FILENAME_RE.match(f.name) and not f.name.startswith(("0000", "1111")))
            if files:
                return files
    return []


def find_combined_file(base_name):
    """The combined file sits next to the snapshot files (where combine.py writes it) or one level
    above: IMS/<set>/<set>/<name> or IMS/<set>/<name>, with or without .txt."""
    folders = (os.path.join(IMS_ROOT, TEST_SET, TEST_SET), os.path.join(IMS_ROOT, TEST_SET))
    for folder in folders:
        for name in (base_name, base_name + ".txt"):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(f"{base_name} nicht gefunden in {' oder '.join(folders)} - erst combine.py laufen lassen")


def select_blocks(n_files):
    """Left block: N_FIRST consecutive files from FIRST_START. Right block: the last N_LAST files.
    The left block is moved forward if it would overlap the right one."""
    last_block = list(range(max(0, n_files - N_LAST), n_files))
    start = max(0, min(FIRST_START, n_files - N_LAST - N_FIRST))
    first_block = list(range(start, min(start + N_FIRST, n_files - N_LAST)))
    return first_block, last_block


# ============================================================
# C++ run and plotting
# ============================================================
# ============================================================
# Timing
# ============================================================
def run_exe(expected_s=None, progress_every=15):
    """Runs the C++ program once. Returns (wall seconds, core milliseconds or None).
    With expected_s (long stream run) the elapsed and remaining time is printed every progress_every seconds."""
    t0 = time.perf_counter()
    proc = subprocess.Popen([EXE, str(d), str(tau), str(window_size), "input"],
                            cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    while True:
        try:
            out, _ = proc.communicate(timeout=progress_every if expected_s else None)
            break
        except subprocess.TimeoutExpired:
            el = time.perf_counter() - t0
            if expected_s == float("inf"):
                msg = f"läuft seit {fmt_dur(el)}"
            elif el < expected_s:
                msg = (f"läuft seit {fmt_dur(el)}, noch ca. {fmt_dur(expected_s - el)} "
                       f"(fertig ca. {finish_clock(expected_s - el)}, {100 * el / expected_s:.0f} %)")
            else:
                msg = f"läuft seit {fmt_dur(el)}, schon {fmt_dur(el - expected_s)} länger als geschätzt"
            print("\r  " + msg + " " * 10, end="", flush=True)
    if expected_s:
        print()
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, EXE, output=out)
    m = CORE_RE.search(out or "")
    return wall, (float(m.group(1)) if m else None)


def fmt_dur(seconds):
    seconds = max(0, int(round(seconds)))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def finish_clock(seconds_from_now):
    return datetime.fromtimestamp(time.time() + seconds_from_now).strftime("%H:%M")


def history_wall_per_sample(mode):
    """Wall seconds per sample from earlier runs with the same version and parameters (median of the last 5).
    Prefers rows of the same mode. Returns (seconds per sample, mode of the rows used) or (None, None)."""
    if not os.path.isfile(TIMING_CSV):
        return None, None
    with open(TIMING_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("version") == VERSION_LABEL and r.get("d") == str(d) and r.get("tau") == str(tau)
                and r.get("window") == str(window_size) and r.get("wall_s") and r.get("n_samples")]
    for wanted in (mode, "stream" if mode == "files" else "files"):
        sel = [float(r["wall_s"]) / int(r["n_samples"]) for r in rows if r["mode"] == wanted][-5:]
        if sel:
            return float(np.median(sel)), wanted
    return None, None


def describe(n_samples, wall, core_ms):
    """One line: wall time, core time, ms per sample, real-time factor (>1 = faster than real time)."""
    budget_ms = 1000.0 / SAMPLE_RATE
    if core_ms is not None:
        per = core_ms / n_samples
        return (f"wall {wall:.2f} s | Kern {core_ms / 1000:.2f} s = {per:.4f} ms/Sample "
                f"(Budget {budget_ms:.3f}, Echtzeitfaktor {budget_ms / per:.1f}x)")
    per = wall * 1000 / n_samples
    return f"wall {wall:.2f} s = {per:.4f} ms/Sample inkl. I/O (Echtzeitfaktor {budget_ms / per:.1f}x)"


def log_timing(mode, label, n_samples, wall, core_ms):
    """Appends one row per C++ run to results/timing.csv."""
    new = not os.path.isfile(TIMING_CSV)
    with open(TIMING_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "version", "mode", "file", "test_set", "bearing", "d", "tau", "window",
                        "n_samples", "wall_s", "core_ms", "ms_per_sample", "realtime_factor"])
        per = (core_ms if core_ms is not None else wall * 1000) / n_samples
        w.writerow([datetime.now().isoformat(timespec="seconds"), VERSION_LABEL, mode, label, TEST_SET, BEARING,
                    d, tau, window_size, n_samples, f"{wall:.4f}",
                    "" if core_ms is None else f"{core_ms:.3f}", f"{per:.6f}", f"{(1000.0 / SAMPLE_RATE) / per:.3f}"])


def run_cpp(values):
    """values -> Data/input.csv -> one fresh C++ run -> (fingerprint (L, 2) or None, wall s, core ms)."""
    pd.Series(values).to_csv(INPUT, index=False, header=False)
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)                      # never plot a stale output
    wall, core_ms = run_exe()
    if not os.path.exists(OUTPUT):
        return None, wall, core_ms
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
    return (np.array(rows) if rows else None), wall, core_ms


def run_cpp_stream(col, base_name):
    """WHOLE combined column -> Data/input.csv -> ONE C++ run. Output is copied to STORED_OUTPUT."""
    combined = find_combined_file(base_name)
    print("Lese", combined, "(nur eine Spalte) ...", flush=True)
    sig = pd.read_csv(combined, header=None, sep="\t", usecols=[col], dtype=np.float64)[col]
    print(f"Schreibe {len(sig):,} Samples nach {INPUT} ...", flush=True)
    sig.to_csv(INPUT, index=False, header=False)
    n_samples = len(sig)
    del sig
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)
    print(f"Starte C++ EINMAL über den ganzen Strom (d={d}, tau={tau}, window={window_size}) ...", flush=True)
    per, src = history_wall_per_sample("stream")
    expected = per * n_samples if per else None
    if expected is None:
        print(f"  Keine früheren Läufe mit VERSION_LABEL={VERSION_LABEL!r}, d={d} in timing.csv - "
              f"Dauer unbekannt, es wird nur die verstrichene Zeit angezeigt.")
        expected = float("inf")
    elif src == "files":
        print(f"  Schätzung aus Einzeldatei-Läufen (nur Füllphase): ca. {fmt_dur(expected)} "
              f"(fertig ca. {finish_clock(expected)}). Für die O(d)-Version ist das eine OBERE Grenze, "
              f"die Sliding-Phase ist schneller.")
    else:
        print(f"  Geschätzte Dauer: ca. {fmt_dur(expected)} (fertig ca. {finish_clock(expected)}), "
              f"aus früheren Stromläufen")
    wall, core_ms = run_exe(expected_s=expected)
    print(f"C++ fertig nach {wall / 60:.1f} min: {describe(n_samples, wall, core_ms)}", flush=True)
    log_timing("stream", base_name, n_samples, wall, core_ms)
    if not os.path.exists(OUTPUT):
        raise RuntimeError("C++ hat keine output.txt geschrieben")
    shutil.copyfile(OUTPUT, STORED_OUTPUT)


def is_window_format(path, max_lines=4 * SNAPSHOT_LEN):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for k, line in enumerate(fh):
            if line.strip() == "---":
                return True
            if k > max_lines:
                break
    return False


def count_stream_files(path, window_mode):
    if window_mode:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip() == "---")
    with open(path, "rb") as fh:
        n_rows = sum(1 for _ in fh)
    return (n_rows + OFFSET) // SNAPSHOT_LEN


def read_stream_segments(path, wanted, window_mode):
    """{file index: (n, 2) array} for the wanted files only, read in one pass over the output."""
    wanted = set(wanted)
    segs = {i: [] for i in wanted}
    block, row, bad = 0, 0, 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if window_mode:
                if line == "---":
                    block += 1
                    continue
                target = block
            else:
                k = row                        # counts EVERY line, also broken ones
                row += 1
                target = k // SNAPSHOT_LEN if (k % SNAPSHOT_LEN) < SNAPSHOT_LEN - OFFSET else None
            if target not in wanted:
                continue
            try:
                x, y = (float(v) for v in line.split(","))
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


def plot_fingerprint(fp, title, png):
    r = np.linalg.norm(fp - fp.mean(0), axis=1)
    rn = (r - r.min()) / (r.max() - r.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    ax.scatter(fp[:, 0], fp[:, 1], s=8, c=colormaps["turbo"](rn), alpha=0.6)
    limit = np.abs(fp).max() * 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=20)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


def draw_ring(ax, fp, label):
    """One small fingerprint of the strip: turbo colour by radius, date above."""
    r = np.linalg.norm(fp - fp.mean(0), axis=1)
    rn = (r - r.min()) / (r.max() - r.min() + 1e-12)
    ax.scatter(fp[:, 0], fp[:, 1], s=1.5, c=colormaps["turbo"](rn), alpha=0.6, linewidths=0)
    limit = np.abs(fp).max() * 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(label, fontsize=8)


def plot_strip(first, last, title, png):
    """Like Fig. 6 of the paper: left block, zigzag, right block in one row.
    first, last: lists of (label, points)."""
    n1, n2 = len(first), len(last)
    ratios = [1] * n1 + [0.35] + [1] * n2
    fig, axes = plt.subplots(1, n1 + 1 + n2, figsize=(2.2 * (n1 + n2) + 0.8, 2.9),
                             gridspec_kw={"width_ratios": ratios})
    for ax, (label, fp) in zip(axes[:n1], first):
        draw_ring(ax, fp, label)
    for ax, (label, fp) in zip(axes[n1 + 1:], last):
        draw_ring(ax, fp, label)
    zz = axes[n1]
    y = np.linspace(0.15, 0.85, 9)
    x = 0.5 + 0.3 * np.array([(-1) ** k for k in range(9)])
    zz.plot(x, y, color="black", linewidth=2.5)
    zz.set_xlim(0, 1)
    zz.set_ylim(0, 1)
    zz.axis("off")
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    if BEARING not in SET_CHANNELS.get(TEST_SET, {}):
        raise ValueError(f"TEST_SET={TEST_SET!r} / BEARING={BEARING!r} ungültig (Lagernummer 1-4, nicht Spaltenindex)")
    col = SET_CHANNELS[TEST_SET][BEARING]
    note = FAILURE_NOTE.get(TEST_SET, {}).get(BEARING, "no failure recorded")
    files = snapshot_files()
    names = [f.name for f in files]
    if MODE not in ("files", "combined", "test"):
        raise ValueError(f"MODE={MODE!r} ungültig: 'files', 'combined' oder 'test'")
    if (MODE == "files" or RUN_CPP) and not os.path.isfile(EXE):
        raise FileNotFoundError(f"C++-Programm nicht gefunden: {EXE} - erst bauen (mingw32-make)")

    if MODE in ("combined", "test"):
        # ---------- STREAM: one C++ run over the whole file, then cut out the selected files
        source = "stream" if MODE == "combined" else "test"
        # block i of the output belongs to original file first_index + i
        first_index = 0 if MODE == "combined" else TEST_FIRST_INDEX
        if RUN_CPP:
            run_cpp_stream(col, STREAM_NAME[MODE])
        elif not os.path.isfile(STORED_OUTPUT):
            raise FileNotFoundError(f"Kein gespeicherter Output {STORED_OUTPUT} - RUN_CPP = True setzen")
        window_mode = is_window_format(STORED_OUTPUT)
        n_files = count_stream_files(STORED_OUTPUT, window_mode)
        print(f"Output-Format: {'WINDOW (neu projiziert pro Fenster)' if window_mode else 'ARRIVAL (Projektion bei Ankunft)'}, "
              f"{n_files} Dateien im Output")
        expected_files = len(names) if MODE == "combined" else TEST_N_FILES
        if names and n_files != expected_files:
            print(f"WARNUNG: {n_files} Dateien im Output, erwartet {expected_files} - "
                  f"Datumsangaben passen dann nicht (combine.py und TEST_* prüfen)")
        if MODE == "combined":
            first_block, last_block = select_blocks(n_files)
        else:   # test: all fingerprints, split into two halves around the zigzag
            half = (n_files + 1) // 2
            first_block, last_block = list(range(half)), list(range(half, n_files))
        indices = first_block + last_block
        print(f"[{source}] {TEST_SET} bearing {BEARING} ({note}), d={d}, tau={tau}: "
              f"links {first_block}, rechts {last_block}")
        segments = read_stream_segments(STORED_OUTPUT, indices, window_mode)
        fps = {}
        for i in indices:
            j = first_index + i               # index of the original file
            name = names[j] if j < len(names) else f"file{j}"
            if i not in segments:
                print(f"  #{j} {name}: kein Fingerprint")
                continue
            fps[i] = (name, segments[i])
            if SAVE_SINGLE_PLOTS:
                plot_fingerprint(segments[i], f"#{j}  {name}", os.path.join(OUT_DIR, f"{j:04d}_{name}_{source}.png"))
        save_strip(fps, first_block, last_block, source, note, first_index)
        return

    # ---------- RESTART: fresh C++ run for every selected original file
    source = "files"
    if not files:
        raise FileNotFoundError(f"{TEST_SET}: keine Snapshot-Dateien unter {IMS_ROOT}")
    first_block, last_block = select_blocks(len(files))
    indices = first_block + last_block
    print(f"[files] {TEST_SET} bearing {BEARING} ({note}), d={d}, tau={tau}: "
          f"{len(files)} Dateien, links {first_block}, rechts {last_block}")
    fps = {}
    per, src = history_wall_per_sample("files")
    if per:
        est = per * SNAPSHOT_LEN * len(indices)
        print(f"Geschätzte Dauer C++ (ohne Plotten): ca. {fmt_dur(est)} (fertig ca. {finish_clock(est)})"
              + ("" if src == "files" else " - aus Stromläufen geschätzt, Füllphase ist langsamer"))
    total_samples, total_wall, total_core, core_ok = 0, 0.0, 0.0, True
    t_loop = time.perf_counter()
    for k, i in enumerate(indices, 1):
        values = pd.read_csv(files[i], header=None, sep="\t", usecols=[col])[col].to_numpy()
        fp, wall, core_ms = run_cpp(values)
        log_timing("files", names[i], len(values), wall, core_ms)
        total_samples += len(values)
        total_wall += wall
        if core_ms is None:
            core_ok = False
        else:
            total_core += core_ms
        if fp is None:
            print(f"  [{k}/{len(indices)}] #{i} {names[i]}: kein Fingerprint ({describe(len(values), wall, core_ms)})")
            continue
        fps[i] = (names[i], fp)
        if SAVE_SINGLE_PLOTS:
            plot_fingerprint(fp, f"#{i}  {names[i]}", os.path.join(OUT_DIR, f"{i:04d}_{names[i]}_{source}.png"))
        left = (time.perf_counter() - t_loop) / k * (len(indices) - k)   # measured, incl. reading + plotting
        eta = f" | noch ca. {fmt_dur(left)} (fertig ca. {finish_clock(left)})" if k < len(indices) else ""
        print(f"  [{k}/{len(indices)}] #{i} {names[i]}: {describe(len(values), wall, core_ms)}{eta}", flush=True)

    if total_samples:
        print(f"Gesamt ({len(indices)} Läufe, nur Füllphase): "
              f"{describe(total_samples, total_wall, total_core if core_ok else None)}")
        print(f"Zeiten angehängt an {TIMING_CSV}")
    save_strip(fps, first_block, last_block, source, note)


def save_strip(fps, first_block, last_block, source, note, first_index=0):
    first = [fps[i] for i in first_block if i in fps]
    last = [fps[i] for i in last_block if i in fps]
    if not first or not last:
        print("Zu wenige Fingerprints für den Streifen")
        return
    png = os.path.join(OUT_DIR, f"strip_{first_index + first_block[0]:04d}-{first_index + first_block[-1]:04d}_last{len(last_block)}_{source}.png")
    plot_strip(first, last, f"IMS {TEST_SET}, bearing {BEARING} ({note}), d = {d}, {source}", png)
    print(f"Streifen gespeichert: {png}")


if __name__ == "__main__":
    main()