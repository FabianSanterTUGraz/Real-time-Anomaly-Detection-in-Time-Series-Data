"""
TDE Window-Size Scan
=====================
Testet mehrere Fenstergrößen (window sizes) fuer die Time Delay Embedding (TDE)
Visualisierung nach Rakuschek et al. 2025 ("Visual Fingerprints of Vibration
Signals Using Time Delay Embeddings").

Ziel: herausfinden, bei welcher Fenstergroesse (falls ueberhaupt) sich eine
klare Ring-/Ellipsenstruktur zeigt, die auf Periodizitaet im Signal hindeutet.

Ablauf:
1. Rohsignal laden und als Sanity-Check plotten (kurzer Ausschnitt).
2. Autokorrelation berechnen, um eine grobe Schaetzung der Periodenlaenge
   in Samples zu bekommen (hilft bei der Wahl sinnvoller Fenstergroessen).
3. TDE fuer mehrere Fenstergroessen berechnen und nebeneinander plotten
   (wie Abbildung 5 im Paper).
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Konfiguration -- hier anpassen
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = r"C:\Users\Abuscom\Desktop\Real-time-Anomaly-Detection-in-Time-Series-Data\LASPI-Detection_and_diagnostics_of_bearing_gear_and_combined_faults_of_gearbox\Healthy_motor\35hz_0%_2090rpm\acc_00001.csv"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_SIGNAL_PLOT_PATH = os.path.join(OUTPUT_DIR, "raw_signal.png")
AUTOCORR_PLOT_PATH = os.path.join(OUTPUT_DIR, "autocorrelation.png")
WINDOW_SCAN_PLOT_PATH = os.path.join(OUTPUT_DIR, "window_scan.png")

# Falls bekannt: Abtastrate in Hz (steht meist in der README des Datensatzes).
# Falls unbekannt, auf None lassen -- die erwartete Periodenlaenge wird dann
# nur aus der Autokorrelation geschaetzt.
SAMPLING_RATE_HZ = None  # z.B. 20480

# Erwartete Rotationsfrequenz aus dem Dateinamen/Ordnernamen (hier: ~35 Hz)
EXPECTED_FREQ_HZ = 35.0

# Fenstergroessen, die getestet werden sollen (wie Abb. 5 im Paper: verdoppelnd)
WINDOW_SIZES = [20, 40, 80, 160, 320, 640, 1280]

# stride > 1 reduziert die Anzahl der Fenster (schneller, weniger Overplotting)
STRIDE = 5

# Wie viele Samples des Rohsignals fuer den Sanity-Check-Plot angezeigt werden
RAW_PLOT_SAMPLES = 3000


# ---------------------------------------------------------------------------
# Funktionen
# ---------------------------------------------------------------------------
def time_delay_embedding(values, d, tau=1, stride=1):
    """Sliding-Window Time Delay Embedding.

    Liefert eine Matrix der Form (n_windows, d).
    """
    values = np.asarray(values)
    n = len(values)
    n_windows = (n - (d - 1) * tau - 1) // stride + 1
    if n_windows <= 0:
        return np.empty((0, d))

    windows = np.empty((n_windows, d))
    for row, start in enumerate(range(0, n - (d - 1) * tau, stride)):
        idx = start + np.arange(d) * tau
        windows[row] = values[idx]
    return windows


def plot_embedding(ax, tde, title="offline-TDE"):
    """Projiziert die TDE-Matrix per PCA auf 2D und plottet sie, eingefaerbt
    nach Radius (Turbo-Colormap), wie im Paper beschrieben.

    WICHTIG: Normalisierung erfolgt zeilenweise (pro Fenster), nicht
    spaltenweise -- das entspricht der "z-normalized TDE" aus dem Paper und
    erfasst die Form/Phase jedes Fensters unabhaengig von lokalem Pegel.
    """
    if len(tde) < 3:
        ax.text(0.5, 0.5, "zu wenige Fenster", ha="center", va="center")
        ax.axis("off")
        ax.set_title(title, fontsize=14)
        return

    row_mean = tde.mean(axis=1, keepdims=True)
    row_std = tde.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1e-8
    tde_scaled = (tde - row_mean) / row_std

    pca = PCA(n_components=2, svd_solver="full")
    projected = pca.fit_transform(tde_scaled)

    scores = np.linalg.norm(projected, axis=1)
    if np.max(scores) - np.min(scores) > 0:
        scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    else:
        scores_norm = np.zeros_like(scores)

    ax.scatter(
        projected[:, 0], projected[:, 1],
        s=3, c=colormaps["turbo"](scores_norm), alpha=0.2,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)


def estimate_period_from_autocorrelation(values, max_lag=5000):
    """Grobe Schaetzung der dominanten Periodenlaenge (in Samples) ueber die
    Autokorrelationsfunktion. Sucht den ersten signifikanten lokalen
    Peak nach lag=0.
    """
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    n = len(values)
    max_lag = min(max_lag, n - 1)

    # Autokorrelation via FFT (schneller als direkte Berechnung)
    result = np.correlate(values, values, mode="full")
    result = result[result.size // 2:]
    result = result[: max_lag + 1]
    result /= result[0]  # normalisieren, sodass ac[0] == 1

    # ersten lokalen Peak nach lag=0 suchen (einfaches Kriterium)
    peak_lag = None
    for i in range(2, len(result) - 1):
        if result[i] > result[i - 1] and result[i] > result[i + 1] and result[i] > 0.1:
            peak_lag = i
            break

    return result, peak_lag


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    print(f"Lade Daten von: {DATA_PATH}")
    values = pd.read_csv(DATA_PATH, header=None)[0].to_numpy()
    print(f"Anzahl Samples: {len(values)}")
    print(f"Mean: {values.mean():.4f}  Std: {values.std():.4f}  "
          f"Min: {values.min():.4f}  Max: {values.max():.4f}")

    # Signal zentrieren (falls DC-Offset vorhanden)
    values_centered = values - values.mean()

    # ---- 1. Sanity-Check: Rohsignal plotten -----------------------------
    n_show = min(RAW_PLOT_SAMPLES, len(values_centered))
    plt.figure(figsize=(12, 3))
    plt.plot(values_centered[:n_show], linewidth=0.7)
    plt.title(f"Rohsignal (erste {n_show} Samples, zentriert)")
    plt.xlabel("Sample-Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(RAW_SIGNAL_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Rohsignal-Plot gespeichert: {RAW_SIGNAL_PLOT_PATH}")

    # ---- 2. Autokorrelation zur Periodenschaetzung ----------------------
    autocorr, peak_lag = estimate_period_from_autocorrelation(values_centered)

    plt.figure(figsize=(10, 3))
    plt.plot(autocorr)
    if peak_lag is not None:
        plt.axvline(peak_lag, color="red", linestyle="--",
                    label=f"erster Peak bei lag={peak_lag}")
        plt.legend()
    plt.title("Autokorrelation (zur groben Periodenschaetzung)")
    plt.xlabel("Lag (Samples)")
    plt.ylabel("Korrelation")
    plt.tight_layout()
    plt.savefig(AUTOCORR_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Autokorrelations-Plot gespeichert: {AUTOCORR_PLOT_PATH}")

    if peak_lag is not None:
        print(f"Geschaetzte dominante Periodenlaenge: {peak_lag} Samples")
    else:
        print("Kein klarer periodischer Peak in der Autokorrelation gefunden "
              "(Signal evtl. stark verrauscht oder nicht periodisch).")

    if SAMPLING_RATE_HZ is not None:
        expected_period_samples = SAMPLING_RATE_HZ / EXPECTED_FREQ_HZ
        print(f"Erwartete Periodenlaenge basierend auf "
              f"{EXPECTED_FREQ_HZ} Hz bei {SAMPLING_RATE_HZ} Hz Abtastrate: "
              f"{expected_period_samples:.1f} Samples")

    # ---- 3. Fenstergroessen-Scan (wie Abb. 5 im Paper) ------------------
    # Falls die Autokorrelation einen Peak gefunden hat, diesen und ein paar
    # Vielfache/Teiler mit in die Liste der Fenstergroessen aufnehmen.
    window_sizes = list(WINDOW_SIZES)
    if peak_lag is not None and peak_lag > 1:
        for factor in (0.5, 1, 2):
            candidate = int(round(peak_lag * factor))
            if candidate > 3 and candidate not in window_sizes:
                window_sizes.append(candidate)
    window_sizes = sorted(set(window_sizes))

    n_plots = len(window_sizes)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    for ax, d in zip(axes, window_sizes):
        tde = time_delay_embedding(values_centered, d=d, tau=1, stride=STRIDE)
        print(f"d={d}: {len(tde)} Fenster")
        plot_embedding(ax, tde, title=f"w={d}")

    plt.tight_layout()
    plt.savefig(WINDOW_SCAN_PLOT_PATH, dpi=150)
    plt.show()
    print(f"Fenstergroessen-Scan gespeichert: {WINDOW_SCAN_PLOT_PATH}")


if __name__ == "__main__":
    main()