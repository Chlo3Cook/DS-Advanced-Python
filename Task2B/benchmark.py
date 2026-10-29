import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
from spike_detection import detect_onsets, detect_onsets_numpy, remove_drift, estimate_noise

# Same input for both versions: the drift-removed recording and the 4.5σ threshold
data = np.load(HERE / "data" / "spike_data.npz")
t = data["t"]
raw = data["raw_data"]
fs = 1 / np.median(np.diff(t))

flat = remove_drift(raw, fs)
threshold = 4.5 * estimate_noise(flat)

# --- Evaluation 1: correctness ---
onsets_loop = detect_onsets(flat, threshold)
onsets_numpy = detect_onsets_numpy(flat, threshold)

print(f"Loop version:  {len(onsets_loop)} onsets")
print(f"NumPy version: {len(onsets_numpy)} onsets")
print(f"Identical indices: {np.array_equal(onsets_loop, onsets_numpy)}")


# --- Evaluation 2: runtime ---
import time


def median_runtime(func, signal, threshold, repeats):
    """Run func several times and return the median runtime in milliseconds."""
    func(signal, threshold)                       # one warm-up run, not timed
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(signal, threshold)
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000


print("\nInput size   Repeats   Loop (ms)   NumPy (ms)   Speedup   Same output")
for copies, repeats in [(1, 50), (10, 20), (100, 5)]:
    signal = np.tile(flat, copies)                # 10,000 / 100,000 / 1,000,000 samples
    same = np.array_equal(detect_onsets(signal, threshold), detect_onsets_numpy(signal, threshold))
    t_loop = median_runtime(detect_onsets, signal, threshold, repeats)
    t_numpy = median_runtime(detect_onsets_numpy, signal, threshold, repeats)
    print(f"{len(signal):>10,}   {repeats:>7}   {t_loop:>9.2f}   {t_numpy:>10.3f}   {t_loop / t_numpy:>6.0f}x   {same}")