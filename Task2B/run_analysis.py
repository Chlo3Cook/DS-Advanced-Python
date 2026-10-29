import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent                 # the Task2B folder, wherever this is run from
sys.path.insert(0, str(HERE / "src"))        # let Python find spike_detection.py
from spike_detection import detect_spikes, remove_drift, estimate_noise

# Load the recording
data = np.load(HERE / "data" / "spike_data.npz")
t = data["t"]
raw = data["raw_data"]

# Sampling rate and duration, computed from the data (not typed in)
fs = 1 / np.median(np.diff(t))
duration = len(t) / fs

# Run the detector with the 2A settings (window 0.2 s, alpha 4.5, dead time 8 ms)
spikes = detect_spikes(raw, fs)
n_spikes = len(spikes)
rate = n_spikes / duration

print(f"Sampling rate: {fs:.0f} Hz, duration: {duration:.1f} s")
print(f"Spikes detected: {n_spikes}")
print(f"Firing rate: {rate:.2f} Hz")


# Sensitivity: how does the count change with alpha and dead time?
print("\nalpha   spikes (5 ms)   spikes (8 ms)   rate 8 ms (Hz)")
for alpha in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]:
    n5 = len(detect_spikes(raw, fs, alpha=alpha, dead_time_s=0.005))
    n8 = len(detect_spikes(raw, fs, alpha=alpha, dead_time_s=0.008))
    print(f"{alpha:5.1f}   {n5:13d}   {n8:13d}   {n8 / duration:14.2f}")

spikes_8 = detect_spikes(raw, fs)            # default settings: alpha 4.5, dead time 8 ms
gaps_ms = np.diff(t[spikes_8]) * 1000
print(f"\nSmallest gap between spikes (8 ms dead time): {gaps_ms.min():.0f} ms")

# --- Borderline events: found at alpha 4.5 but not at alpha 6 ---
flat = remove_drift(raw, fs)
sigma = estimate_noise(flat)

spikes_main = detect_spikes(raw, fs, alpha=4.5, dead_time_s=0.005)
spikes_strict = detect_spikes(raw, fs, alpha=6.0)

tol = 3   # samples: the same spike can cross slightly earlier at a lower threshold
matched = np.array([np.any(np.abs(spikes_strict - i) <= tol) for i in spikes_main])
borderline = spikes_main[~matched]

print(f"\nBorderline events (found at alpha 4.5, not at 6.0): {len(borderline)}")
for i in borderline:
    peak = flat[i:i + 5].max()   # highest point in the 5 ms after the onset
    print(f"  t = {t[i]:.3f} s, peak = {peak / sigma:.1f} sigma above baseline")

# --- Figure: detected spikes on the trace ---
results = HERE / "results"
results.mkdir(exist_ok=True)   # create the folder if it doesn't exist yet

# --- What did the 8 ms dead time remove? ---
from spike_detection import detect_onsets   # if not already imported at the top

all_onsets = detect_onsets(flat, 4.5 * sigma)            # every crossing, before dead time
dropped = np.setdiff1d(all_onsets, spikes_8)              # crossings the dead time removed
print(f"\nCrossings removed by the 8 ms dead time: {len(dropped)}")
for i in dropped:
    prev = spikes_8[spikes_8 < i].max()                   # the kept spike just before it
    peak = flat[i:i + 5].max() / sigma
    print(f"  t = {t[i]:.3f} s, {(i - prev) / fs * 1000:.0f} ms after a kept spike, peak {peak:.1f} sigma")




fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].plot(t, raw, linewidth=0.5, color="gray")
axes[0].plot(t[spikes_main], np.full(len(spikes_main), raw.max() + 0.5), "v", color="tab:blue")
axes[0].set_ylabel("Voltage (a.u.)")
axes[0].set_title(f"Raw recording with {len(spikes_main)} detected spikes (alpha = 4.5)")

axes[1].plot(t, flat, linewidth=0.5, color="gray")
axes[1].axhline(4.5 * sigma, color="tab:red", linestyle="--", label="threshold (4.5σ)")
axes[1].plot(t[spikes_main], flat[spikes_main], "o", color="tab:blue", markersize=4, label="detected")
axes[1].plot(t[borderline], flat[borderline], "o", color="tab:orange", markersize=8,
             markerfacecolor="none", label="borderline (not at 6σ)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Voltage (a.u.)")
axes[1].set_title("Drift-removed signal")
axes[1].legend()

plt.tight_layout()
fig.savefig(results / "detected_spikes.png", dpi=150)
plt.show()


# --- Figure: zoom on borderline events vs a clear spike ---
clear = spikes_strict[np.argmin(np.abs(t[spikes_strict] - 4.69))]   # the clear spike near 4.69 s (Figure C in 2A)
to_show = [("Clear spike", clear)] + [("Borderline", i) for i in borderline]

fig, axes = plt.subplots(1, len(to_show), figsize=(14, 4), squeeze=False)
axes = axes[0]
half = int(0.015 * fs)   # ±15 ms around each event

for ax, (label, i) in zip(axes, to_show):
    window = slice(max(i - half, 0), min(i + half, len(flat)))
    ax.plot(t[window], flat[window], marker=".", linewidth=0.8)
    ax.axhline(4.5 * sigma, color="tab:red", linestyle="--", label="4.5σ")
    ax.axhline(6.0 * sigma, color="tab:purple", linestyle=":", label="6σ")
    peak = flat[i:i + 5].max() / sigma
    ax.set_title(f"{label} at {t[i]:.3f} s ({peak:.1f}σ)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drift-removed voltage (a.u.)")
axes[0].legend()

plt.tight_layout()
fig.savefig(results / "borderline_zoom.png", dpi=150)
plt.show()