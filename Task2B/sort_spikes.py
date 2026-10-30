import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
from spike_detection import detect_spikes, remove_drift, estimate_noise
from waveforms import extract_waveforms

# Load and detect, exactly as in Part 5
data = np.load(HERE / "data" / "spike_data.npz")
t = data["t"]
raw = data["raw_data"]
fs = 1 / np.median(np.diff(t))

spikes = detect_spikes(raw, fs)          # 42 spike onsets
flat = remove_drift(raw, fs)             # the drift-removed signal the detector used

# Cut out each spike's shape
waveforms, peaks = extract_waveforms(flat, spikes)
print(f"Waveform table: {waveforms.shape[0]} spikes x {waveforms.shape[1]} samples")

# --- Figure: all waveforms overlaid ---
lags_ms = np.arange(-2, 9) / fs * 1000   # time relative to the peak, in ms

fig, ax = plt.subplots(figsize=(7, 4))
for w in waveforms:
    ax.plot(lags_ms, w, color="gray", alpha=0.4, linewidth=0.8)
ax.plot(lags_ms, waveforms.mean(axis=0), color="tab:blue", linewidth=2.5, label="average")
ax.set_xlabel("Time relative to peak (ms)")
ax.set_ylabel("Drift-removed voltage (a.u.)")
ax.set_title(f"All {waveforms.shape[0]} detected spike waveforms, aligned on the peak")
ax.legend()

results = HERE / "results"
results.mkdir(exist_ok=True)
fig.savefig(results / "waveforms.png", dpi=150)

# --- PCA on the waveforms (same steps as pca_toy.py) ---
W_centered = waveforms - waveforms.mean(axis=0)            # subtract the average spike shape
U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
explained = S**2 / np.sum(S**2)
scores = U * S                                              # each spike's coordinates along each PC

print("Fraction of variance explained by PC1-PC5:", np.round(explained[:5], 3))

peak_heights = waveforms[:, 2]                              # column 2 = the peak (2 samples after window start)
r = np.corrcoef(scores[:, 0], peak_heights)[0, 1]
print(f"Correlation between PC1 score and peak height: {r:.2f}")

# --- Figure: variance explained, component shapes, and scores ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(np.arange(1, 6), explained[:5] * 100)
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Variance explained (%)")
axes[0].set_title("How much each component explains")

axes[1].plot(lags_ms, Vt[0], marker="o", label=f"PC1 ({explained[0]:.0%})")
axes[1].plot(lags_ms, Vt[1], marker="o", label=f"PC2 ({explained[1]:.0%})")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].set_xlabel("Time relative to peak (ms)")
axes[1].set_ylabel("Component weight")
axes[1].set_title("What each component looks like")
axes[1].legend()

axes[2].scatter(scores[:, 0], scores[:, 1], s=25)
axes[2].set_xlabel("PC1 score")
axes[2].set_ylabel("PC2 score")
axes[2].set_title("Each dot = one spike")

plt.tight_layout()
fig.savefig(results / "waveform_pca.png", dpi=150)
spread = W_centered.var(axis=0).mean()
print(f"Average spike-to-spike variance per time point: {spread:.3f}")
print(f"Noise variance (sigma squared):                 {estimate_noise(flat)**2:.3f}")
plt.show()