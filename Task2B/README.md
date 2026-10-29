# Assignment 2B: Spike detection

## Part 4: Detector and tests

The detector (`src/spike_detection.py`) implements my Part 3 specification as five small
functions: `remove_drift` (0.2 s moving median), `estimate_noise` (MAD), `detect_onsets`
(upward threshold crossings), `apply_dead_time` (5 ms), and `detect_spikes`, which chains
them with a threshold of α × σ (α = 4.5). It returns the sample index of each spike's onset.

Run the tests from the `Task2B` folder with `pytest`.

### How the tests establish correctness

Each test runs a function on a small, hand-built signal where I know the correct answer,
and checks the output against it. Each one targets a specific way the detector could fail:

- **No samples above threshold** → zero events (no false detections).
- **One isolated sample above** → one event at the right index.
- **Several consecutive samples above** → one event, not several (each spike counted once).
- **Two separate intervals** → two events (separate spikes are not merged).
- **Signal starts above threshold** → an event at index 0 (the first sample has no
  previous sample, so it needs a special rule).
- **Dead time** → an onset 3 ms after another is dropped; onsets 10 ms apart are both kept.
- **Drift removal** → a constant offset becomes zero.
- **Noise estimate** → on simulated noise with SD = 1, MAD returns ≈ 1.

Each test was written before or alongside the code and checked to fail when the code was
missing, so a passing test shows the code, not the test, is doing the work.

### What the tests do not show

The tests show the code follows my spike definition correctly. They do not show that the
definition itself (the window, α, dead time) is right for the real recording, since there
is no ground truth for it. That is evaluated in Part 5 through the visual check,
comparison with my eyeball count, and sensitivity to the parameters.


## Part 5: Firing rate and sensitivity

### Primary estimate

**42 spikes in 10 s = 4.2 Hz**, using α = 4.5, a 0.2 s drift window, and an 8 ms dead time.
This falls within my eyeball estimate from 2A (35–45 spikes, ~4 Hz).

### Parameter comparison

| α | Spikes (5 ms dead time) | Spikes (8 ms dead time) | Rate, 8 ms (Hz) |
|---|---|---|---|
| 3.0 | 55 | 49 | 4.9 |
| 3.5 | 49 | 44 | 4.4 |
| 4.0 | 49 | 43 | 4.3 |
| 4.5 | 44 | 42 | 4.2 |
| 5.0 | 43 | 42 | 4.2 |
| 5.5 – 7.0 | 42 | 42 | 4.2 |

Figures: `results/detected_spikes.png` (detected spikes on the trace) and
`results/borderline_zoom.png` (the two events that changed the dead time).

### How stable is the estimate?

With my original 5 ms dead time, the count was 44 at α = 4.5 but 42 at α ≥ 5.5. Zooming in
on the two extra events showed they were not separate spikes: each was the decaying tail of a
spike dipping below the threshold and crossing back above it exactly 5 ms after the spike's
onset, so the 5 ms dead time let it through. I increased the dead time to 8 ms. This removed
only those two crossings (both 5 ms after a detected spike), and the smallest remaining gap
between spikes is 11 ms, so no real spikes appear to have been merged.

With the 8 ms dead time, the count is **42 at every α from 4.5 to 7**, so the estimate does not
depend on the exact threshold within that range. Below α ≈ 4.5 the count rises (49 at α = 3),
as the threshold approaches the noise level and noise begins to cross it. I consider the
estimate stable at 4.2 Hz, provided α is kept at 4.5 or above.

### Assumptions

Interpreting 4.2 Hz as this neuron's firing rate during the recording assumes:

- **Every detected event is a spike from one neuron.** Spikes have similar heights, which suggests
  a single unit, but no spike sorting was done.
- **No spikes are hidden in the noise.** Spikes smaller than ~4.5σ would be missed.
- **No two spikes occur within 8 ms of each other.** Closer pairs would be counted as one.
  The smallest observed gap is 11 ms.
- **Spikes are positive-going**, so negative deflections are ignored.
- **The noise is roughly constant and close to Gaussian**, so one σ applies to the whole recording.
- **The drift is fully removed** by the 0.2 s moving median.
- **The recording is continuous** (10,000 samples at 1 kHz, no gaps), and 1 kHz sampling is
  enough to capture each spike.
- **A single average is meaningful:** 4.2 Hz describes the whole 10 s, and does not show whether
  the neuron fired faster or slower at different times.


  ## Part 6: Loop vs NumPy

Both versions implement the same onset rule (a sample above the threshold whose previous
sample is not above it, or the first sample if it is above), on the same input (the
drift-removed recording and the 4.5σ threshold).

- **Version A, reference (`detect_onsets`):** a Python `for` loop that checks each sample
  and its predecessor one at a time. It reads like the rule in words, so it serves as the
  readable reference.
- **Version B, NumPy (`detect_onsets_numpy`):** no Python loop. `above = signal > threshold`
  compares every sample at once; `above[1:] & ~above[:-1]` compares each sample with the one
  before it by lining up two shifted copies of the array; the first sample is handled
  separately; `np.flatnonzero` returns the onset indices.

Run with `python benchmark.py` from the `Task2B` folder.

### Evaluation 1: Correctness

- All five onset tests run