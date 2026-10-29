import numpy as np
import pytest
from spike_detection import (detect_onsets, detect_onsets_numpy,
                             remove_drift, estimate_noise, apply_dead_time)

# Both versions of the onset detector; each onset test runs on both.
ONSET_VERSIONS = [detect_onsets, detect_onsets_numpy]


# --- Onset detection: run on both the loop and the NumPy version ---

@pytest.mark.parametrize("detect", ONSET_VERSIONS)
def test_no_samples_above_threshold_gives_zero_events(detect):
    signal = np.array([0.0, 0.5, 1.0, 0.2])
    events = detect(signal, threshold=1.5)
    assert len(events) == 0


@pytest.mark.parametrize("detect", ONSET_VERSIONS)
def test_one_isolated_sample_gives_one_event(detect):
    signal = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    events = detect(signal, threshold=1.5)
    assert len(events) == 1
    assert list(events) == [2]


@pytest.mark.parametrize("detect", ONSET_VERSIONS)
def test_consecutive_samples_above_count_as_one_event(detect):
    signal = np.array([0.0, 3.0, 4.0, 2.0, 0.0, 0.0])
    events = detect(signal, threshold=1.5)
    assert len(events) == 1
    assert list(events) == [1]


@pytest.mark.parametrize("detect", ONSET_VERSIONS)
def test_two_separate_intervals_give_two_events(detect):
    signal = np.array([0.0, 3.0, 3.0, 0.0, 0.0, 4.0, 0.0])
    events = detect(signal, threshold=1.5)
    assert len(events) == 2
    assert list(events) == [1, 5]


@pytest.mark.parametrize("detect", ONSET_VERSIONS)
def test_signal_starting_above_threshold_detects_event_at_start(detect):
    signal = np.array([3.0, 3.0, 0.0, 0.0, 0.0])
    events = detect(signal, threshold=1.5)
    assert len(events) == 1
    assert list(events) == [0]


# --- The other detector steps: one version each ---

def test_dead_time_drops_onsets_closer_than_dead_time():
    onsets = np.array([10, 13, 30])           # 13 is only 3 ms after 10
    kept = apply_dead_time(onsets, fs=1000, dead_time_s=0.005)
    assert list(kept) == [10, 30]


def test_dead_time_keeps_onsets_farther_apart():
    onsets = np.array([10, 20])               # 10 ms apart
    kept = apply_dead_time(onsets, fs=1000, dead_time_s=0.005)
    assert list(kept) == [10, 20]


def test_remove_drift_removes_constant_offset():
    signal = np.full(1000, 5.0)               # flat line at 5: all "drift", no spikes
    flat = remove_drift(signal, fs=1000)
    assert np.allclose(flat, 0.0)


def test_estimate_noise_recovers_known_sd():
    rng = np.random.default_rng(0)            # seeded, so the same "random" numbers every run
    noise = rng.normal(0, 1, 100_000)         # pure noise with SD = 1
    assert abs(estimate_noise(noise) - 1.0) < 0.02