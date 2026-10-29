import numpy as np
from spike_detection import detect_onsets, remove_drift, estimate_noise, apply_dead_time

def test_no_samples_above_threshold_gives_zero_events():
    signal = np.array([0.0, 0.5, 1.0, 0.2])
    events = detect_onsets(signal, threshold=1.5)
    assert len(events) == 0

def test_one_isolated_sample_gives_one_event():
    signal = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    events = detect_onsets(signal, threshold=1.5)
    assert len(events) == 1
    assert list(events) == [2]

def test_consecutive_samples_above_count_as_one_event():
    signal = np.array([0.0, 3.0, 4.0, 2.0, 0.0, 0.0])
    events = detect_onsets(signal, threshold=1.5)
    assert len(events) == 1
    assert list(events) == [1]

def test_two_separate_intervals_give_two_events():
    signal = np.array([0.0, 3.0, 3.0, 0.0, 0.0, 4.0, 0.0])
    events = detect_onsets(signal, threshold=1.5)
    assert len(events) == 2
    assert list(events) == [1, 5]

def test_signal_starting_above_threshold_detects_event_at_start():
    signal = np.array([3.0, 3.0, 0.0, 0.0, 0.0])
    events = detect_onsets(signal, threshold=1.5)
    assert len(events) == 1
    assert list(events) == [0]

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