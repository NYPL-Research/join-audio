import numpy as np
import pytest

from src.audio_utils import find_overlap, stream_join_audio_memory


@pytest.fixture
def sr():
    return 48000


@pytest.fixture
def sine_wave(sr):
    # Generate 1.0 second 440 Hz sinusoidal signal.
    t = np.linspace(0, 1, sr)
    return np.sin(2 * np.pi * 440 * t)


def test_find_overlap_perfect(sine_wave, sr):
    """
    Validation of exact overlap detection (0.5s).
    """
    # Generate test signals with exactly 0.5 second temporal overlap.
    # Signal A: [0.0, 0.8]s
    # Signal B: [0.3, 1.0]s (overlap: [0.3, 0.8]s, duration 0.5s).

    # Construct signals: sig1 = [head, overlap], sig2 = [overlap, tail].

    A = np.random.randn(int(0.5 * sr))
    B = np.random.randn(int(0.5 * sr))  # Common overlap segment.
    C = np.random.randn(int(0.5 * sr))

    sig1 = np.concatenate((A, B))
    sig2 = np.concatenate((B, C))

    overlap_samples, corr, _ = find_overlap(sig1, sig2, sr, max_overlap_sec=1.0)

    expected_overlap = len(B)
    assert overlap_samples == expected_overlap
    assert corr > 0.99


def test_find_overlap_none(sr):
    """
    Test distinct signals should have low correlation.
    """
    sig1 = np.random.randn(sr)
    sig2 = np.random.randn(sr)

    # Stochastic signals are statistically unlikely to yield significant correlation.
    overlap_samples, corr, _ = find_overlap(sig1, sig2, sr, max_overlap_sec=0.5)

    # Acknowledge potential for spurious matches; verify correlation coefficient
    # remains below threshold.
    assert corr < 0.3


def test_stream_join_memory_concat(sr):
    """
    Test joining with 0 overlap (concatenation).
    """
    sig1 = np.ones(100)
    sig2 = np.ones(100) * 2

    joined = stream_join_audio_memory(sig1, sig2, 0, sr)

    assert len(joined) == 200
    assert np.all(joined[:100] == 1)
    assert np.all(joined[100:] == 2)


def test_stream_join_memory_crossfade(sr):
    """
    Test output length logic with crossfade/overlap.
    """
    # Define test overlap of 50 samples.
    overlap = 50
    sig1 = np.zeros(100)
    sig2 = np.zeros(100)

    joined = stream_join_audio_memory(sig1, sig2, overlap, sr)

    # Calculated length: length(input1) + length(input2) - overlap.
    expected = 100 + 100 - 50
    assert len(joined) == expected
