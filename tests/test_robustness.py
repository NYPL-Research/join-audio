import numpy as np
import pytest
import soundfile as sf

from src.audio_utils import find_overlap, stream_join_audio
from src.checksum_utils import calculate_frame_checksums, compare_checksums


@pytest.fixture
def sr():
    return 48000


@pytest.fixture
def tmp_audio_pair(tmp_path, sr):
    """Creates a pair of overlapping files for join testing."""
    f1 = tmp_path / "part1.wav"
    f2 = tmp_path / "part2.wav"

    # 2 seconds total, 1 second overlap
    # Part 1: [0, 1.5]
    # Part 2: [0.5, 2.0] -> Overlap is [0.5, 1.5]
    t = np.linspace(0, 2, 2 * sr)
    data = np.sin(2 * np.pi * 440 * t)

    sf.write(str(f1), data[: int(1.5 * sr)], sr, subtype="PCM_24")
    sf.write(str(f2), data[int(0.5 * sr) :], sr, subtype="PCM_24")

    return f1, f2, int(1.0 * sr)


def test_clipping_prevention_attenuate_in1(tmp_path, tmp_audio_pair, sr):
    """Verifies that if In1 is louder, it is attenuated (clipping prevention)."""
    f1, f2, overlap = tmp_audio_pair
    out = tmp_path / "joined.wav"

    # Make f1 louder than f2
    data1, _ = sf.read(str(f1))
    sf.write(str(f1), data1 * 1.5, sr, subtype="PCM_24")

    # gain_ratio (In1/In2) will be ~1.5
    # Tool should attenuate In1 by 1/1.5 (~0.66)
    stream_join_audio(str(f1), str(f2), str(out), overlap, sr, gain_ratio=1.5)

    out_data, _ = sf.read(str(out))
    # Max value should not exceed ~1.0 (since original sine was 1.0)
    assert np.max(np.abs(out_data)) <= 1.01


def test_clipping_prevention_boost_in2(tmp_path, tmp_audio_pair, sr):
    """Verifies that if In2 is louder, it is boosted (safe matching)."""
    f1, f2, overlap = tmp_audio_pair
    out = tmp_path / "joined.wav"

    # Make f2 louder than f1 (ratio In1/In2 = 0.5)
    data1, _ = sf.read(str(f1))
    sf.write(str(f1), data1 * 0.5, sr, subtype="PCM_24")

    # Tool should boost In2 by 0.5 (scaling it down to match In1)
    stream_join_audio(str(f1), str(f2), str(out), overlap, sr, gain_ratio=0.5)

    out_data, _ = sf.read(str(out))
    assert np.max(np.abs(out_data)) <= 0.55


def test_overlap_zero_variance_silence(sr):
    """Verifies that silent files don't cause division by zero in correlation."""
    data1 = np.zeros(sr)
    data2 = np.zeros(sr)

    # Should return 0 overlap and 0 correlation, not crash
    samples, corr, ratio = find_overlap(data1, data2, sr)
    assert corr == 0.0
    assert ratio == 1.0


def test_crossfade_duration_limit(tmp_path, tmp_audio_pair, sr):
    """Verifies that crossfade duration is capped by overlap size."""
    f1, f2, overlap = tmp_audio_pair
    out = tmp_path / "joined.wav"

    # Request 10s crossfade on 1s overlap
    stream_join_audio(str(f1), str(f2), str(out), overlap, sr, crossfade_duration=10.0)

    # Verification is mostly that it doesn't crash and output length is correct
    assert out.exists()
    info = sf.info(str(out))
    # Length should be L1 + L2 - Overlap
    assert info.frames == int(1.5 * sr) + int(1.5 * sr) - overlap


def test_bit_identity_integrity(tmp_path, sr):
    """Confirm checksums yield 100% matches for bit-identical data."""
    data = np.random.uniform(-1, 1, sr)
    f1 = tmp_path / "f1.wav"
    f2 = tmp_path / "f2.wav"

    sf.write(str(f1), data, sr, subtype="PCM_24")
    sf.write(str(f2), data, sr, subtype="PCM_24")

    c1 = calculate_frame_checksums(f1)
    c2 = calculate_frame_checksums(f2)

    assert compare_checksums(c1, c2) == 1.0


def test_channel_mismatch(tmp_path, sr):
    """Verify exception is raised when joining mono and stereo files."""
    mono = np.random.uniform(-1, 1, sr)
    stereo = np.random.uniform(-1, 1, (sr, 2))

    f1 = tmp_path / "mono.wav"
    f2 = tmp_path / "stereo.wav"
    out = tmp_path / "out.wav"

    sf.write(str(f1), mono, sr, subtype="PCM_24")
    sf.write(str(f2), stereo, sr, subtype="PCM_24")

    # Multiplication of mono data by stereo gain or vice versa during stream join
    # should trigger a broadcasting error or soundfile mismatch.
    with pytest.raises(Exception):
        stream_join_audio(str(f1), str(f2), str(out), 100, sr)


def test_correlation_threshold_cli(tmp_path, capsys):
    """Verify CLI aborts operation when correlation confidence is below 0.999."""
    import sys
    from unittest.mock import patch

    from src.main import main

    sr = 44100
    # Use uncorrelated noise to ensure low correlation.
    data1 = np.random.uniform(-1, 1, sr * 2)
    data2 = np.random.uniform(-1, 1, sr * 2)

    f1 = tmp_path / "f1.wav"
    f2 = tmp_path / "f2.wav"
    sf.write(str(f1), data1, sr, subtype="PCM_24")
    sf.write(str(f2), data2, sr, subtype="PCM_24")

    test_args = ["prog", str(f1), str(f2), "-o", str(tmp_path / "out.wav")]

    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    assert "ERROR: Correlation confidence" in captured.out
    assert "Aborting join" in captured.out


def test_no_match_gain_cli(tmp_path, sr, capsys):
    """Verify --no-match-gain preserves source amplitudes in output."""
    import sys
    from unittest.mock import patch

    from src.main import main

    # Use distinct amplitudes: In1 = 0.8, In2 = 0.4.
    data1 = np.random.uniform(-1, 1, sr) * 0.8
    data2 = np.random.uniform(-1, 1, sr) * 0.4
    # Create perfect overlap region for detection.
    data2[: sr // 2] = data1[sr // 2 :]

    f1 = tmp_path / "f1.wav"
    f2 = tmp_path / "f2.wav"
    out = tmp_path / "out_no_match.wav"
    sf.write(str(f1), data1, sr, subtype="PCM_24")
    sf.write(str(f2), data2, sr, subtype="PCM_24")

    test_args = ["prog", str(f1), str(f2), "-o", str(out), "--no-match-gain"]

    with patch.object(sys, "argv", test_args):
        main()

    out_data, _ = sf.read(str(out))
    # Output segment corresponding to In2 should maintain ~0.4 peak amplitude.
    assert np.max(np.abs(out_data[-sr // 4 :])) == pytest.approx(0.4, rel=1e-2)
