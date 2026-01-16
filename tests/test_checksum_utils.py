import hashlib

import numpy as np
import pytest
import soundfile as sf

from src.checksum_utils import calculate_frame_checksums, compare_checksums


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary 24-bit PCM WAV file for verification tests."""
    path = tmp_path / "test.wav"
    sr = 48000
    # Generate stochastic noise to ensure data variety.
    data = np.random.uniform(-1, 1, (sr, 2)).astype(np.float64)
    sf.write(str(path), data, sr, subtype="PCM_24")
    return path, data, sr


def test_checksum_bit_identity(temp_audio_file):
    """
    Verify that file data yields identical checksums to its on-disk
    representation.
    """
    path, _, _ = temp_audio_file

    # Load data back from disk to account for PCM_24 quantization.
    # This ensures bitwise identity with the on-disk format.
    data_from_disk, _ = sf.read(str(path), dtype="float64")

    # Compute checksums via utility.
    checksums = calculate_frame_checksums(path)

    # Manually compute expected hex for the first block (4096 frames).
    expected_data = data_from_disk[:4096]
    m = hashlib.md5()
    m.update(np.ascontiguousarray(expected_data).tobytes())
    expected_hex = m.hexdigest()

    assert checksums[0] == expected_hex


def test_checksum_mismatch(temp_audio_file):
    """Verify that modified audio data yields distinct checksums."""
    path, data, sr = temp_audio_file

    # Alter a single sample and save as a new file.
    path2 = path.parent / "test2.wav"
    data2 = data.copy()
    data2[0, 0] += 0.01  # Significant enough to change quantized value.
    sf.write(str(path2), data2, sr, subtype="PCM_24")

    c1 = calculate_frame_checksums(path)
    c2 = calculate_frame_checksums(path2)

    # Initial blocks must differ due to the injected change.
    assert c1[0] != c2[0]

    # Overall similarity should be strictly less than unity.
    similarity = compare_checksums(c1, c2)
    assert similarity < 1.0


def test_compare_checksums_identical():
    """Verify 100% match for structurally identical checksum lists."""
    c = ["a", "b", "c"]
    assert compare_checksums(c, c) == 1.0


def test_compare_checksums_empty():
    """Verify 0% similarity score when one or both lists are empty."""
    assert compare_checksums([], ["a"]) == 0.0
    assert compare_checksums(["a"], []) == 0.0
