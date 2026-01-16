import os

import numpy as np

from src.audio_utils import find_overlap, stream_join_audio_memory


def test_synthetic_sine_wave():
    print("Generating synthetic data...")
    sr = 96000
    duration = 10.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    # Generate linear frequency sweep (chirp) to prevent phase ambiguity during
    # correlation matching.
    from scipy.signal import chirp

    original = chirp(t, f0=100, f1=10000, t1=duration, method="linear")

    # Inject additive white Gaussian noise to simulate realistic signal conditions.
    original += 0.01 * np.random.randn(len(t))

    # Partition signal with 2.0 second temporal overlap.
    # Segment 1: [0.0, 6.0]s
    # Segment 2: [4.0, 10.0]s
    # Overlap: [4.0, 6.0]s (2.0s duration).

    cut1 = int(6 * sr)
    cut2_start = int(4 * sr)

    data1 = original[:cut1]
    data2 = original[cut2_start:]

    print(f"File 1 length: {len(data1)/sr}s")
    print(f"File 2 length: {len(data2)/sr}s")
    print(f"True Overlap: 2.0s ({cut1 - cut2_start} samples)")

    # Infrastructure for automated end-to-end CLI validation.
    os.makedirs("test_outputs", exist_ok=True)
    # sf.write("test_outputs/f1.flac", data1, sr)
    # sf.write("test_outputs/f2.flac", data2, sr)

    # Execute temporal overlap detection validation.
    print("\nTesting overlap detection...")
    overlap_samples, corr, _ = find_overlap(data1, data2, sr)
    overlap_sec = overlap_samples / sr
    print(f"Detected Overlap: {overlap_sec:.4f}s ({overlap_samples} samples)")
    print(f"Correlation: {corr:.4f}")

    msg = f"Overlap detection failed! Got {overlap_sec}"
    assert abs(overlap_sec - 2.0) < 0.001, msg

    # Execute signal concatenation and crossfade validation.
    print("\nTesting joining...")
    joined = stream_join_audio_memory(data1, data2, overlap_samples, samplerate=sr)

    print(f"Original Length: {len(original)}")
    print(f"Joined Length: {len(joined)}")

    # Validate output sample count satisfies: len(in1) + len(in2) - overlap.

    # Verification: 6s (in1) + 6s (in2) - 2s (overlap) = 10s (original).

    assert len(joined) == len(
        original
    ), f"Length mismatch: {len(joined)} vs {len(original)}"

    # Validate content integrity: linear crossfade of identical signals must yield
    # source signal within floating-point precision limits.

    diff = np.max(np.abs(original - joined))
    print(f"Max difference: {diff}")

    assert diff < 1e-5, "Joined audio differs significantly from original!"

    print("\nSUCCESS: Synthetic test passed.")


if __name__ == "__main__":
    test_synthetic_sine_wave()
