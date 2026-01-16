from pathlib import Path
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.signal
import soundfile as sf


def load_audio(file_path: Union[str, Path]) -> Tuple[npt.NDArray[np.float64], int]:
    """Loads an entire audio file into memory.

    Args:
        file_path: Path to the audio file.

    Returns:
        A tuple containing (data, samplerate).
    """
    data, samplerate = sf.read(str(file_path))
    return data, samplerate


def get_audio_metadata(file_path: Union[str, Path]) -> Tuple[int, int, int]:
    """Retrieves basic audio metadata without loading the full file.

    Args:
        file_path: Path to the audio file.

    Returns:
        A tuple containing (samplerate, frames, channels).
    """
    with sf.SoundFile(str(file_path)) as f:
        return f.samplerate, f.frames, f.channels


def load_audio_segment(
    file_path: Union[str, Path], start_frame: int, frames_to_read: int
) -> Tuple[npt.NDArray[np.float64], int]:
    """Loads a specific range of frames from an audio file.

    Args:
        file_path: Path to the audio file.
        start_frame: The frame index to start reading from.
        frames_to_read: Number of frames to read.

    Returns:
        A tuple containing (data, samplerate).
    """
    with sf.SoundFile(str(file_path)) as f:
        f.seek(start_frame)
        data = f.read(frames_to_read)
        return data, f.samplerate


def find_overlap(
    data1: npt.NDArray[np.float64],
    data2: npt.NDArray[np.float64],
    samplerate: int,
    max_overlap_sec: int = 30,
) -> Tuple[int, float, float]:
    """Finds the optimal overlap between two audio segments using cross-correlation.

    Args:
        data1: Audio data at the end of the first file.
        data2: Audio data at the start of the second file.
        samplerate: Sample rate of the audio in Hz.
        max_overlap_sec: Maximum duration in seconds to search for overlap.

    Returns:
        A tuple containing (overlap_samples, max_corr).
            overlap_samples: Best overlap length in units of samples.
            max_corr: Normalized correlation coefficient (0.0 to 1.0).
    """
    # Calculate search window size in samples.
    window_samples = int(max_overlap_sec * samplerate)

    # Define segments for correlation based on tail of first input and head of
    # second input.
    seg1 = data1[-window_samples:] if len(data1) > window_samples else data1
    seg2 = data2[:window_samples] if len(data2) > window_samples else data2

    # Normalize signal amplitudes to mean 0 and standard deviation 1 for correlation
    # stability.
    seg1_norm = seg1 - np.mean(seg1)
    std1 = np.std(seg1_norm)
    if std1 > 0:
        seg1_norm /= std1

    seg2_norm = seg2 - np.mean(seg2)
    std2 = np.std(seg2_norm)
    if std2 > 0:
        seg2_norm /= std2

    # Average signal across channels for multi-channel correlation compatibility.
    if seg1.ndim > 1:
        s1 = np.mean(seg1_norm, axis=1)
        s2 = np.mean(seg2_norm, axis=1)
    else:
        s1 = seg1_norm
        s2 = seg2_norm

    # Perform cross-correlation
    correlation = scipy.signal.correlate(s1, s2, mode="full")
    lags = scipy.signal.correlation_lags(len(s1), len(s2), mode="full")

    # Restrict search to positive lags (overlaps between 0 and window_samples)
    # A negative lag would imply s2 starts BEFORE s1, which is not an overlap.
    valid_indices = lags >= 0
    correlation = correlation[valid_indices]
    lags = lags[valid_indices]

    # Identify lag index corresponding to maximum cross-correlation.
    max_idx = np.argmax(correlation)
    lag = lags[max_idx]

    overlap_samples = len(s1) - lag

    gain_ratio = 1.0
    # Compute Pearson correlation coefficient and gain ratio within the identified
    # temporal overlap.
    if overlap_samples > 0:
        # Constrain overlap duration within segment bounds.
        n = min(len(seg1), len(seg2))
        ov = min(overlap_samples, n)

        # Enforce minimum sample count for statistical significance.
        if ov > 10:
            part1 = seg1[-ov:]
            part2 = seg2[:ov]

            # Calculate Pearson correlation: covariance(X,Y) / (std(X) * std(Y)).
            p1 = part1 - np.mean(part1)
            p2 = part2 - np.mean(part2)

            denom = np.std(p1) * np.std(p2) * len(p1)
            if denom > 0:
                max_corr = np.sum(p1 * p2) / denom
            else:
                max_corr = 0.0

            # Determine gain ratio (In1/In2) for amplitude normalization.
            # Exclude low-amplitude samples to ensure gain ratio stability.
            mask_stable = (np.abs(part1) > 1e-5) & (np.abs(part2) > 1e-5)
            if np.any(mask_stable):
                gain_ratio = float(np.mean(part1[mask_stable] / part2[mask_stable]))

            # Compute mean correlation across all channels for multi-channel inputs.
            if part1.ndim > 1:
                corrs = []
                for ch in range(part1.shape[1]):
                    c1 = part1[:, ch] - np.mean(part1[:, ch])
                    c2 = part2[:, ch] - np.mean(part2[:, ch])
                    denom_c = np.std(c1) * np.std(c2) * len(c1)
                    if denom_c > 0:
                        corrs.append(np.sum(c1 * c2) / denom_c)
                    else:
                        corrs.append(0.0)
                max_corr = np.mean(corrs)
        else:
            max_corr = 0.0
    else:
        max_corr = 0.0

    return int(overlap_samples), float(max_corr), float(gain_ratio)


def stream_join_audio(
    file1: Union[str, Path],
    file2: Union[str, Path],
    output_path: Union[str, Path],
    overlap_samples: int,
    samplerate: int,
    subtype: str = "PCM_24",
    crossfade_duration: float = 0.1,
    block_size: int = 1024 * 1024,
    gain_ratio: float = 1.0,
) -> None:
    """Joins two files using streaming I/O with optional gain matching.

    Args:
        file1: Path to the first audio file.
        file2: Path to the second audio file.
        output_path: Path to save the joined file.
        overlap_samples: Number of samples of overlap to join on.
        samplerate: Sample rate of the input/output audio.
        subtype: Codec subtype to write (e.g., 'PCM_24', 'FLAC').
        crossfade_duration: Duration in seconds for the crossfade at join point.
        block_size: Buffer size in frames for reading/writing.
        gain_ratio: Ratio to apply to file2 (In1/In2) for volume matching.
    """
    # Get metadata
    info1 = sf.info(file1)
    _ = sf.info(file2)

    channels = info1.channels

    # Initialize SoundFile objects for input files.
    f1 = sf.SoundFile(file1)
    f2 = sf.SoundFile(file2)

    # Define output file path.
    # Use RF64 format for WAV outputs to exceed 4GB file size limit.
    output_file = Path(output_path)
    fmt = "RF64" if output_file.suffix.lower() == ".wav" else None

    # Determine gain factors for amplitude matching.
    # If Input 1 is louder than Input 2 (gain_ratio > 1.0), attenuate Input 1
    # to match Input 2 level and prevent clipping.

    # Define tolerance for unity gain matching (1e-4 corresponds to ~0.0008 dB).
    # If signal levels are equivalent within tolerance, bypass gain scaling
    # to maintain bit-perfect integrity of source data.
    if abs(1.0 - gain_ratio) < 1e-4:
        g1, g2 = 1.0, 1.0
        # Maintain original signal levels.
    elif gain_ratio > 1.0:
        # In1 is louder. Attenuate In1 to match In2
        g1 = 1.0 / gain_ratio
        g2 = 1.0
        print(
            f"DEBUG: Input 1 is louder. Attenuating Input 1 by {g1:.4f} "
            f"to match Input 2."
        )
    else:
        # In2 is louder. Boost In2 (g2 = gain_ratio) to match In1
        g1 = 1.0
        g2 = gain_ratio
        print(
            f"DEBUG: Input 2 is louder. Boosting Input 2 by {g2:.4f} to match Input 1."
        )

    with sf.SoundFile(
        output_path,
        "w",
        samplerate=samplerate,
        channels=channels,
        subtype=subtype,
        format=fmt,
    ) as out_f:

        # 1. Write the initial non-overlapping segment of the first input file.
        frames_to_write_f1 = info1.frames - overlap_samples

        current_frame = 0
        while current_frame < frames_to_write_f1:
            read_len = min(block_size, frames_to_write_f1 - current_frame)
            data = f1.read(read_len)
            out_f.write(data * g1)
            current_frame += read_len

        # 2. Process the overlap region using a linear crossfade transition.
        fade_samples = int(crossfade_duration * samplerate)
        fade_samples = min(fade_samples, overlap_samples)

        # Load overlapping segments from both input files.
        data1_overlap = f1.read(overlap_samples)
        data2_overlap = f2.read(overlap_samples)

        # Initialize crossfade ramps at the start of the overlap region.
        actual_fade = fade_samples

        ramp_down = np.linspace(1, 0, actual_fade)
        ramp_up = np.linspace(0, 1, actual_fade)

        if channels > 1:
            ramp_down = ramp_down[:, np.newaxis]
            ramp_up = ramp_up[:, np.newaxis]

        # Apply gain scaling and linear crossfade to the transition segment.
        faded = (data1_overlap[:actual_fade] * g1 * ramp_down) + (
            data2_overlap[:actual_fade] * g2 * ramp_up
        )

        # Write the crossfaded transition to the output file.
        out_f.write(faded)

        # Write the remaining overlap segment from the second input file.
        # Ensure consistent gain application for the second input.
        out_f.write(data2_overlap[actual_fade:] * g2)

        # 3. Write the remaining segment of the second input file.
        while True:
            data = f2.read(block_size)
            if len(data) == 0:
                break
            out_f.write(data * g2)

    f1.close()
    f2.close()


def extract_clip(
    joined_data: npt.NDArray[np.float64],
    join_point_idx: int,
    samplerate: int,
    duration: float = 5.0,
) -> npt.NDArray[np.float64]:
    """Extracts a focused audio segment from in-memory audio data.

    Args:
        joined_data: The full audio data to extract from.
        join_point_idx: The sample index to center the clip on.
        samplerate: Sample rate of the audio data.
        duration: Targeted total duration of the clip in seconds.

    Returns:
        The extracted clip as a NumPy array.
    """
    half_samples = int((duration / 2) * samplerate)
    start = max(0, join_point_idx - half_samples)
    end = min(len(joined_data), join_point_idx + half_samples)
    return joined_data[start:end]


def extract_clip_from_files(
    file1: Union[str, Path],
    file2: Union[str, Path],
    overlap_samples: int,
    output_path: Union[str, Path],
    samplerate: int,
    duration: float = 5.0,
    subtype: str = "PCM_24",
) -> None:
    """Generates a preview clip without loading full files into memory.

    Args:
        file1: Path to the first audio file.
        file2: Path to the second audio file.
        overlap_samples: Number of samples of overlap to join on.
        output_path: Path where the clip will be saved.
        samplerate: Sample rate of the audio files.
        duration: Targeted total duration of the clip in seconds.
        subtype: Codec subtype to write (e.g., 'PCM_24').
    """
    half_samples = int((duration / 2) * samplerate)

    info1 = sf.info(file1)

    # Determine frame indices for clip extraction.
    # We need:
    # 1. End of File 1 (just before overlap starts) to serve as pre-join context
    # 2. The Overlap Region (mixed)
    # 3. Start of File 2 (after overlap) to serve as post-join context

    # Identify transition point at the beginning of the overlap sequence.

    f1_end_unique = info1.frames - overlap_samples

    # Calculate File 1 segment bounds (pre-join context plus overlap region).
    f1_seg_start = max(0, f1_end_unique - half_samples)
    f1_seg_len = (f1_end_unique - f1_seg_start) + overlap_samples

    data1_seg, _ = load_audio_segment(file1, f1_seg_start, f1_seg_len)

    # Calculate File 2 segment bounds (overlap region plus post-join context).
    f2_seg_len = overlap_samples + half_samples
    data2_seg, _ = load_audio_segment(file2, 0, f2_seg_len)

    # Execute audio join operation in memory.
    joined_clip = stream_join_audio_memory(
        data1_seg, data2_seg, overlap_samples, samplerate
    )

    # Extract the temporal center of the joined segment.
    # Map join point to the terminal index of unique first input data.
    join_idx_in_clip = f1_end_unique - f1_seg_start

    final_clip = extract_clip(joined_clip, join_idx_in_clip, samplerate, duration)

    sf.write(output_path, final_clip, samplerate, subtype=subtype)


def stream_join_audio_memory(
    data1: npt.NDArray[np.float64],
    data2: npt.NDArray[np.float64],
    overlap_samples: int,
    samplerate: int,
    crossfade_duration: float = 0.1,
) -> npt.NDArray[np.float64]:
    """In-memory shortcut for joining audio segments.

    Args:
        data1: Final audio segment from the first part.
        data2: Initial audio segment from the second part.
        overlap_samples: Overlap amount in samples.
        samplerate: Sample rate for crossfade calculation.
        crossfade_duration: Crossfade ramp length in seconds.

    Returns:
        The joined audio data as a NumPy array.
    """
    if overlap_samples <= 0:
        return np.concatenate((data1, data2))

    fade_samples = int(crossfade_duration * samplerate)
    fade_samples = min(fade_samples, overlap_samples)

    total_len = len(data1) + len(data2) - overlap_samples
    output: npt.NDArray[np.float64]
    if data1.ndim == 2:
        output = np.zeros((total_len, data1.shape[1]), dtype=data1.dtype)
    else:
        output = np.zeros(total_len, dtype=data1.dtype)

    cut_idx = len(data1) - overlap_samples
    end_msg_idx = cut_idx
    output[:end_msg_idx] = data1[:end_msg_idx]

    actual_fade = min(overlap_samples, fade_samples)

    # Apply linear crossfade to overlapping segments.
    ramp_down = np.linspace(1, 0, actual_fade)
    ramp_up = np.linspace(0, 1, actual_fade)

    if data1.ndim == 2:
        ramp_down = ramp_down[:, np.newaxis]
        ramp_up = ramp_up[:, np.newaxis]

    faded = (data1[cut_idx : cut_idx + actual_fade] * ramp_down) + (
        data2[0:actual_fade] * ramp_up
    )

    output[end_msg_idx : end_msg_idx + actual_fade] = faded
    output[end_msg_idx + actual_fade :] = data2[actual_fade:]

    return output
