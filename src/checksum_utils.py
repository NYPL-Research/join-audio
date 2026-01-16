import hashlib
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import soundfile as sf


def calculate_frame_checksums(
    file_path: Union[str, Path],
    start_frame: int = 0,
    num_frames: int = -1,
    block_size: int = 4096,
) -> List[str]:
    """Calculates granular MD5 checksums for a specific audio segment.

    Args:
        file_path: Path to the audio file.
        start_frame: The frame index to start reading from.
        num_frames: Total number of frames to process (-1 for all).
        block_size: Number of frames per checksum block.

    Returns:
        A list of MD5 hex strings.
    """
    checksums = []
    with sf.SoundFile(str(file_path)) as f:
        f.seek(start_frame)

        frames_left = num_frames if num_frames >= 0 else f.frames - start_frame

        while frames_left > 0:
            read_len = min(block_size, frames_left)
            # Utilize float64 for consistent bitwise representation during MD5 calculation.  # noqa: E501
            data = f.read(read_len, dtype="float64")
            if len(data) == 0:
                break

            m = hashlib.md5()
            # Maintain C-contiguous array layout for stable hashing results.
            m.update(np.ascontiguousarray(data).tobytes())
            checksums.append(m.hexdigest())
            frames_left -= read_len

    return checksums


def compare_checksums(checksums1: List[str], checksums2: List[str]) -> float:
    """Calculates the similarity percentage between two checksum lists.

    Args:
        checksums1: First list of MD5 hex strings.
        checksums2: Second list of MD5 hex strings.

    Returns:
        The percentage of matching blocks (0.0 to 1.0).
    """
    if not checksums1 or not checksums2:
        return 0.0

    matches = 0
    first_mismatch = -1
    for i, (c1, c2) in enumerate(zip(checksums1, checksums2)):
        if c1 == c2:
            matches += 1
        elif first_mismatch == -1:
            first_mismatch = i

    return matches / max(len(checksums1), len(checksums2))


def verify_join_integrity(
    file1: Union[str, Path],
    file2: Union[str, Path],
    output_file: Union[str, Path],
    overlap_samples: int,
    block_size: int = 4096,
) -> Tuple[float, float, float, float]:
    """Verifies output integrity by comparing bit-perfect and crossfaded segments.

    Args:
        file1: First input file (pre-join).
        file2: Second input file (post-join).
        output_file: Resulting joined file.
        overlap_samples: Samples of overlap.
        block_size: Alignment for verification blocks.

    Returns:
        A tuple of (part1_match, part2_match, join_vs_in1_match, join_vs_in2_match).
    """
    info1 = sf.info(str(file1))
    unique_len1 = max(0, info1.frames - overlap_samples)

    # Verify Part 1: Head of first input versus head of joined output.
    c1_part1 = calculate_frame_checksums(file1, 0, unique_len1, block_size)
    cout_part1 = calculate_frame_checksums(output_file, 0, unique_len1, block_size)
    match1 = compare_checksums(c1_part1, cout_part1)

    # Verify Part 2: Tail of joined output versus tail of second input.
    out_start2 = unique_len1 + overlap_samples
    in2_start2 = overlap_samples
    info2 = sf.info(str(file2))
    frames_part2 = max(0, info2.frames - overlap_samples)

    c2_part2 = calculate_frame_checksums(file2, in2_start2, frames_part2, block_size)
    cout_part2 = calculate_frame_checksums(
        output_file, out_start2, frames_part2, block_size
    )
    match2 = compare_checksums(c2_part2, cout_part2)

    # Validate bit-perfect integrity against both inputs in the transition region.
    if overlap_samples > 0:
        c1_join = calculate_frame_checksums(
            file1, unique_len1, overlap_samples, block_size
        )
        c2_join = calculate_frame_checksums(file2, 0, overlap_samples, block_size)
        cout_join = calculate_frame_checksums(
            output_file, unique_len1, overlap_samples, block_size
        )

        match_join1 = compare_checksums(c1_join, cout_join)
        match_join2 = compare_checksums(c2_join, cout_join)
    else:
        match_join1, match_join2 = 1.0, 1.0

    return float(match1), float(match2), float(match_join1), float(match_join2)
