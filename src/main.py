"""CLI tool for joining two overlapping audio files (FLAC/WAV)."""

import argparse
import datetime
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio_utils import (
    extract_clip_from_files,
    find_overlap,
    get_audio_metadata,
    load_audio_segment,
    stream_join_audio,
)
from src.checksum_utils import verify_join_integrity


def main() -> None:
    """Primary entry point for the audio joining CLI.

    Parses command-line arguments, calculates overlap between two audio files,
    performs a streaming join with crossfade, generates a preview clip,
    and logs the transaction.
    """
    parser = argparse.ArgumentParser(description="Join two audio files with overlap.")
    parser.add_argument("file1", help="First audio file")
    parser.add_argument("file2", help="Second audio file")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output joined file path (defaults to smart filename from inputs).",
    )
    parser.add_argument(
        "--clip-output",
        "-c",
        default=None,
        help="Output clip file path (extension optional). Default is None (no clip).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["flac", "wav"],
        default=None,
        help="Output format (flac or wav). Defaults to input file format.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Perform frame-level MD5 checksum verification after join.",
    )

    parser.add_argument(
        "--match-gain",
        action="store_true",
        default=True,
        help="Match volume levels between parts (default: True)",
    )
    parser.add_argument(
        "--no-match-gain",
        action="store_false",
        dest="match_gain",
        help="Disable automatic gain matching",
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=0.1,
        help="Crossfade duration in seconds (default: 0.1).",
    )
    args = parser.parse_args()

    file1_path = Path(args.file1)
    file2_path = Path(args.file2)

    print(f"Loading files:\n  1: {file1_path}\n  2: {file2_path}")

    # Validate existence of input file paths.
    if not file1_path.exists():
        print(f"Error: File '{file1_path}' not found.")
        return
    if not file2_path.exists():
        print(f"Error: File '{file2_path}' not found.")
        return

    # Resolve output audio format.
    if args.format is None:
        # Infer format from first input file extension.
        ext = file1_path.suffix.lower().replace(".", "")
        if ext in ["wav", "flac"]:
            args.format = ext
        else:
            args.format = "flac"  # Default fallback

    print(f"Output Format: {args.format}")

    # Extract audio metadata from input files.
    sr1, frames1, ch1 = get_audio_metadata(str(file1_path))
    sr2, frames2, ch2 = get_audio_metadata(str(file2_path))

    if sr1 != sr2:
        print(f"Warning: Sample rates differ ({sr1} vs {sr2}). Result will use {sr1}.")
    sr = sr1

    print(f"Sample Rate: {sr} Hz")
    print(f"Lengths: {frames1/sr:.2f}s, {frames2/sr:.2f}s")

    # Initialize parameters for cross-correlation overlap detection.
    # Constrain search window to the temporal transition between Input 1 and Input 2.
    max_overlap_sec = 30
    overlap_window = int(max_overlap_sec * sr)

    print("Loading segments for overlap detection...")

    # Load tail segment of the first input file.
    start1 = max(0, frames1 - overlap_window)
    len1 = frames1 - start1
    seg1, _ = load_audio_segment(str(file1_path), start1, len1)

    # Load head segment of the second input file.
    len2 = min(frames2, overlap_window)
    seg2, _ = load_audio_segment(str(file2_path), 0, len2)

    print("Calculating overlap...")
    overlap_samples, max_corr, gain_ratio = find_overlap(
        seg1, seg2, sr, max_overlap_sec=max_overlap_sec
    )
    overlap_sec = overlap_samples / sr
    gain_db = 20 * np.log10(gain_ratio) if gain_ratio > 0 else 0

    print("Match found!")
    print(f"  Overlap Amount: {overlap_sec:.4f} seconds ({overlap_samples} samples)")
    print(f"  Correlation Confidence: {max_corr:.4f}")
    if args.match_gain:
        print(f"  Detected Gain Difference: {gain_db:.2f} dB")

    if max_corr < 0.999:
        print(f"ERROR: Correlation confidence {max_corr:.4f} is too low (min 0.999).")
        print("Aborting join to prevent poor quality results.")
        return

    # Execute audio join operation.
    print("Joining audio (streaming)...")

    # Construct output file path.
    target_ext = f".{args.format.lower()}"

    output_file = Path(args.output) if args.output else None
    if output_file is None:
        # Generate heuristic default filename based on input basenames.
        # Strip sequence identifiers (e.g., 'p01', 'p02') to identify common base names.
        base1 = file1_path.stem
        base2 = file2_path.stem

        pattern = r"p\d+"

        clean1 = re.sub(pattern, "", base1)
        clean2 = re.sub(pattern, "", base2)

        if clean1 == clean2 and clean1 != base1:
            # Use cleaned base name for consistent sequence joining.
            output_file = Path(clean1)
            print(f"Auto-generated output filename: {output_file}")
        else:
            output_file = Path("joined_output")

    # Enforce target file extension.
    if output_file.suffix.lower() != target_ext:
        output_file = output_file.with_suffix(target_ext)

    print(f"Saving joined file to: {output_file}")

    # Revert to unity gain if matching is explicitly disabled.
    active_gain_ratio = gain_ratio if args.match_gain else 1.0

    stream_join_audio(
        str(file1_path),
        str(file2_path),
        str(output_file),
        overlap_samples,
        sr,
        subtype="PCM_24",
        gain_ratio=active_gain_ratio,
        crossfade_duration=args.crossfade,
    )

    # Extract preview clip centered on join point.
    if args.clip_output:
        print("Creating clip...")

        # Define preview clip output path.
        clip_output = Path(args.clip_output)
        if clip_output.suffix.lower() != target_ext:
            clip_output = clip_output.with_suffix(target_ext)

        extract_clip_from_files(
            str(file1_path),
            str(file2_path),
            overlap_samples,
            str(clip_output),
            sr,
            duration=5.0,
            subtype="PCM_24",
        )

        print(f"Saved clip to: {clip_output}")

    # Post-join integrity verification.
    match1, match2, mj1, mj2 = 0.0, 0.0, 0.0, 0.0
    if args.verify:
        print("Verifying integrity (calculating checksums)...")
        match1, match2, mj1, mj2 = verify_join_integrity(
            str(file1_path), str(file2_path), str(output_file), overlap_samples
        )
        print("Integrity Check:")
        print(f"  Part 1 (Input 1): {match1*100:.2f}% match")
        print(f"  Part 2 (Input 2): {match2*100:.2f}% match")
        print(f"  Join Region (vs In 1): {mj1*100:.2f}% match")
        print(f"  Join Region (vs In 2): {mj2*100:.2f}% match")

    # Transaction logging.
    # Collect terminal metadata for log entry.
    info1 = sf.info(str(file1_path))
    info2 = sf.info(str(file2_path))
    info_out = sf.info(str(output_file))

    log_entry = (
        f"Timestamp: {datetime.datetime.now().isoformat()}\n"
        f"Input 1: {file1_path} (Codec: {info1.subtype})\n"
        f"Input 2: {file2_path} (Codec: {info2.subtype})\n"
        f"Search Window: {max_overlap_sec} seconds\n"
        f"Overlap Found: {overlap_samples} samples ({overlap_sec:.4f} seconds)\n"
        f"Correlation Confidence: {max_corr:.4f}\n"
        f"Output: {output_file} (Codec: {info_out.subtype})\n"
    )

    if args.verify:
        log_entry += (
            f"Integrity Part 1: {match1*100:.2f}% match\n"
            f"Integrity Part 2: {match2*100:.2f}% match\n"
            f"Integrity Join (vs In 1): {mj1*100:.2f}% match\n"
            f"Integrity Join (vs In 2): {mj2*100:.2f}% match\n"
        )

    log_entry += f"{'-'*40}\n"

    log_file = output_file.with_name(f"{output_file.stem}_join.log")
    with open(log_file, "a") as f:
        f.write(log_entry)

    print(f"Log updated: {log_file}")
    print("Done.")


if __name__ == "__main__":
    main()
