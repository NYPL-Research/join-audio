# Audio Joiner

CLI tool designed to join overlapping audio files (FLAC or WAV) with precision and memory efficiency. Made to rejoin audio files that have been split into multiple parts due to limitations of baseline WAV's maximum file size (2-4 GB). Recommended for use on production, mezzanine, or edit files; not intended for preservation files unless further verification of data integrity has been performed.

## Features

- **Memory Efficient**: Uses streaming I/O to process large files (multi-gigabyte) with minimal RAM footprint (<100MB).
- **Auto-Overlap Detection**: Calculates the optimal join point using cross-correlation between file segments.
- **Strict Quality Control**: Only joins files if the correlation confidence exceeds **0.999**.
- **Volume Normalization**: Automatically calculates gain ratios and attenuates louder segments to match levels and prevent digital clipping.
- **Smooth Crossfading**: Applies a configurable crossfade at the join point to eliminate pops and clicks.
- **Smart Naming**: Automatically suggests output filenames based on common sequence patterns (e.g., `p01`, `p02`).
- **Preview Clips**: Generates a 5-second sample clip centered on the join point for instant verification.
- **Detailed Logging**: Records every join operation, including timestamps and correlation metrics, in detailed `.log` files.
- **Payload Integrity Verification**: Optional frame-level MD5 checksumming to ensure the joined output matches the original inputs.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone <repo_url>
cd join_audio

# Install dependencies
poetry install
```

## Usage

### Basic Join
```bash
poetry run join_parts part1.wav part2.wav
```

### Advanced Usage
```bash
poetry run join_parts part1.flac part2.flac --output joined.flac --format flac --clip-output preview.flac --verify
```

### CLI Arguments
- `file1`, `file2`: Input audio files.
- `--output`, `-o`: Custom output path (defaults to a smart filename).
- `--clip-output`, `-c`: Custom path for the 5-second preview clip. Default is `None` (no clip generated).
- `--format`, `-f`: Output format (`flac` or `wav`).
- `--verify`: Performs frame-level MD5 checksum verification after the join.
- `--match-gain`: Match volume levels between parts (default: True).
- `--no-match-gain`: Disable automatic gain matching.
- `--crossfade`: Crossfade duration in seconds (default: 0.1).

### Quality Assurance
Use `nox` to run the full QA suite (tests, linting, formatting, type checking):

```bash
poetry run nox
```

Individual sessions:
- `poetry run nox -s tests`: Run unit tests with `pytest`.
- `poetry run nox -s mypy`: Run static type checking.
- `poetry run nox -s black`: Check code formatting.
- `poetry run nox -s lint`: Run `flake8` linting.

## Project Structure
- `src/main.py`: CLI entry point and orchestration.
- `src/audio_utils.py`: Core logic for streaming, correlation, and crossfading.
- `src/checksum_utils.py`: MD5 checksumming and verification logic.
- `tests/`: Comprehensive test suite including synthetic audio validation.
