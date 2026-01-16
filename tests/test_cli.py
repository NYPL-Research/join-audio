import sys
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from src.main import main


def test_cli_help(capsys):
    """Verify that --help displays comprehensive usage information."""
    with patch.object(sys, "argv", ["join_parts", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0

    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "--match-gain" in captured.out
    assert "--no-match-gain" in captured.out


def test_cli_missing_files(capsys):
    """Verify error reporting when input file paths do not exist."""
    with patch.object(
        sys, "argv", ["join_parts", "nonexistent1.wav", "nonexistent2.wav"]
    ):
        main()

    captured = capsys.readouterr()
    assert "Error: File 'nonexistent1.wav' not found." in captured.out


@patch("src.main.find_overlap")
def test_cli_low_correlation_abort(mock_overlap, capsys, tmp_path):
    """Verify that the tool aborts when correlation confidence falls below threshold."""
    f1 = tmp_path / "f1.wav"
    f2 = tmp_path / "f2.wav"
    sr = 48000
    # Create valid but minimal audio files to satisfy library requirements.
    dummy_data = np.zeros((1000, 2))
    sf.write(str(f1), dummy_data, sr, subtype="PCM_24")
    sf.write(str(f2), dummy_data, sr, subtype="PCM_24")

    # Inject low correlation coefficient via mock.
    mock_overlap.return_value = (100, 0.5, 1.0)

    with patch("src.main.get_audio_metadata", return_value=(sr, 1000, 2)):
        with patch.object(sys, "argv", ["join_parts", str(f1), str(f2)]):
            main()

    captured = capsys.readouterr()
    assert "ERROR: Correlation confidence 0.5000 is too low" in captured.out
    assert "Aborting join" in captured.out
