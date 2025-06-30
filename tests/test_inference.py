from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gan_face_generate.inference import main


@dataclass(slots=True)
class MockArgs:
    """Mocked arguments for testing the inference script.

    Attributes:
        save_file (Path): Path to save the generated output.
        dry_run (bool): If True, performs a dry run without saving output.
    """

    save_file: Path
    dry_run: bool = False


@pytest.mark.integration
@pytest.mark.parametrize("dry_run", [True, False])
@patch("gan_face_generate.inference.parse_args")
def test_inference(mock_args: MagicMock, tmp_path: Path, dry_run: bool) -> None:
    """Integration test for the `main()` function of the inference module.

    This test simulates command-line arguments using a patched `parse_args` function.
    It checks that the `main()` function executes without error for both dry-run and normal modes.

    Args:
        mock_args (MagicMock): Mocked `parse_args` function to simulate CLI input.
        tmp_path (Path): Temporary directory provided by pytest for file output.
        dry_run (bool): Whether to simulate a dry run (no file output).
    """
    mock_args.return_value = MockArgs(
        dry_run=dry_run, save_file=tmp_path / "output.png"
    )
    main()
