"""Tests for CLI argument parsing and main entry point."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_folder_path_required(self):
        """Test that folder path is a required argument."""
        from cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([])

    def test_folder_path_accepted(self, tmp_path):
        """Test that folder path is accepted as positional argument."""
        from cli import parse_args

        args = parse_args([str(tmp_path)])
        assert args.folder_path == str(tmp_path)

    def test_prompt_optional(self, tmp_path):
        """Test that prompt is optional."""
        from cli import parse_args

        args = parse_args([str(tmp_path)])
        assert args.prompt is None

    def test_prompt_accepted(self, tmp_path):
        """Test that prompt can be provided."""
        from cli import parse_args

        args = parse_args([str(tmp_path), "--prompt", "detect cars"])
        assert args.prompt == "detect cars"

    def test_model_variant_default(self, tmp_path):
        """Test default model variant is 'n' (nano)."""
        from cli import parse_args

        args = parse_args([str(tmp_path)])
        assert args.variant == 'n'

    def test_model_variant_choices(self, tmp_path):
        """Test model variant choices: n, s, m."""
        from cli import parse_args

        for variant in ['n', 's', 'm']:
            args = parse_args([str(tmp_path), "--variant", variant])
            assert args.variant == variant

    def test_invalid_variant_rejected(self, tmp_path):
        """Test invalid variant is rejected."""
        from cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([str(tmp_path), "--variant", "x"])

    def test_output_dir_optional(self, tmp_path):
        """Test output directory is optional with default."""
        from cli import parse_args

        args = parse_args([str(tmp_path)])
        assert args.output_dir == "output"

    def test_output_dir_custom(self, tmp_path):
        """Test custom output directory."""
        from cli import parse_args

        args = parse_args([str(tmp_path), "--output-dir", "my_results"])
        assert args.output_dir == "my_results"

    def test_confidence_threshold_default(self, tmp_path):
        """Test default confidence threshold."""
        from cli import parse_args

        args = parse_args([str(tmp_path)])
        assert args.confidence == 0.5

    def test_confidence_threshold_custom(self, tmp_path):
        """Test custom confidence threshold."""
        from cli import parse_args

        args = parse_args([str(tmp_path), "--confidence", "0.7"])
        assert args.confidence == 0.7

    def test_finetune_epochs_default(self, tmp_path):
        """Test default fine-tuning epochs."""
        from cli import parse_args

        args = parse_args([str(tmp_path)])
        assert args.epochs == 10

    def test_finetune_epochs_custom(self, tmp_path):
        """Test custom fine-tuning epochs."""
        from cli import parse_args

        args = parse_args([str(tmp_path), "--epochs", "50"])
        assert args.epochs == 50

    def test_save_model_flag(self, tmp_path):
        """Test save model flag."""
        from cli import parse_args

        args = parse_args([str(tmp_path)])
        assert args.save_model is False

        args = parse_args([str(tmp_path), "--save-model"])
        assert args.save_model is True

    def test_model_output_path(self, tmp_path):
        """Test model output path for saving fine-tuned model."""
        from cli import parse_args

        args = parse_args([str(tmp_path), "--model-output", "my_model.pth"])
        assert args.model_output == "my_model.pth"


class TestCLIValidation:
    """Test CLI input validation."""

    def test_folder_must_exist(self, tmp_path):
        """Test that folder path must exist."""
        from cli import validate_args

        class Args:
            folder_path = "/nonexistent/path"
            prompt = None

        with pytest.raises(ValueError, match="Folder path does not exist"):
            validate_args(Args())

    def test_folder_must_be_directory(self, tmp_path):
        """Test that folder path must be a directory."""
        from cli import validate_args

        # Create a file instead of directory
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        class Args:
            folder_path = str(file_path)
            prompt = None

        with pytest.raises(ValueError, match="Path is not a directory"):
            validate_args(Args())

    def test_folder_must_contain_images(self, tmp_path):
        """Test that folder must contain images."""
        from cli import validate_args

        class Args:
            folder_path = str(tmp_path)
            prompt = None

        with pytest.raises(ValueError, match="No images found"):
            validate_args(Args())

    def test_valid_folder_with_images(self, tmp_path):
        """Test validation passes with valid folder containing images."""
        from cli import validate_args

        # Create a dummy image file
        (tmp_path / "test.jpg").write_bytes(b"fake image")

        class Args:
            folder_path = str(tmp_path)
            prompt = None
            confidence = 0.5
            epochs = 10
            variant = 'n'
            output_dir = "output"
            save_model = False
            model_output = None

        # Should not raise
        validate_args(Args())


class TestCLIMain:
    """Test main CLI entry point."""

    @patch('cli.run_detection')
    def test_main_without_prompt_calls_detection(self, mock_detection, tmp_path):
        """Test main function calls detection when no prompt."""
        from cli import main

        # Create a dummy image
        (tmp_path / "test.jpg").write_bytes(b"fake image")

        with patch('sys.argv', ['cli.py', str(tmp_path)]):
            main()

        mock_detection.assert_called_once()

    @patch('cli.run_finetune_pipeline')
    def test_main_with_prompt_calls_finetune(self, mock_finetune, tmp_path):
        """Test main function calls finetune pipeline when prompt provided."""
        from cli import main

        # Create a dummy image
        (tmp_path / "test.jpg").write_bytes(b"fake image")

        with patch('sys.argv', ['cli.py', str(tmp_path), '--prompt', 'detect cars']):
            main()

        mock_finetune.assert_called_once()
