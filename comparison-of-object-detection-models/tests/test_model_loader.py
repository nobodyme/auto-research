"""
Tests for model loading and initialization.
Following TDD approach - these tests define expected behavior.
"""
import pytest
import torch
from pathlib import Path


class TestModelLoader:
    """Test suite for model loading functionality."""

    def test_load_yolov8_model(self):
        """Test loading YOLOv8 model."""
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader.load_model('yolov8n')

        assert model is not None
        assert hasattr(model, 'predict')

    def test_load_yolov10_model(self):
        """Test loading YOLOv10 model."""
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader.load_model('yolov10n')

        assert model is not None
        assert hasattr(model, 'predict')

    def test_load_yolov11_model(self):
        """Test loading YOLOv11 model."""
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader.load_model('yolov11n')

        assert model is not None
        assert hasattr(model, 'predict')

    def test_load_rtdetr_model(self):
        """Test loading RT-DETR model."""
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader.load_model('rtdetr-l')

        assert model is not None
        assert hasattr(model, 'predict')

    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        with pytest.raises(ValueError):
            loader.load_model('invalid_model_name')

    def test_get_model_info(self):
        """Test retrieval of model information."""
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader.load_model('yolov8n')
        info = loader.get_model_info(model)

        assert 'parameters' in info
        assert 'model_size_mb' in info
        assert isinstance(info['parameters'], (int, float))
        assert isinstance(info['model_size_mb'], (int, float))
