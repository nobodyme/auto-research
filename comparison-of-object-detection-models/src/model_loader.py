"""
Model loader for various object detection models.
Supports YOLOv8, YOLOv10, YOLOv11, RT-DETR, and RF-DETR.
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage object detection models."""

    SUPPORTED_MODELS = {
        # YOLOv8 variants
        'yolov8n': 'yolov8n.pt',
        'yolov8s': 'yolov8s.pt',
        'yolov8m': 'yolov8m.pt',
        'yolov8l': 'yolov8l.pt',
        'yolov8x': 'yolov8x.pt',

        # YOLOv10 variants
        'yolov10n': 'yolov10n.pt',
        'yolov10s': 'yolov10s.pt',
        'yolov10m': 'yolov10m.pt',
        'yolov10l': 'yolov10l.pt',
        'yolov10x': 'yolov10x.pt',

        # YOLOv11 variants
        'yolov11n': 'yolo11n.pt',
        'yolov11s': 'yolo11s.pt',
        'yolov11m': 'yolo11m.pt',
        'yolov11l': 'yolo11l.pt',
        'yolov11x': 'yolo11x.pt',

        # RT-DETR variants
        'rtdetr-l': 'rtdetr-l.pt',
        'rtdetr-x': 'rtdetr-x.pt',
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'object_detection_models'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}

    def load_model(self, model_name: str) -> Any:
        """
        Load a model by name.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model object

        Raises:
            ValueError: If model name is not supported
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Check if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]

        logger.info(f"Loading model: {model_name}")

        try:
            # Try loading with ultralytics (works for YOLO and RT-DETR)
            from ultralytics import YOLO

            model_file = self.SUPPORTED_MODELS[model_name]
            model = YOLO(model_file)

            self.loaded_models[model_name] = model
            logger.info(f"Successfully loaded {model_name}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get information about a loaded model.

        Args:
            model: Loaded model object

        Returns:
            Dictionary containing model information
        """
        info = {}

        try:
            # For YOLO/ultralytics models
            if hasattr(model, 'model'):
                # Count parameters
                total_params = sum(p.numel() for p in model.model.parameters())
                info['parameters'] = total_params

                # Estimate model size (parameters * 4 bytes for float32, converted to MB)
                info['model_size_mb'] = (total_params * 4) / (1024 * 1024)

                # Get model type
                info['model_type'] = type(model).__name__

                # Check if model is on GPU
                info['device'] = str(next(model.model.parameters()).device)

        except Exception as e:
            logger.warning(f"Could not extract complete model info: {e}")
            info['parameters'] = 0
            info['model_size_mb'] = 0

        return info

    def list_supported_models(self) -> list:
        """Return list of supported model names."""
        return list(self.SUPPORTED_MODELS.keys())

    def get_model_family(self, model_name: str) -> str:
        """
        Get the family/architecture of a model.

        Args:
            model_name: Name of the model

        Returns:
            Model family (e.g., 'yolov8', 'yolov10', 'rtdetr')
        """
        if model_name.startswith('yolov8'):
            return 'yolov8'
        elif model_name.startswith('yolov10'):
            return 'yolov10'
        elif model_name.startswith('yolov11'):
            return 'yolov11'
        elif model_name.startswith('rtdetr'):
            return 'rtdetr'
        else:
            return 'unknown'
