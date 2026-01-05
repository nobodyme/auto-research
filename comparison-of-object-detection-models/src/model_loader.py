"""
Model loader for various object detection models.
Supports YOLOv8, YOLOv10, YOLOv11, RT-DETR, and RF-DETR.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFDETRWrapper:
    """
    Wrapper for RF-DETR models to provide a unified interface compatible with YOLO models.
    """

    def __init__(self, model_variant: str = 'n'):
        """
        Initialize RF-DETR wrapper.

        Args:
            model_variant: Model variant ('n', 's', 'm')
        """
        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
        except ImportError:
            raise ImportError(
                "RF-DETR not installed. Install with: pip install rfdetr"
            )

        self.variant = model_variant.lower()

        # Load appropriate model variant
        if self.variant == 'n':
            self.model = RFDETRNano()
        elif self.variant == 's':
            self.model = RFDETRSmall()
        elif self.variant == 'm':
            self.model = RFDETRMedium()
        else:
            raise ValueError(f"Unsupported RF-DETR variant: {model_variant}")

        # Optimize for inference
        self.model.optimize_for_inference()

        logger.info(f"Loaded RF-DETR-{self.variant.upper()} and optimized for inference")

    def predict(self, source, conf=0.25, verbose=False, device='cpu', **kwargs):
        """
        Perform prediction (compatible with YOLO interface).

        Args:
            source: Image path, numpy array, or PIL image
            conf: Confidence threshold
            verbose: Whether to print verbose output
            device: Device to use (ignored, RF-DETR manages this internally)
            **kwargs: Additional arguments (ignored)

        Returns:
            List of Results objects (YOLO-compatible format)
        """
        # Load image if it's a path
        if isinstance(source, (str, Path)):
            import cv2
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Could not load image: {source}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
            image = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Run inference
        detections = self.model.predict(image, threshold=conf)

        # Convert to YOLO-compatible Results format
        results = [RFDETRResults(detections, image.shape)]

        return results

    def val(self, data='coco.yaml', device='cpu', verbose=False, plots=False, **kwargs):
        """
        Validation method (placeholder for COCO evaluation).
        Note: RF-DETR doesn't have built-in COCO validation like Ultralytics.

        Returns:
            Dummy results object for compatibility
        """
        logger.warning(
            "RF-DETR does not have built-in COCO validation. "
            "Using manual evaluation approach."
        )

        # Return a dummy results object
        class DummyResults:
            def __init__(self):
                self.box = type('obj', (object,), {
                    'map': 0.0,    # Will be filled by manual evaluation
                    'map50': 0.0,
                    'map75': 0.0,
                    'maps': [0.0, 0.0, 0.0]
                })()

        return DummyResults()


class RFDETRResults:
    """
    Results wrapper for RF-DETR to match YOLO Results interface.
    """

    def __init__(self, detections: List[Dict], image_shape: tuple):
        """
        Initialize results.

        Args:
            detections: List of detection dictionaries from RF-DETR
            image_shape: Shape of input image (H, W, C)
        """
        self.detections = detections
        self.image_shape = image_shape

        # Create COCO class names (RF-DETR is trained on COCO)
        self.names = {i: name for i, name in enumerate([
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ])}

        # Parse boxes
        self._parse_boxes()

    def _parse_boxes(self):
        """Parse detection results into YOLO-compatible format."""
        if not self.detections:
            self.boxes = type('obj', (object,), {
                'xyxy': torch.tensor([]),
                'conf': torch.tensor([]),
                'cls': torch.tensor([])
            })()
            return

        xyxy_list = []
        conf_list = []
        cls_list = []

        for det in self.detections:
            # RF-DETR returns: {bbox: [x1, y1, x2, y2], confidence: float, class_id: int}
            xyxy_list.append(det['bbox'])
            conf_list.append(det['confidence'])
            cls_list.append(det['class_id'])

        self.boxes = type('obj', (object,), {
            'xyxy': torch.tensor(xyxy_list) if xyxy_list else torch.tensor([]),
            'conf': torch.tensor(conf_list) if conf_list else torch.tensor([]),
            'cls': torch.tensor(cls_list) if cls_list else torch.tensor([])
        })()

    @property
    def boxes(self):
        """Get boxes attribute."""
        return self._boxes

    @boxes.setter
    def boxes(self, value):
        """Set boxes attribute."""
        self._boxes = value


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

        # RF-DETR variants (Roboflow)
        'rfdetr-n': 'rfdetr',  # Nano: 30.5M params, 2.32ms latency
        'rfdetr-s': 'rfdetr',  # Small: 32.1M params, 3.52ms latency
        'rfdetr-m': 'rfdetr',  # Medium: 33.7M params, 4.52ms latency
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
            # Check if it's an RF-DETR model
            if model_name.startswith('rfdetr'):
                variant = model_name.split('-')[1]  # Extract 'n', 's', or 'm'
                model = RFDETRWrapper(model_variant=variant)
                self.loaded_models[model_name] = model
                logger.info(f"Successfully loaded {model_name}")
                return model

            # Otherwise, try loading with ultralytics (works for YOLO and RT-DETR)
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
            # For RF-DETR models
            if isinstance(model, RFDETRWrapper):
                # RF-DETR parameter counts based on variant
                param_counts = {
                    'n': 30.5e6,  # 30.5M parameters
                    's': 32.1e6,  # 32.1M parameters
                    'm': 33.7e6,  # 33.7M parameters
                }
                total_params = param_counts.get(model.variant, 30.5e6)
                info['parameters'] = int(total_params)
                info['model_size_mb'] = (total_params * 4) / (1024 * 1024)
                info['model_type'] = 'RFDETRWrapper'
                info['device'] = 'cpu'  # RF-DETR manages device internally

            # For YOLO/ultralytics models
            elif hasattr(model, 'model'):
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
            Model family (e.g., 'yolov8', 'yolov10', 'rtdetr', 'rfdetr')
        """
        if model_name.startswith('yolov8'):
            return 'yolov8'
        elif model_name.startswith('yolov10'):
            return 'yolov10'
        elif model_name.startswith('yolov11'):
            return 'yolov11'
        elif model_name.startswith('rtdetr'):
            return 'rtdetr'
        elif model_name.startswith('rfdetr'):
            return 'rfdetr'
        else:
            return 'unknown'
