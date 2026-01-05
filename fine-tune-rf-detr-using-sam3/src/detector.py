"""RF-DETR Object Detection Module.

This module provides a unified interface for RF-DETR object detection
with support for all model variants (Nano, Small, Medium).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# COCO class names for visualization
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def get_available_variants() -> Dict[str, str]:
    """Get available RF-DETR model variants.

    Returns:
        Dictionary mapping variant code to full name.
    """
    return {
        'n': 'RF-DETR Nano',
        's': 'RF-DETR Small',
        'm': 'RF-DETR Medium'
    }


def save_results_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save detection results to JSON file.

    Args:
        results: List of detection results.
        output_path: Path to save JSON file.
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for result in results:
        serializable = {
            'image_path': result['image_path'],
            'detections': []
        }
        for det in result['detections']:
            serializable['detections'].append({
                'bbox': [float(x) for x in det['bbox']],
                'confidence': float(det['confidence']),
                'class_id': int(det['class_id']),
                'class_name': det.get('class_name', 'unknown')
            })
        serializable_results.append(serializable)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


class RFDETRDetector:
    """RF-DETR Object Detector with unified interface.

    Supports Nano, Small, and Medium variants of RF-DETR.
    """

    def __init__(
        self,
        variant: str = 'n',
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """Initialize RF-DETR detector.

        Args:
            variant: Model variant ('n', 's', or 'm').
            confidence_threshold: Minimum confidence for detections.
            device: Device to run model on (None for auto-detect).

        Raises:
            ValueError: If invalid variant specified.
        """
        if variant not in ['n', 's', 'm']:
            raise ValueError(f"Invalid variant: {variant}. Must be 'n', 's', or 'm'.")

        self.variant = variant
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = self._load_model()

        logger.info(f"Loaded RF-DETR {get_available_variants()[variant]}")

    def _load_model(self) -> Any:
        """Load RF-DETR model.

        Returns:
            Loaded RF-DETR model.
        """
        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium

            if self.variant == 'n':
                model = RFDETRNano()
            elif self.variant == 's':
                model = RFDETRSmall()
            else:  # 'm'
                model = RFDETRMedium()

            # Optimize for inference
            if hasattr(model, 'optimize_for_inference'):
                model.optimize_for_inference()

            return model

        except ImportError:
            logger.warning("RF-DETR not installed. Using mock model for testing.")
            return self._create_mock_model()

    def _create_mock_model(self) -> Any:
        """Create mock model for testing when RF-DETR is not installed."""
        class MockModel:
            def predict(self, image_path, **kwargs):
                class MockDetection:
                    xyxy = np.array([[100, 100, 200, 200]])
                    confidence = np.array([0.9])
                    class_id = np.array([0])
                return MockDetection()

            def optimize_for_inference(self):
                pass

            def save(self, path):
                pass

        return MockModel()

    def detect(self, image_path: str) -> Dict[str, Any]:
        """Run detection on a single image.

        Args:
            image_path: Path to image file.

        Returns:
            Dictionary with image_path and detections list.
        """
        logger.debug(f"Detecting objects in: {image_path}")

        # Run inference
        result = self.model.predict(image_path, conf=self.confidence_threshold)

        # Parse results
        detections = []

        if hasattr(result, 'xyxy'):
            # Direct RF-DETR output
            for i in range(len(result.xyxy)):
                conf = float(result.confidence[i])
                if conf >= self.confidence_threshold:
                    class_id = int(result.class_id[i])
                    detections.append({
                        'bbox': result.xyxy[i].tolist(),
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': self._get_class_name(class_id)
                    })
        else:
            # Handle alternative output formats
            if hasattr(result, '__iter__'):
                for det in result:
                    if hasattr(det, 'boxes'):
                        for box in det.boxes:
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            if conf >= self.confidence_threshold:
                                class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                                detections.append({
                                    'bbox': box.xyxy[0].tolist(),
                                    'confidence': conf,
                                    'class_id': class_id,
                                    'class_name': self._get_class_name(class_id)
                                })

        return {
            'image_path': image_path,
            'detections': detections
        }

    def detect_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Run detection on all images in a folder.

        Args:
            folder_path: Path to folder containing images.

        Returns:
            List of detection results for each image.
        """
        folder = Path(folder_path)
        results = []

        # Get all image files
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))

        image_files = sorted(set(image_files))

        logger.info(f"Processing {len(image_files)} images...")

        for image_path in image_files:
            try:
                result = self.detect(str(image_path))
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")

        return results

    def annotate_image(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        output_path: str,
        color: tuple = (0, 255, 0),
        thickness: int = 2
    ) -> None:
        """Annotate image with bounding boxes.

        Args:
            image_path: Path to input image.
            detections: List of detection dictionaries.
            output_path: Path to save annotated image.
            color: BGR color for boxes.
            thickness: Line thickness.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return

        # Draw bounding boxes
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det.get('class_name', 'obj')}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 10, label_size[1] + 10)

            cv2.putText(
                image, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Save annotated image
        cv2.imwrite(output_path, image)
        logger.debug(f"Saved annotated image: {output_path}")

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID.

        Args:
            class_id: COCO class ID.

        Returns:
            Class name string.
        """
        if 0 <= class_id < len(COCO_CLASSES):
            return COCO_CLASSES[class_id]
        return f'class_{class_id}'

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information.
        """
        variants_info = {
            'n': {'name': 'RF-DETR Nano', 'parameters': '30.5M', 'latency': '2.32ms'},
            's': {'name': 'RF-DETR Small', 'parameters': '32.1M', 'latency': '3.52ms'},
            'm': {'name': 'RF-DETR Medium', 'parameters': '33.7M', 'latency': '4.52ms'}
        }

        info = variants_info[self.variant].copy()
        info['variant'] = self.variant
        info['confidence_threshold'] = self.confidence_threshold

        # Try to get actual parameter count
        try:
            if hasattr(self.model, 'parameters'):
                total_params = sum(p.numel() for p in self.model.parameters())
                info['parameters'] = f'{total_params / 1e6:.1f}M'
        except Exception:
            pass

        return info


class RFDETRFineTuneable(RFDETRDetector):
    """RF-DETR detector with fine-tuning support.

    Extends RFDETRDetector with methods for fine-tuning on custom data.
    """

    def __init__(
        self,
        variant: str = 'n',
        confidence_threshold: float = 0.5,
        num_classes: int = 1,
        device: Optional[str] = None
    ):
        """Initialize fine-tuneable RF-DETR detector.

        Args:
            variant: Model variant.
            confidence_threshold: Confidence threshold.
            num_classes: Number of custom classes.
            device: Device to use.
        """
        self.num_classes = num_classes
        super().__init__(variant, confidence_threshold, device)

    def load_pretrained(self, weights_path: str) -> None:
        """Load pre-trained weights.

        Args:
            weights_path: Path to weights file.
        """
        import torch

        if hasattr(self.model, 'load_state_dict'):
            state_dict = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from: {weights_path}")

    def save_weights(self, output_path: str) -> None:
        """Save model weights.

        Args:
            output_path: Path to save weights.
        """
        import torch

        if hasattr(self.model, 'state_dict'):
            torch.save(self.model.state_dict(), output_path)
            logger.info(f"Saved weights to: {output_path}")
        elif hasattr(self.model, 'save'):
            self.model.save(output_path)
            logger.info(f"Saved model to: {output_path}")

    def update_class_names(self, class_names: List[str]) -> None:
        """Update class names for custom dataset.

        Args:
            class_names: List of class names.
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        logger.info(f"Updated class names: {class_names}")

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID.

        Args:
            class_id: Class ID.

        Returns:
            Class name string.
        """
        if hasattr(self, 'class_names') and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return super()._get_class_name(class_id)
