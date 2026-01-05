"""SAM3 Integration Module for prompt-based object detection.

This module provides integration with Meta's Segment Anything Model 3 (SAM3)
for text-prompted object detection and segmentation.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def check_sam3_available() -> bool:
    """Check if SAM3 is available.

    Returns:
        True if SAM3 can be imported.
    """
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        return True
    except ImportError:
        return False


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """Convert binary mask to bounding box.

    Args:
        mask: 2D binary mask array.

    Returns:
        Bounding box as [x_min, y_min, x_max, y_max].
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def validate_detections(
    detections: List[Dict[str, Any]],
    filter_empty: bool = False
) -> bool:
    """Validate detection results.

    Args:
        detections: List of detection results.
        filter_empty: Whether to filter out images with no detections.

    Returns:
        True if detections are valid.
    """
    if not detections:
        return False

    for det in detections:
        if 'image_path' not in det:
            return False
        if 'detections' not in det:
            return False

    return True


def prepare_training_data(
    sam3_results: List[Dict[str, Any]],
    output_dir: str,
    class_name: str = "object"
) -> Dict[str, Any]:
    """Prepare training data from SAM3 detection results.

    Args:
        sam3_results: List of SAM3 detection results.
        output_dir: Directory to save training data.
        class_name: Name of the detected class.

    Returns:
        Dictionary with training data paths and info.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / 'images'
    images_dir.mkdir(exist_ok=True)

    # Copy images to training directory
    for result in sam3_results:
        src_path = Path(result['image_path'])
        if src_path.exists():
            dst_path = images_dir / src_path.name
            shutil.copy2(src_path, dst_path)

    # Export annotations to COCO format
    annotations_file = output_path / 'annotations.json'
    export_to_coco_format(sam3_results, str(annotations_file), class_name)

    return {
        'images_dir': str(images_dir),
        'annotations_file': str(annotations_file),
        'num_samples': len(sam3_results),
        'class_name': class_name
    }


def export_to_coco_format(
    sam3_results: List[Dict[str, Any]],
    output_file: str,
    class_name: str = "object"
) -> None:
    """Export SAM3 results to COCO format.

    Args:
        sam3_results: List of SAM3 detection results.
        output_file: Path to save COCO JSON.
        class_name: Name of the detected class.
    """
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': class_name, 'supercategory': 'object'}
        ]
    }

    annotation_id = 1

    for img_id, result in enumerate(sam3_results, start=1):
        image_path = Path(result['image_path'])

        # Try to get image dimensions
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]
            else:
                width, height = 640, 480  # Default
        except Exception:
            width, height = 640, 480

        coco_data['images'].append({
            'id': img_id,
            'file_name': image_path.name,
            'width': width,
            'height': height
        })

        for det in result.get('detections', []):
            bbox = det['bbox']
            # COCO format: [x, y, width, height]
            x_min, y_min, x_max, y_max = bbox
            coco_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': coco_bbox,
                'area': (x_max - x_min) * (y_max - y_min),
                'iscrowd': 0,
                'segmentation': []  # Could add mask RLE here
            })
            annotation_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    logger.info(f"Exported {len(coco_data['images'])} images, "
                f"{len(coco_data['annotations'])} annotations to {output_file}")


class SAM3Detector:
    """SAM3 detector for text-prompted object detection.

    Uses Meta's Segment Anything Model 3 for detecting objects
    based on text prompts.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize SAM3 detector.

        Args:
            device: Device to run model on (None for auto-detect).
        """
        self.device = device
        self.model, self.processor = self._load_model()

    def _load_model(self) -> Tuple[Any, Any]:
        """Load SAM3 model and processor.

        Returns:
            Tuple of (model, processor).
        """
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model()
            processor = Sam3Processor(model)

            logger.info("Loaded SAM3 model")
            return model, processor

        except ImportError:
            logger.warning("SAM3 not installed. Using mock model for testing.")
            return self._create_mock_model()

    def _create_mock_model(self) -> Tuple[Any, Any]:
        """Create mock model for testing."""
        class MockProcessor:
            def set_image(self, image):
                return {'image': image}

            def set_text_prompt(self, state, prompt):
                # Return mock detections
                return {
                    'masks': np.array([[[True, False], [False, True]]]),
                    'boxes': np.array([[100, 100, 200, 200]]),
                    'scores': np.array([0.9])
                }

        class MockModel:
            pass

        return MockModel(), MockProcessor()

    def detect(
        self,
        image_path: str,
        prompt: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Detect objects in image using text prompt.

        Args:
            image_path: Path to image file.
            prompt: Text prompt describing objects to detect.
            confidence_threshold: Minimum confidence threshold.

        Returns:
            Dictionary with detections and masks.
        """
        logger.debug(f"SAM3 detecting '{prompt}' in: {image_path}")

        try:
            from PIL import Image
            image = Image.open(image_path)
        except ImportError:
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image and run inference
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=prompt
        )

        # Parse results
        masks = output.get('masks', np.array([]))
        boxes = output.get('boxes', np.array([]))
        scores = output.get('scores', np.array([]))

        detections = []
        for i in range(len(boxes)):
            score = float(scores[i]) if i < len(scores) else 0.0
            if score >= confidence_threshold:
                detections.append({
                    'bbox': boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i]),
                    'confidence': score,
                    'class_id': 0,
                    'class_name': prompt
                })

        return {
            'image_path': image_path,
            'detections': detections,
            'masks': masks,
            'prompt': prompt
        }

    def detect_folder(
        self,
        folder_path: str,
        prompt: str,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect objects in all images in a folder.

        Args:
            folder_path: Path to folder containing images.
            prompt: Text prompt for detection.
            confidence_threshold: Minimum confidence threshold.

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

        logger.info(f"SAM3 processing {len(image_files)} images with prompt: '{prompt}'")

        for image_path in image_files:
            try:
                result = self.detect(str(image_path), prompt, confidence_threshold)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information.
        """
        return {
            'name': 'SAM3',
            'full_name': 'Segment Anything Model 3',
            'parameters': '848M',
            'capabilities': ['text_prompting', 'segmentation', 'detection']
        }


class SAM3AnnotationGenerator:
    """Generate annotations for training from SAM3 detections.

    This class provides utilities for generating training annotations
    from SAM3 detections that can be used to fine-tune RF-DETR.
    """

    def __init__(self, detector: Optional[SAM3Detector] = None):
        """Initialize annotation generator.

        Args:
            detector: Optional SAM3Detector instance.
        """
        self.detector = detector or SAM3Detector()

    def generate_annotations(
        self,
        folder_path: str,
        prompt: str,
        output_dir: str,
        class_name: str = "object",
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Generate annotations for a folder of images.

        Args:
            folder_path: Path to folder with images.
            prompt: Text prompt for detection.
            output_dir: Directory to save annotations.
            class_name: Class name for annotations.
            confidence_threshold: Minimum confidence.

        Returns:
            Dictionary with annotation statistics.
        """
        # Detect objects using SAM3
        results = self.detector.detect_folder(
            folder_path, prompt, confidence_threshold
        )

        # Prepare training data
        training_data = prepare_training_data(results, output_dir, class_name)

        # Statistics
        total_detections = sum(len(r['detections']) for r in results)
        images_with_detections = sum(1 for r in results if r['detections'])

        return {
            **training_data,
            'total_detections': total_detections,
            'images_with_detections': images_with_detections,
            'images_without_detections': len(results) - images_with_detections
        }

    def save_masks(
        self,
        results: List[Dict[str, Any]],
        output_dir: str
    ) -> None:
        """Save segmentation masks to files.

        Args:
            results: SAM3 detection results with masks.
            output_dir: Directory to save masks.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for result in results:
            if 'masks' in result and result['masks'] is not None:
                image_name = Path(result['image_path']).stem
                masks = result['masks']

                for i, mask in enumerate(masks):
                    mask_path = output_path / f"{image_name}_mask_{i}.png"
                    # Convert boolean mask to uint8
                    mask_img = (mask.astype(np.uint8) * 255)
                    try:
                        import cv2
                        cv2.imwrite(str(mask_path), mask_img)
                    except Exception as e:
                        logger.warning(f"Failed to save mask: {e}")

        logger.info(f"Saved masks to {output_dir}")
