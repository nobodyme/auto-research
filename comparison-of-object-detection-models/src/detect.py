"""
Object detection on images with annotation.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import logging
from tqdm import tqdm

from .model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Perform object detection on images."""

    def __init__(
        self,
        model_name: str = 'yolov8n',
        confidence_threshold: float = 0.25,
        use_gpu: bool = True
    ):
        """
        Initialize object detector.

        Args:
            model_name: Name of the model to use
            confidence_threshold: Minimum confidence for detections
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if use_gpu else 'cpu'

        # Load model
        loader = ModelLoader()
        self.model = loader.load_model(model_name)

        logger.info(f"Initialized ObjectDetector with {model_name}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"Device: {self.device}")

    def detect(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Perform object detection on a single image.

        Args:
            image_path: Path to the image

        Returns:
            Dictionary containing detection results
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run inference
        results = self.model.predict(
            str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )

        # Parse results
        detections = []
        if len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        'bbox': box.tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': result.names[cls_id]
                    })

        return {
            'image_path': str(image_path),
            'detections': detections,
            'num_detections': len(detections)
        }

    def detect_batch(self, image_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Perform object detection on a batch of images.

        Args:
            image_dir: Directory containing images

        Returns:
            List of detection results for each image
        """
        image_dir = Path(image_dir)

        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return []

        logger.info(f"Found {len(image_paths)} images")

        # Process each image
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.detect(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

        return results

    def annotate_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        detections: Optional[List[Dict]] = None
    ) -> None:
        """
        Annotate image with bounding boxes.

        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            detections: Pre-computed detections (if None, will run detection)
        """
        image_path = Path(image_path)
        output_path = Path(output_path)

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Get detections if not provided
        if detections is None:
            result = self.detect(image_path)
            detections = result['detections']

        # Draw bounding boxes
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']

            # Convert bbox to integers
            x1, y1, x2, y2 = map(int, bbox)

            # Choose color based on class (simple hash)
            color = self._get_color(detection['class_id'])

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, label_size[1] + 10)

            # Draw label background
            cv2.rectangle(
                image,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0], label_y),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Save annotated image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved annotated image to {output_path}")

    def detect_and_annotate_batch(
        self,
        image_dir: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Detect and annotate a batch of images.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save annotated images

        Returns:
            List of detection results
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get detection results
        results = self.detect_batch(image_dir)

        # Annotate each image
        for result in tqdm(results, desc="Annotating images"):
            image_path = Path(result['image_path'])
            output_path = output_dir / image_path.name

            try:
                self.annotate_image(
                    image_path,
                    output_path,
                    detections=result['detections']
                )
            except Exception as e:
                logger.error(f"Failed to annotate {image_path}: {e}")
                continue

        logger.info(f"Annotated {len(results)} images, saved to {output_dir}")
        return results

    def _get_color(self, class_id: int) -> tuple:
        """
        Get a color for a class ID.

        Args:
            class_id: Class ID

        Returns:
            BGR color tuple
        """
        # Generate a deterministic color based on class_id
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
