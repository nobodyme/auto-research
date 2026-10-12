"""Tests for SAM3 integration module."""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSAM3Detector:
    """Test SAM3 detector class for prompt-based segmentation."""

    def test_sam3_initialization(self):
        """Test SAM3 detector initializes correctly."""
        from sam3_integration import SAM3Detector

        detector = SAM3Detector()
        assert detector.model is not None
        assert detector.processor is not None

    def test_detect_with_prompt(self, tmp_path):
        """Test detection with text prompt."""
        from sam3_integration import SAM3Detector

        detector = SAM3Detector()

        # Create dummy image
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), img)

        results = detector.detect(str(image_path), prompt="detect cars")

        assert 'detections' in results
        assert 'masks' in results
        assert 'prompt' in results
        assert results['prompt'] == "detect cars"

    def test_detect_returns_bounding_boxes(self, tmp_path):
        """Test detection returns bounding boxes from SAM3."""
        from sam3_integration import SAM3Detector

        detector = SAM3Detector()

        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), img)

        results = detector.detect(str(image_path), prompt="cars")

        # Mock model returns at least one detection
        assert 'detections' in results
        if results['detections']:
            assert 'bbox' in results['detections'][0]
            assert len(results['detections'][0]['bbox']) == 4

    def test_detect_returns_masks(self, tmp_path):
        """Test detection returns segmentation masks."""
        from sam3_integration import SAM3Detector

        detector = SAM3Detector()

        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), img)

        results = detector.detect(str(image_path), prompt="cars")

        assert 'masks' in results
        assert results['masks'] is not None


class TestBatchSAM3Detection:
    """Test batch detection with SAM3."""

    def test_detect_folder_with_prompt(self, tmp_path):
        """Test folder detection with prompt."""
        from sam3_integration import SAM3Detector

        detector = SAM3Detector()

        # Create dummy images
        import cv2
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"test_{i}.jpg"), img)

        results = detector.detect_folder(str(tmp_path), prompt="cars")

        assert len(results) == 3


class TestDatasetPreparation:
    """Test dataset preparation for fine-tuning."""

    def test_prepare_training_data(self, tmp_path):
        """Test preparing training data from SAM3 detections."""
        from sam3_integration import prepare_training_data

        # Create dummy image files
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'img1.jpg'), img)
        cv2.imwrite(str(tmp_path / 'img2.jpg'), img)

        sam3_results = [
            {
                'image_path': str(tmp_path / 'img1.jpg'),
                'detections': [
                    {'bbox': [100, 100, 200, 200], 'confidence': 0.9}
                ],
                'masks': np.array([[[True, False], [False, True]]])
            },
            {
                'image_path': str(tmp_path / 'img2.jpg'),
                'detections': [
                    {'bbox': [50, 50, 150, 150], 'confidence': 0.85}
                ],
                'masks': np.array([[[False, True], [True, False]]])
            }
        ]

        output_dir = tmp_path / 'training_data'
        training_data = prepare_training_data(sam3_results, str(output_dir), class_name="car")

        assert 'images_dir' in training_data
        assert 'annotations_file' in training_data
        assert training_data['num_samples'] == 2

    def test_export_to_coco_format(self, tmp_path):
        """Test exporting to COCO format for training."""
        from sam3_integration import export_to_coco_format

        # Create a real image for dimensions
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'img1.jpg'), img)

        sam3_results = [
            {
                'image_path': str(tmp_path / 'img1.jpg'),
                'detections': [
                    {'bbox': [100, 100, 200, 200], 'confidence': 0.9}
                ],
                'masks': np.array([[[True, False], [False, True]]])
            }
        ]

        output_file = tmp_path / 'annotations.json'
        export_to_coco_format(sam3_results, str(output_file), class_name="car")

        assert output_file.exists()

        with open(output_file) as f:
            coco_data = json.load(f)

        assert 'images' in coco_data
        assert 'annotations' in coco_data
        assert 'categories' in coco_data
        assert len(coco_data['images']) == 1
        assert len(coco_data['annotations']) == 1

    def test_mask_to_bbox_conversion(self):
        """Test converting mask to bounding box."""
        from sam3_integration import mask_to_bbox

        mask = np.zeros((100, 100), dtype=bool)
        mask[20:50, 30:60] = True

        bbox = mask_to_bbox(mask)

        # np.where returns indices - last index is 49, 59 (exclusive end - 1)
        assert bbox == [30, 20, 59, 49]  # [x_min, y_min, x_max, y_max]

    def test_validate_detections(self):
        """Test validating detections before training."""
        from sam3_integration import validate_detections

        # Valid detections
        valid = [
            {
                'image_path': '/path/to/img.jpg',
                'detections': [{'bbox': [10, 10, 50, 50], 'confidence': 0.9}]
            }
        ]
        assert validate_detections(valid) is True

        # Invalid - empty list
        assert validate_detections([]) is False


class TestSAM3ModelLoading:
    """Test SAM3 model loading and configuration."""

    def test_check_sam3_installation(self):
        """Test checking SAM3 installation."""
        from sam3_integration import check_sam3_available

        # This should return True/False based on availability
        result = check_sam3_available()
        assert isinstance(result, bool)

    def test_get_sam3_model_info(self):
        """Test getting SAM3 model information."""
        from sam3_integration import SAM3Detector

        detector = SAM3Detector()
        info = detector.get_model_info()

        assert 'name' in info
        assert info['name'] == 'SAM3'
        assert 'parameters' in info


class TestAnnotationGenerator:
    """Test annotation generator."""

    def test_generate_annotations(self, tmp_path):
        """Test generating annotations from images."""
        from sam3_integration import SAM3AnnotationGenerator

        # Create dummy images
        import cv2
        for i in range(2):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i}.jpg"), img)

        generator = SAM3AnnotationGenerator()
        output_dir = tmp_path / 'output'

        result = generator.generate_annotations(
            folder_path=str(tmp_path),
            prompt="detect objects",
            output_dir=str(output_dir),
            class_name="object"
        )

        assert 'images_dir' in result
        assert 'annotations_file' in result
        assert 'total_detections' in result
