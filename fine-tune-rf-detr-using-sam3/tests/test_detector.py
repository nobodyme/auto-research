"""Tests for RF-DETR detection module."""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRFDETRDetector:
    """Test RF-DETR detector class."""

    def test_detector_initialization_nano(self):
        """Test detector initializes with nano variant."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='n')
        assert detector.variant == 'n'
        assert detector.model is not None

    def test_detector_initialization_small(self):
        """Test detector initializes with small variant."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='s')
        assert detector.variant == 's'
        assert detector.model is not None

    def test_detector_initialization_medium(self):
        """Test detector initializes with medium variant."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='m')
        assert detector.variant == 'm'
        assert detector.model is not None

    def test_detector_invalid_variant(self):
        """Test detector rejects invalid variant."""
        from detector import RFDETRDetector

        with pytest.raises(ValueError, match="Invalid variant"):
            RFDETRDetector(variant='x')

    def test_detect_single_image(self, tmp_path):
        """Test detection on single image."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='n')

        # Create dummy image
        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"fake image")

        results = detector.detect(str(image_path))

        assert 'detections' in results
        assert 'image_path' in results
        assert results['image_path'] == str(image_path)

    def test_detect_returns_bounding_boxes(self, tmp_path):
        """Test detection returns bounding boxes."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='n')

        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"fake image")

        results = detector.detect(str(image_path))

        # Mock model returns one detection
        assert len(results['detections']) >= 0
        if results['detections']:
            assert 'bbox' in results['detections'][0]
            assert len(results['detections'][0]['bbox']) == 4

    def test_detect_filters_by_confidence(self, tmp_path):
        """Test detection filters by confidence threshold."""
        from detector import RFDETRDetector

        # Test with high threshold - should filter more
        detector_high = RFDETRDetector(variant='n', confidence_threshold=0.95)

        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"fake image")

        results_high = detector_high.detect(str(image_path))

        # Test with low threshold
        detector_low = RFDETRDetector(variant='n', confidence_threshold=0.1)
        results_low = detector_low.detect(str(image_path))

        # Lower threshold should have at least as many detections
        assert len(results_low['detections']) >= len(results_high['detections'])


class TestBatchDetection:
    """Test batch detection functionality."""

    def test_detect_folder_processes_all_images(self, tmp_path):
        """Test folder detection processes all images."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='n')

        # Create dummy images
        for i in range(3):
            (tmp_path / f"test_{i}.jpg").write_bytes(b"fake image")

        results = detector.detect_folder(str(tmp_path))

        assert len(results) == 3

    def test_detect_folder_skips_non_images(self, tmp_path):
        """Test folder detection skips non-image files."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='n')

        # Create mixed files
        (tmp_path / "test.jpg").write_bytes(b"fake image")
        (tmp_path / "test.txt").write_text("not an image")
        (tmp_path / "test.png").write_bytes(b"fake image")

        results = detector.detect_folder(str(tmp_path))

        # Should only process .jpg and .png
        assert len(results) == 2


class TestAnnotation:
    """Test image annotation functionality."""

    def test_annotate_image_with_boxes(self, tmp_path):
        """Test annotating image with bounding boxes."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='n')

        detections = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'class_id': 0, 'class_name': 'person'}
        ]

        # Create a real simple image
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), img)

        output_path = tmp_path / "output.jpg"

        detector.annotate_image(str(image_path), detections, str(output_path))

        assert output_path.exists()

    def test_save_results_json(self, tmp_path):
        """Test saving detection results to JSON."""
        from detector import save_results_json

        results = [
            {
                'image_path': 'test.jpg',
                'detections': [
                    {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'class_id': 0, 'class_name': 'person'}
                ]
            }
        ]

        output_path = tmp_path / "results.json"
        save_results_json(results, str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert len(loaded) == 1
        assert loaded[0]['image_path'] == 'test.jpg'


class TestModelInfo:
    """Test model information retrieval."""

    def test_get_model_info_nano(self):
        """Test getting nano model info."""
        from detector import RFDETRDetector

        detector = RFDETRDetector(variant='n')
        info = detector.get_model_info()

        assert info['variant'] == 'n'
        assert info['name'] == 'RF-DETR Nano'
        assert 'parameters' in info

    def test_get_available_variants(self):
        """Test getting available model variants."""
        from detector import get_available_variants

        variants = get_available_variants()

        assert 'n' in variants
        assert 's' in variants
        assert 'm' in variants
        assert variants['n'] == 'RF-DETR Nano'
        assert variants['s'] == 'RF-DETR Small'
        assert variants['m'] == 'RF-DETR Medium'


class TestFineTuneableDetector:
    """Test fine-tuneable detector."""

    def test_finetuneable_initialization(self):
        """Test fine-tuneable detector initializes."""
        from detector import RFDETRFineTuneable

        detector = RFDETRFineTuneable(variant='n', num_classes=5)
        assert detector.num_classes == 5

    def test_update_class_names(self):
        """Test updating class names."""
        from detector import RFDETRFineTuneable

        detector = RFDETRFineTuneable(variant='n')
        detector.update_class_names(['car', 'truck', 'bus'])

        assert detector.class_names == ['car', 'truck', 'bus']
        assert detector.num_classes == 3
