"""
Tests for object detection on images.
Following TDD approach - these tests define expected behavior.
"""
import pytest
import numpy as np
from pathlib import Path
import cv2


class TestObjectDetection:
    """Test suite for object detection functionality."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(img_path), img)
        return img_path

    @pytest.fixture
    def sample_images_dir(self, tmp_path):
        """Create a directory with sample images."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(3):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = img_dir / f"test_image_{i}.jpg"
            cv2.imwrite(str(img_path), img)

        return img_dir

    def test_detect_single_image(self, sample_image):
        """Test detection on a single image."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(model_name='yolov8n')
        results = detector.detect(sample_image)

        assert results is not None
        assert 'detections' in results
        assert 'image_path' in results

    def test_detect_batch_images(self, sample_images_dir):
        """Test detection on a batch of images."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(model_name='yolov8n')
        results = detector.detect_batch(sample_images_dir)

        assert len(results) == 3
        for result in results:
            assert 'detections' in result
            assert 'image_path' in result

    def test_annotate_image(self, sample_image, tmp_path):
        """Test image annotation with bounding boxes."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(model_name='yolov8n')
        output_path = tmp_path / "annotated.jpg"

        detector.annotate_image(
            sample_image,
            output_path
        )

        assert output_path.exists()

    def test_batch_detection_with_annotation(self, sample_images_dir, tmp_path):
        """Test batch detection with annotation output."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(model_name='yolov8n')
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        detector.detect_and_annotate_batch(
            sample_images_dir,
            output_dir
        )

        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 3

    def test_confidence_threshold(self, sample_image):
        """Test confidence threshold filtering."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(
            model_name='yolov8n',
            confidence_threshold=0.5
        )
        results = detector.detect(sample_image)

        # All detections should have confidence >= 0.5
        if results['detections']:
            for detection in results['detections']:
                assert detection['confidence'] >= 0.5
