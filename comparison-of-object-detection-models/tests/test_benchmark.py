"""
Tests for benchmark evaluation functionality.
Following TDD approach - these tests define expected behavior.
"""
import pytest
import numpy as np
from pathlib import Path


class TestBenchmarkEvaluator:
    """Test suite for benchmark evaluation."""

    def test_coco_evaluator_initialization(self):
        """Test COCO evaluator can be initialized."""
        from src.benchmark import COCOEvaluator

        evaluator = COCOEvaluator()
        assert evaluator is not None

    def test_evaluate_model_on_coco(self):
        """Test model evaluation on COCO dataset."""
        from src.benchmark import COCOEvaluator
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader.load_model('yolov8n')
        evaluator = COCOEvaluator(max_images=10)  # Small subset for testing

        results = evaluator.evaluate(model)

        assert 'mAP' in results
        assert 'mAP50' in results
        assert 'mAP75' in results
        assert 'inference_time_ms' in results
        assert 'fps' in results

    def test_latency_measurement(self):
        """Test latency measurement functionality."""
        from src.benchmark import LatencyBenchmark
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader.load_model('yolov8n')
        benchmark = LatencyBenchmark()

        latency_results = benchmark.measure_latency(
            model,
            num_iterations=10,
            warmup_iterations=2
        )

        assert 'mean_latency_ms' in latency_results
        assert 'std_latency_ms' in latency_results
        assert 'min_latency_ms' in latency_results
        assert 'max_latency_ms' in latency_results
        assert 'fps' in latency_results
        assert latency_results['mean_latency_ms'] > 0
        assert latency_results['fps'] > 0

    def test_model_comparison(self):
        """Test comparing multiple models."""
        from src.benchmark import ModelComparator
        from src.model_loader import ModelLoader

        loader = ModelLoader()
        models = {
            'yolov8n': loader.load_model('yolov8n'),
        }

        comparator = ModelComparator()
        comparison_results = comparator.compare_models(
            models,
            max_images=5  # Small subset for testing
        )

        assert len(comparison_results) > 0
        assert 'yolov8n' in comparison_results
        assert 'mAP' in comparison_results['yolov8n']
        assert 'latency_ms' in comparison_results['yolov8n']
        assert 'parameters' in comparison_results['yolov8n']
