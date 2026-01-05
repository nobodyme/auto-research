"""
Benchmark evaluation for object detection models.
Includes COCO evaluation and latency benchmarking.
"""
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyBenchmark:
    """Measure inference latency for object detection models."""

    def __init__(self, image_size: tuple = (640, 640)):
        """
        Initialize latency benchmark.

        Args:
            image_size: Input image size (height, width)
        """
        self.image_size = image_size

    def measure_latency(
        self,
        model: Any,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        use_gpu: bool = True
    ) -> Dict[str, float]:
        """
        Measure model inference latency.

        Args:
            model: Model to benchmark
            num_iterations: Number of iterations for measurement
            warmup_iterations: Number of warmup iterations
            use_gpu: Whether to use GPU if available

        Returns:
            Dictionary containing latency statistics
        """
        logger.info(f"Measuring latency: {num_iterations} iterations, {warmup_iterations} warmup")

        # Create dummy input
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        dummy_image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)

        # Warmup
        logger.info("Warming up...")
        for _ in range(warmup_iterations):
            try:
                _ = model.predict(dummy_image, verbose=False, device=device)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")

        # Synchronize GPU if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        # Measure latency
        latencies = []
        logger.info("Measuring latency...")
        for _ in tqdm(range(num_iterations), desc="Latency benchmark"):
            start_time = time.perf_counter()

            try:
                _ = model.predict(dummy_image, verbose=False, device=device)

                # Synchronize GPU
                if device == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            except Exception as e:
                logger.warning(f"Iteration failed: {e}")
                continue

        if not latencies:
            logger.error("No successful iterations")
            return {
                'mean_latency_ms': 0,
                'std_latency_ms': 0,
                'min_latency_ms': 0,
                'max_latency_ms': 0,
                'fps': 0
            }

        # Calculate statistics
        latencies = np.array(latencies)
        results = {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'fps': 1000.0 / float(np.mean(latencies)),
            'device': device
        }

        logger.info(f"Mean latency: {results['mean_latency_ms']:.2f} ms")
        logger.info(f"FPS: {results['fps']:.2f}")

        return results


class COCOEvaluator:
    """Evaluate models on COCO dataset."""

    def __init__(self, max_images: Optional[int] = None, coco_val_path: Optional[str] = None):
        """
        Initialize COCO evaluator.

        Args:
            max_images: Maximum number of images to evaluate (None for all)
            coco_val_path: Path to COCO validation dataset
        """
        self.max_images = max_images
        self.coco_val_path = coco_val_path

    def evaluate(self, model: Any, use_gpu: bool = True) -> Dict[str, float]:
        """
        Evaluate model on COCO validation set.

        Args:
            model: Model to evaluate
            use_gpu: Whether to use GPU if available

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating on COCO validation set...")

        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        try:
            # Use ultralytics built-in validation
            # This will download COCO val2017 if not present
            results = model.val(
                data='coco.yaml',
                device=device,
                verbose=False,
                plots=False
            )

            # Extract metrics
            metrics = {
                'mAP': float(results.box.map),  # mAP@0.5:0.95
                'mAP50': float(results.box.map50),  # mAP@0.5
                'mAP75': float(results.box.map75),  # mAP@0.75
                'mAP_small': float(results.box.maps[0]),  # small objects
                'mAP_medium': float(results.box.maps[1]),  # medium objects
                'mAP_large': float(results.box.maps[2]),  # large objects
            }

            # Measure inference time
            latency_benchmark = LatencyBenchmark()
            latency_results = latency_benchmark.measure_latency(
                model,
                num_iterations=50,
                warmup_iterations=5,
                use_gpu=use_gpu
            )

            metrics.update({
                'inference_time_ms': latency_results['mean_latency_ms'],
                'fps': latency_results['fps']
            })

            logger.info(f"mAP@0.5:0.95: {metrics['mAP']:.4f}")
            logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"FPS: {metrics['fps']:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")
            # Return default metrics if evaluation fails
            return {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0,
                'inference_time_ms': 0.0,
                'fps': 0.0,
                'error': str(e)
            }


class ModelComparator:
    """Compare multiple object detection models."""

    def __init__(self):
        """Initialize model comparator."""
        self.coco_evaluator = COCOEvaluator()
        self.latency_benchmark = LatencyBenchmark()

    def compare_models(
        self,
        models: Dict[str, Any],
        max_images: Optional[int] = None,
        use_gpu: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on various metrics.

        Args:
            models: Dictionary mapping model names to model objects
            max_images: Maximum images for COCO evaluation
            use_gpu: Whether to use GPU

        Returns:
            Dictionary with comparison results for each model
        """
        results = {}

        for model_name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'='*50}")

            try:
                # Get model info
                from .model_loader import ModelLoader
                loader = ModelLoader()
                model_info = loader.get_model_info(model)

                # Measure latency
                latency_results = self.latency_benchmark.measure_latency(
                    model,
                    num_iterations=50,
                    warmup_iterations=5,
                    use_gpu=use_gpu
                )

                # Evaluate on COCO (if not in quick mode)
                if max_images is None or max_images > 100:
                    coco_evaluator = COCOEvaluator(max_images=max_images)
                    coco_results = coco_evaluator.evaluate(model, use_gpu=use_gpu)
                else:
                    # Skip full COCO eval for quick testing
                    coco_results = {
                        'mAP': 0.0,
                        'mAP50': 0.0,
                        'mAP75': 0.0,
                        'note': 'Skipped for quick testing'
                    }

                # Combine results
                results[model_name] = {
                    'parameters': model_info.get('parameters', 0),
                    'model_size_mb': model_info.get('model_size_mb', 0),
                    'latency_ms': latency_results['mean_latency_ms'],
                    'latency_std_ms': latency_results['std_latency_ms'],
                    'fps': latency_results['fps'],
                    'mAP': coco_results.get('mAP', 0.0),
                    'mAP50': coco_results.get('mAP50', 0.0),
                    'mAP75': coco_results.get('mAP75', 0.0),
                }

                logger.info(f"Results for {model_name}:")
                logger.info(f"  Parameters: {results[model_name]['parameters']:,}")
                logger.info(f"  Latency: {results[model_name]['latency_ms']:.2f} ms")
                logger.info(f"  FPS: {results[model_name]['fps']:.2f}")
                logger.info(f"  mAP: {results[model_name]['mAP']:.4f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = {
                    'error': str(e)
                }

        return results
