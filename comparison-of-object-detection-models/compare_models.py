#!/usr/bin/env python3
"""
Main script to compare object detection models.
Benchmarks multiple models and creates comparison visualizations.
"""
import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.model_loader import ModelLoader
from src.benchmark import ModelComparator, COCOEvaluator, LatencyBenchmark
from src.rf100_evaluator import SimpleRoboflowEvaluator
from src.visualize import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare object detection models on various benchmarks'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['yolov8n', 'yolov10n', 'yolov11n', 'rtdetr-l'],
        help='Models to compare (default: yolov8n yolov10n yolov11n rtdetr-l)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip full COCO evaluation (for testing)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )

    parser.add_argument(
        '--latency-iterations',
        type=int,
        default=100,
        help='Number of iterations for latency measurement (default: 100)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("="*60)
    logger.info("Object Detection Model Comparison")
    logger.info("="*60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = not args.no_gpu

    # Initialize components
    logger.info("\nInitializing components...")
    loader = ModelLoader()
    comparator = ModelComparator()
    visualizer = ResultsVisualizer(output_dir)

    # Load models
    logger.info(f"\nLoading models: {args.models}")
    models = {}

    for model_name in args.models:
        try:
            logger.info(f"Loading {model_name}...")
            model = loader.load_model(model_name)
            models[model_name] = model
            logger.info(f"✓ {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")
            continue

    if not models:
        logger.error("No models loaded successfully. Exiting.")
        return 1

    logger.info(f"\nSuccessfully loaded {len(models)} models")

    # Run comparison
    logger.info("\n" + "="*60)
    logger.info("Running Benchmarks")
    logger.info("="*60)

    max_images = 100 if args.quick else None
    results = comparator.compare_models(
        models,
        max_images=max_images,
        use_gpu=use_gpu
    )

    # Check fine-tuning capability
    logger.info("\n" + "="*60)
    logger.info("Checking Fine-tuning Capability")
    logger.info("="*60)

    ft_evaluator = SimpleRoboflowEvaluator()
    for model_name, model in models.items():
        ft_result = ft_evaluator.evaluate_fine_tuning(model_name, model)
        results[model_name]['fine_tuning_supported'] = ft_result['fine_tuning_supported']
        logger.info(f"{model_name}: Fine-tuning {'✓ supported' if ft_result['fine_tuning_supported'] else '✗ not supported'}")

    # Save raw results
    results_file = output_dir / 'comparison_results.json'
    with open(results_file, 'w') as f:
        # Convert any non-serializable values
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in model_results.items()
            }
        json.dump(serializable_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Create visualizations
    logger.info("\n" + "="*60)
    logger.info("Creating Visualizations")
    logger.info("="*60)

    visualizer.create_all_plots(results)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Summary")
    logger.info("="*60)

    # Find best models
    best_mAP = max(results.items(), key=lambda x: x[1].get('mAP', 0))
    best_fps = max(results.items(), key=lambda x: x[1].get('fps', 0))
    best_latency = min(
        results.items(),
        key=lambda x: x[1].get('latency_ms', float('inf'))
    )

    logger.info(f"\nBest Accuracy (mAP): {best_mAP[0]} ({best_mAP[1]['mAP']*100:.2f}%)")
    logger.info(f"Best Speed (FPS): {best_fps[0]} ({best_fps[1]['fps']:.2f} FPS)")
    logger.info(f"Best Latency: {best_latency[0]} ({best_latency[1]['latency_ms']:.2f} ms)")

    # Efficiency score (balanced mAP and FPS)
    efficiency_scores = {}
    for model_name, model_results in results.items():
        mAP = model_results.get('mAP', 0)
        fps = model_results.get('fps', 0)
        # Normalize and combine (simple geometric mean)
        if mAP > 0 and fps > 0:
            efficiency_scores[model_name] = (mAP * fps) ** 0.5

    if efficiency_scores:
        best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
        logger.info(f"\nBest Overall (Efficiency): {best_efficiency[0]}")
        logger.info(f"  - Balanced score: {best_efficiency[1]:.4f}")
        logger.info(f"  - mAP: {results[best_efficiency[0]]['mAP']*100:.2f}%")
        logger.info(f"  - FPS: {results[best_efficiency[0]]['fps']:.2f}")

    logger.info("\n" + "="*60)
    logger.info(f"All results and visualizations saved to: {output_dir}")
    logger.info("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
