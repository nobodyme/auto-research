#!/usr/bin/env python3
"""
Object detection script for processing folders of images.
Uses the best performing model identified from benchmarks.
"""
import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.detect import ObjectDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform object detection on a folder of images'
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing input images'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save annotated images (default: output)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model to use (if not specified, uses best from comparison results)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )

    parser.add_argument(
        '--save-results',
        type=str,
        default='detection_results.json',
        help='Save detection results to JSON file (default: detection_results.json)'
    )

    return parser.parse_args()


def load_best_model_from_results(results_file: Path) -> str:
    """
    Load the best model name from comparison results.

    Args:
        results_file: Path to comparison results JSON

    Returns:
        Name of the best model
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Calculate efficiency scores
        efficiency_scores = {}
        for model_name, model_results in results.items():
            mAP = model_results.get('mAP', 0)
            fps = model_results.get('fps', 0)
            if mAP > 0 and fps > 0:
                # Geometric mean of normalized mAP and FPS
                efficiency_scores[model_name] = (mAP * fps) ** 0.5

        if efficiency_scores:
            best_model = max(efficiency_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Selected best model from results: {best_model}")
            return best_model
        else:
            logger.warning("Could not determine best model, using default: yolov11n")
            return 'yolov11n'

    except Exception as e:
        logger.warning(f"Could not load comparison results: {e}")
        logger.warning("Using default model: yolov11n")
        return 'yolov11n'


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("="*60)
    logger.info("Object Detection on Images")
    logger.info("="*60)

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model to use
    if args.model:
        model_name = args.model
        logger.info(f"Using specified model: {model_name}")
    else:
        # Try to load best model from comparison results
        results_file = Path('results/comparison_results.json')
        if results_file.exists():
            model_name = load_best_model_from_results(results_file)
        else:
            logger.warning("No comparison results found, using default: yolov11n")
            model_name = 'yolov11n'

    # Initialize detector
    logger.info(f"\nInitializing detector with {model_name}...")
    logger.info(f"Confidence threshold: {args.confidence}")

    try:
        detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=args.confidence,
            use_gpu=not args.no_gpu
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1

    # Run detection
    logger.info(f"\nProcessing images from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    try:
        results = detector.detect_and_annotate_batch(
            input_dir,
            output_dir
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return 1

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Detection Summary")
    logger.info("="*60)

    total_images = len(results)
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / total_images if total_images > 0 else 0

    logger.info(f"\nProcessed images: {total_images}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Average detections per image: {avg_detections:.2f}")

    # Count detections by class
    class_counts = {}
    for result in results:
        for detection in result['detections']:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    if class_counts:
        logger.info("\nDetected objects by class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {class_name}: {count}")

    # Save results to JSON
    if args.save_results:
        results_file = output_dir / args.save_results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nDetailed results saved to: {results_file}")

    logger.info(f"\nAnnotated images saved to: {output_dir}")
    logger.info("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
