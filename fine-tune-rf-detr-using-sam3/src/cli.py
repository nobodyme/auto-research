#!/usr/bin/env python3
"""CLI tool for RF-DETR object detection with SAM3 fine-tuning support."""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='RF-DETR Object Detection CLI with SAM3 Fine-tuning Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection on a folder of images
  python cli.py /path/to/images

  # Detection with specific RF-DETR variant
  python cli.py /path/to/images --variant m

  # Fine-tune with SAM3 using a text prompt
  python cli.py /path/to/images --prompt "detect yellow school buses"

  # Fine-tune and save the model
  python cli.py /path/to/images --prompt "detect cars" --save-model --model-output finetuned_rfdetr.pth
        """
    )

    # Required arguments
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing images for detection'
    )

    # Optional arguments
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for SAM3 detection. When provided, enables fine-tuning mode.'
    )

    parser.add_argument(
        '--variant',
        type=str,
        choices=['n', 's', 'm'],
        default='n',
        help='RF-DETR model variant: n=Nano, s=Small, m=Medium (default: n)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save detection results (default: output)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (default: 0.5)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs for fine-tuning (default: 10)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for fine-tuning (default: 8)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate for fine-tuning (default: 1e-4)'
    )

    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the fine-tuned model'
    )

    parser.add_argument(
        '--model-output',
        type=str,
        default=None,
        help='Path to save the fine-tuned model'
    )

    parser.add_argument(
        '--class-name',
        type=str,
        default='object',
        help='Class name for detected objects when fine-tuning (default: object)'
    )

    parser.add_argument(
        '--annotate',
        action='store_true',
        default=True,
        help='Save annotated images with bounding boxes (default: True)'
    )

    parser.add_argument(
        '--no-annotate',
        action='store_true',
        help='Disable saving annotated images'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args(args)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed arguments namespace.

    Raises:
        ValueError: If arguments are invalid.
    """
    # Check folder exists
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder path does not exist: {args.folder_path}")

    # Check it's a directory
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {args.folder_path}")

    # Check for images
    images = get_image_files(args.folder_path)
    if not images:
        raise ValueError(f"No images found in folder: {args.folder_path}")

    # Validate confidence threshold
    if not 0 <= args.confidence <= 1:
        raise ValueError(f"Confidence threshold must be between 0 and 1: {args.confidence}")

    # Validate epochs
    if args.epochs < 1:
        raise ValueError(f"Epochs must be at least 1: {args.epochs}")

    logger.info(f"Found {len(images)} images in {args.folder_path}")


def get_image_files(folder_path: str) -> List[Path]:
    """Get list of image files in folder.

    Args:
        folder_path: Path to folder.

    Returns:
        List of image file paths.
    """
    folder = Path(folder_path)
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f'*{ext}'))
        images.extend(folder.glob(f'*{ext.upper()}'))

    return sorted(images)


def run_detection(args: argparse.Namespace) -> None:
    """Run object detection on folder of images.

    Args:
        args: Parsed arguments namespace.
    """
    from detector import RFDETRDetector, save_results_json

    logger.info(f"Running RF-DETR detection (variant: {args.variant})")

    # Initialize detector
    detector = RFDETRDetector(
        variant=args.variant,
        confidence_threshold=args.confidence
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run detection on folder
    results = detector.detect_folder(args.folder_path)

    # Save annotated images if requested
    annotate = args.annotate and not args.no_annotate
    if annotate:
        annotated_dir = output_dir / 'annotated'
        annotated_dir.mkdir(exist_ok=True)

        for result in results:
            image_path = result['image_path']
            output_path = annotated_dir / Path(image_path).name
            detector.annotate_image(
                image_path,
                result['detections'],
                str(output_path)
            )

        logger.info(f"Saved annotated images to {annotated_dir}")

    # Save results JSON
    results_file = output_dir / 'detection_results.json'
    save_results_json(results, str(results_file))
    logger.info(f"Saved detection results to {results_file}")

    # Print summary
    total_detections = sum(len(r['detections']) for r in results)
    logger.info(f"Detection complete: {len(results)} images, {total_detections} total detections")


def run_finetune_pipeline(args: argparse.Namespace) -> None:
    """Run SAM3-based fine-tuning pipeline.

    Args:
        args: Parsed arguments namespace.
    """
    from finetune import FineTuningPipeline, TrainingConfig

    logger.info(f"Running fine-tuning pipeline with prompt: '{args.prompt}'")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = FineTuningPipeline(variant=args.variant)

    # Generate training data from SAM3 detections
    logger.info("Generating training data using SAM3...")
    training_data = pipeline.generate_training_data(
        folder_path=args.folder_path,
        prompt=args.prompt,
        class_name=args.class_name,
        output_dir=str(output_dir / 'training_data')
    )

    logger.info(f"Generated {training_data['num_samples']} training samples")

    # Fine-tune RF-DETR
    logger.info(f"Fine-tuning RF-DETR for {args.epochs} epochs...")
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    result = pipeline.finetune(training_data, config=config)

    logger.info(f"Fine-tuning complete. Best mAP: {result.get('best_mAP', 'N/A')}")

    # Save model if requested
    if args.save_model:
        model_path = args.model_output or str(output_dir / 'finetuned_rfdetr.pth')
        pipeline.save_model(model_path)
        logger.info(f"Saved fine-tuned model to {model_path}")

    # Run detection with fine-tuned model
    logger.info("Running detection with fine-tuned model...")
    results = pipeline.detect_folder(args.folder_path)

    # Save annotated images
    annotate = args.annotate and not args.no_annotate
    if annotate:
        annotated_dir = output_dir / 'annotated_finetuned'
        annotated_dir.mkdir(exist_ok=True)

        for res in results:
            image_path = res['image_path']
            output_path = annotated_dir / Path(image_path).name
            pipeline.detector.annotate_image(
                image_path,
                res['detections'],
                str(output_path)
            )

        logger.info(f"Saved annotated images to {annotated_dir}")

    # Save results
    from detector import save_results_json
    results_file = output_dir / 'finetuned_detection_results.json'
    save_results_json(results, str(results_file))

    # Print summary
    total_detections = sum(len(r['detections']) for r in results)
    logger.info(f"Pipeline complete: {len(results)} images, {total_detections} detections")


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate arguments
        validate_args(args)

        # Run appropriate pipeline
        if args.prompt:
            run_finetune_pipeline(args)
        else:
            run_detection(args)

    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
