# Fine-tune RF-DETR using SAM3

A CLI tool for object detection using RF-DETR with SAM3-based fine-tuning support.

## Overview

This tool provides two main capabilities:

1. **Object Detection**: Run RF-DETR object detection on a folder of images with any model variant (Nano, Small, Medium)
2. **Fine-tuning with SAM3**: Use Meta's Segment Anything Model 3 (SAM3) to detect objects via text prompts, then fine-tune RF-DETR using those detections

## Features

- Support for all RF-DETR variants (Nano, Small, Medium)
- Text-prompted object detection using SAM3
- Automatic COCO-format annotation generation
- Fine-tuning pipeline with configurable hyperparameters
- Annotated image output with bounding boxes
- JSON export of detection results

## Installation

```bash
# Clone and install
cd fine-tune-rf-detr-using-sam3
pip install -r requirements.txt

# For SAM3 (requires HuggingFace authentication)
pip install git+https://github.com/facebookresearch/sam3.git
huggingface-cli login  # Authenticate for SAM3 weights
```

### Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

## Usage

### Basic Object Detection

Run RF-DETR detection on a folder of images:

```bash
python src/cli.py /path/to/images

# With specific model variant
python src/cli.py /path/to/images --variant m  # Medium model

# With custom confidence threshold
python src/cli.py /path/to/images --confidence 0.7
```

### Fine-tuning with SAM3

Use SAM3 to detect objects with a text prompt, then fine-tune RF-DETR:

```bash
# Fine-tune to detect specific objects
python src/cli.py /path/to/images --prompt "detect yellow school buses"

# With more epochs and save the model
python src/cli.py /path/to/images --prompt "detect cars" --epochs 50 --save-model
```

### CLI Options

```
positional arguments:
  folder_path           Path to folder containing images for detection

optional arguments:
  --prompt TEXT         Text prompt for SAM3 detection (enables fine-tuning mode)
  --variant {n,s,m}     RF-DETR variant: n=Nano, s=Small, m=Medium (default: n)
  --output-dir DIR      Output directory (default: output)
  --confidence FLOAT    Confidence threshold 0-1 (default: 0.5)
  --epochs INT          Fine-tuning epochs (default: 10)
  --batch-size INT      Training batch size (default: 8)
  --learning-rate FLOAT Learning rate (default: 1e-4)
  --save-model          Save the fine-tuned model
  --model-output PATH   Path to save fine-tuned model
  --class-name TEXT     Class name for detected objects (default: object)
  --no-annotate         Disable annotated image output
  --verbose             Enable verbose logging
```

## Architecture

```
fine-tune-rf-detr-using-sam3/
├── src/
│   ├── cli.py              # CLI entry point
│   ├── detector.py         # RF-DETR detection module
│   ├── sam3_integration.py # SAM3 prompt-based detection
│   └── finetune.py         # Fine-tuning pipeline
├── tests/
│   ├── test_cli.py
│   ├── test_detector.py
│   ├── test_sam3_integration.py
│   └── test_finetune.py
├── requirements.txt
├── notes.md
└── README.md
```

## How It Works

### Detection Mode (No Prompt)

1. Load RF-DETR model (Nano/Small/Medium variant)
2. Process each image in the folder
3. Apply confidence threshold filtering
4. Generate annotated images with bounding boxes
5. Export results to JSON

### Fine-tuning Mode (With Prompt)

1. Load SAM3 model
2. Process images with text prompt to detect objects
3. Convert SAM3 detections to COCO format annotations
4. Fine-tune RF-DETR on the generated training data
5. Run detection with fine-tuned model
6. Save results and optionally save the model

## Model Variants

| Variant | Name | Parameters | Latency | mAP |
|---------|------|------------|---------|-----|
| n | RF-DETR Nano | 30.5M | 2.32ms | High |
| s | RF-DETR Small | 32.1M | 3.52ms | Higher |
| m | RF-DETR Medium | 33.7M | 4.52ms | 54.7% |

## Output

### Detection Results

```
output/
├── annotated/              # Images with bounding boxes
│   ├── image1.jpg
│   └── image2.jpg
└── detection_results.json  # JSON with all detections
```

### JSON Format

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "detections": [
      {
        "bbox": [100, 100, 200, 200],
        "confidence": 0.95,
        "class_id": 0,
        "class_name": "person"
      }
    ]
  }
]
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## References

- [RF-DETR](https://github.com/roboflow/rfdetr) - Roboflow's state-of-the-art transformer detector
- [SAM3](https://github.com/facebookresearch/sam3) - Meta's Segment Anything Model 3
- [COCO Format](https://cocodataset.org/#format-data) - Annotation format specification

## License

MIT License
