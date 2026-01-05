# Object Detection Models Comparison

A comprehensive comparison of low latency near real-time object detection models, evaluating performance on COCO and RF100-VL benchmarks with graphical analysis.

## 📋 Overview

This project compares state-of-the-art object detection models optimized for low latency inference:

- **YOLOv8** (Ultralytics) - Stable baseline with strong performance
- **YOLOv10** (THU-MIG) - NMS-free architecture for reduced latency variance
- **YOLOv11** (Ultralytics) - Latest YOLO iteration
- **RT-DETR** (Baidu) - Transformer-based real-time detector
- **RF-DETR** (Roboflow) - SOTA real-time detection transformer

### Key Features

✅ **COCO Benchmark Evaluation** - mAP metrics on Microsoft COCO dataset
✅ **Latency Benchmarking** - FPS and inference time measurements
✅ **RF100-VL Framework** - Support for Roboflow's multi-domain benchmark
✅ **Fine-tuning Capability** - All models support transfer learning
✅ **Test-Driven Development** - Comprehensive test suite with pytest
✅ **Batch Image Processing** - Automated detection and annotation
✅ **Visualization Suite** - Graphical comparison charts

## 🔬 Research Findings

### Model Selection Rationale

Based on 2025 research and benchmarks:

1. **YOLOv10** achieves 1.8× faster inference than RT-DETR-R18 with similar accuracy
2. **YOLOv10-x** delivers 54.4% mAP with 23% faster speed than RTDETRv2-x
3. **RT-DETR** reaches 53.1% AP at 108 FPS on T4 GPU
4. **YOLOv8** provides strong balance: 53.9% mAP with real-time speeds
5. All selected models support fine-tuning for custom datasets

### Performance Highlights

| Model | Architecture | Key Advantage | COCO mAP |
|-------|-------------|---------------|----------|
| YOLOv8 | CNN-based | Proven stability, excellent documentation | 53.9% |
| YOLOv10 | NMS-free YOLO | Lowest latency variance, 2.8× fewer params | 54.4% |
| YOLOv11 | Enhanced CNN | Latest optimizations, incremental improvements | ~40% (nano) |
| RT-DETR | Transformer | Superior in complex scenes, global context | 53.1% |
| **RF-DETR** | **Transformer** | **SOTA accuracy, optimized inference** | **54.7%** (M) |

**RF-DETR Standout Performance:**
- RF-DETR-M: 73.6% AP₅₀, 54.7% AP₅₀:₉₅ on COCO
- RF-DETR-N: 67.6% AP₅₀, 48.4% AP₅₀:₉₅ with only 2.32ms latency
- Up to 2× inference speedup with optimization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository (if not already done)
cd comparison-of-object-detection-models

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

## 📊 Running Benchmarks

### Full Model Comparison

```bash
# Compare all models with full COCO evaluation
python compare_models.py

# Quick comparison (skip full COCO eval)
python compare_models.py --quick

# Compare specific models
python compare_models.py --models yolov8n yolov10n yolov11n rfdetr-n

# Compare RF-DETR variants
python compare_models.py --models rfdetr-n rfdetr-s rfdetr-m

# CPU-only mode
python compare_models.py --no-gpu
```

**Output:**
- `results/comparison_results.json` - Raw benchmark data
- `results/latency_comparison.png` - Inference speed charts
- `results/accuracy_comparison.png` - mAP comparison bars
- `results/efficiency_frontier.png` - Accuracy vs Latency trade-off
- `results/model_size_comparison.png` - Parameters and size metrics
- `results/summary_table.png` - Comprehensive summary table
- `results/summary_table.csv` - CSV export for further analysis

### Object Detection on Images

```bash
# Detect objects in a folder of images
python detect_images.py /path/to/images --output-dir annotated_output

# Use specific model
python detect_images.py /path/to/images --model yolov11n

# Adjust confidence threshold
python detect_images.py /path/to/images --confidence 0.5

# Save detection results to JSON
python detect_images.py /path/to/images --save-results detections.json
```

**Features:**
- Automatically uses the best model from benchmark results
- Annotates images with bounding boxes and labels
- Supports all common image formats (JPG, PNG, BMP, TIFF, WebP)
- Exports detailed detection results to JSON

## 🧪 Testing

The project follows Test-Driven Development (TDD) principles:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model_loader.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

- ✅ Model loading and initialization
- ✅ Benchmark evaluation (COCO, latency)
- ✅ Object detection on single and batch images
- ✅ Image annotation with bounding boxes
- ✅ Confidence threshold filtering

## 📁 Project Structure

```
comparison-of-object-detection-models/
├── src/
│   ├── __init__.py
│   ├── model_loader.py       # Model loading and management
│   ├── benchmark.py           # COCO evaluation and latency tests
│   ├── rf100_evaluator.py    # RF100-VL benchmark framework
│   ├── detect.py              # Object detection on images
│   └── visualize.py           # Results visualization
├── tests/
│   ├── test_model_loader.py
│   ├── test_benchmark.py
│   └── test_detect.py
├── compare_models.py          # Main comparison script
├── detect_images.py           # Batch detection script
├── requirements.txt
├── notes.md                   # Development notes
└── README.md                  # This file
```

## 🎯 Benchmark Metrics

### COCO Evaluation
- **mAP@0.5:0.95** - Standard COCO metric
- **mAP@0.5** - IoU threshold at 0.5
- **mAP@0.75** - IoU threshold at 0.75
- Per-size metrics (small, medium, large objects)

### Latency Benchmarking
- **Mean Latency** - Average inference time (ms)
- **FPS** - Frames per second
- **P50, P95, P99** - Percentile latencies
- **Standard Deviation** - Latency variance

### Model Efficiency
- **Parameters** - Total trainable parameters
- **Model Size** - Memory footprint (MB)
- **FLOPs** - Computational complexity
- **Efficiency Score** - Balanced mAP and FPS metric

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from src.model_loader import ModelLoader
from src.benchmark import ModelComparator

loader = ModelLoader()
model = loader.load_model('yolov10n')

# Get model information
info = loader.get_model_info(model)
print(f"Parameters: {info['parameters']:,}")
print(f"Size: {info['model_size_mb']:.2f} MB")
```

### Programmatic Detection

```python
from src.detect import ObjectDetector

detector = ObjectDetector(
    model_name='yolov11n',
    confidence_threshold=0.3
)

# Single image
result = detector.detect('image.jpg')
print(f"Found {result['num_detections']} objects")

# Batch processing
results = detector.detect_and_annotate_batch(
    'input_images/',
    'output_images/'
)
```

### Custom Visualization

```python
from src.visualize import ResultsVisualizer
import json

with open('results/comparison_results.json') as f:
    results = json.load(f)

visualizer = ResultsVisualizer('custom_plots')
visualizer.create_all_plots(results)
```

## 📝 Methodology

### Benchmark Protocol

1. **Model Selection** - Based on 2025 SOTA research
2. **Warmup Phase** - 10 iterations to stabilize GPU/CPU
3. **Latency Measurement** - 100 iterations with mean/std/percentiles
4. **COCO Evaluation** - Full validation set (5000 images)
5. **Fine-tuning Check** - Verify training API availability
6. **Visualization** - Generate comparison charts

### Hardware Configuration

Benchmarks run on available hardware (CPU or GPU):
- GPU: CUDA-capable device (if available)
- CPU: Fallback for systems without GPU
- Image size: 640×640 (standard for YOLO models)

## 🎨 Sample Visualizations

The comparison generates professional publication-quality plots:

1. **Latency Comparison** - Bar charts for inference time and FPS
2. **Accuracy Comparison** - Grouped bars for mAP metrics
3. **Efficiency Frontier** - Scatter plot showing accuracy/speed trade-offs
4. **Model Size** - Comparison of parameters and memory footprint
5. **Summary Table** - Comprehensive metric overview

## 🔍 Understanding Results

### Choosing the Best Model

- **Highest Accuracy**: Select model with best mAP
- **Fastest Speed**: Select model with highest FPS/lowest latency
- **Best Balance**: Use efficiency score (geometric mean of mAP and FPS)
- **Smallest Size**: For edge deployment, choose minimal parameters

### Trade-off Considerations

- **YOLOv8n/s**: Best for edge devices (low memory)
- **YOLOv10**: Best for consistent low latency
- **YOLOv11**: Best overall balance
- **RT-DETR**: Best for complex scenes requiring global context
- **RF-DETR-N**: Best for ultra-low latency with strong accuracy (2.32ms)
- **RF-DETR-M**: Best for highest accuracy (54.7% mAP) with real-time speed

## 🛠️ Fine-tuning Models

All compared models support fine-tuning on custom datasets:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov11n.pt')

# Fine-tune on custom dataset (YOLO format)
model.train(
    data='custom_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 📚 References & Sources

- [Best Object Detection Models 2025 - Roboflow](https://blog.roboflow.com/best-object-detection-models/)
- [RF-DETR: SOTA Real-Time Object Detection Model - Roboflow](https://blog.roboflow.com/rf-detr/)
- [RF-DETR GitHub Repository](https://github.com/roboflow/rf-detr)
- [Ultralytics Model Comparisons](https://docs.ultralytics.com/compare/)
- [YOLOv8 vs YOLOv10 Comparison](https://docs.ultralytics.com/compare/yolov8-vs-yolov10/)
- [RTDETRv2 vs YOLOv10 Technical Comparison](https://docs.ultralytics.com/compare/rtdetr-vs-yolov10/)
- [YOLOv10 GitHub Repository](https://github.com/THU-MIG/yolov10)
- [RF100-VL Benchmark](https://rf100-vl.org/)
- [Roboflow 100 Dataset](https://www.rf100.org/)

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Use smaller models or CPU mode
python compare_models.py --models yolov8n yolov10n --no-gpu
```

**COCO Dataset Download**
- First run downloads COCO val2017 (~1GB)
- Automatic via Ultralytics
- Stored in `~/.cache/ultralytics`

**Import Errors**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt --upgrade
```

## 🤝 Contributing

This is a research project for comparing object detection models. To extend:

1. Add new models in `src/model_loader.py`
2. Implement custom benchmarks in `src/benchmark.py`
3. Create new visualization types in `src/visualize.py`
4. Write tests following TDD approach

## 📄 License

This project uses open-source models and frameworks:
- Ultralytics (AGPL-3.0)
- PyTorch (BSD)
- OpenCV (Apache 2.0)

## ✨ Acknowledgments

- Ultralytics team for YOLO series
- THU-MIG for YOLOv10
- Baidu for RT-DETR
- Roboflow for RF100-VL benchmark
- Microsoft for COCO dataset

---

**Note**: This comparison focuses on low-latency models suitable for real-time applications. For highest accuracy regardless of speed, consider larger models like YOLOv8x, YOLOv10x, or RT-DETRv2-x.
