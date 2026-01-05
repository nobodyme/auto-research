Evaluating object detection models for real-time, low-latency scenarios, this project benchmarks YOLOv8, YOLOv10, YOLOv11, and RT-DETR using both COCO and RF100-VL datasets. Each model is assessed for accuracy (mAP), inference speed (latency, FPS), memory footprint, and versatility (fine-tuning, batch processing). Notably, YOLOv10 achieves the best latency-to-accuracy ratio, outperforming transformer-based RT-DETR in speed while remaining competitive in precision, whereas RT-DETR excels in handling complex scenes. The methodology incorporates robust benchmarking scripts, automated annotation, and visualization for direct metric comparison, supporting both PyTorch and CUDA environments. Full results and model management features are available via [Ultralytics comparison tools](https://docs.ultralytics.com/compare/) and the [YOLOv10 repository](https://github.com/THU-MIG/yolov10).

**Key findings:**
- YOLOv10 offers up to 1.8× faster inference than RT-DETR with similar accuracy, and uses significantly fewer parameters.
- YOLOv8 remains a balanced baseline for speed, accuracy, and ease of deployment.
- RT-DETR is preferred for complex detection tasks requiring global context, despite higher latency.
- All models support transfer learning, enabling adaptation to custom datasets and domains.
