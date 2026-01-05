Delivering a thorough head-to-head analysis of modern object detection models, this project benchmarks YOLOv8, YOLOv10, YOLOv11, RT-DETR, and RF-DETR across both COCO and RF100-VL datasets, targeting low-latency, near real-time deployment. Performance metrics such as mean Average Precision (mAP), frames per second (FPS), and inference latency are compared graphically, helping pinpoint optimal models for balanced speed and accuracy. Notably, transformer-based RF-DETR models lead in COCO mAP, while YOLOv10 sets a new standard in low and stable latency. All evaluated models support fine-tuning for custom domains, with results and code accessible for reproducible experimentation.

Key findings:
- [RF-DETR](https://github.com/roboflow/rf-detr) achieves the highest accuracy (54.7% mAP) at real-time speeds; RF-DETR-N reaches 2.32ms latency.
- [YOLOv10](https://github.com/THU-MIG/yolov10) is 1.8× faster than RT-DETR-R18 with similar accuracy and offers stable latency.
- YOLOv8 combines proven stability and documentation, while RT-DETR excels in complex scene understanding.
- All models support batch detection, automated annotation, and robust visualization for efficient workflow integration.
