# Real-Time Object Detection and Classification (Team Coconut)

This repository contains the source code and documentation for a high-performance real-time object detection system developed by **Team Coconut**. The project leverages state-of-the-art deep learning frameworks to balance inference speed and accuracy for time-critical applications.

## 👥 Team Members
* **Shuqin Dai**
* **Yuan Fan**

---

## 📝 Project Overview

### Abstract
This project focuses on the development of a system for real-time object detection and classification in dynamic video streams. By utilizing the **YOLO (You Only Look Once)** framework and training on the **MS COCO dataset**, the system identifies multiple object classes under varying environmental conditions. Key performance metrics include **mean Average Precision (mAP)** and **Frames Per Second (FPS)**.

### Problem Definition
Modern autonomous systems (e.g., autonomous vehicles, automated surveillance) require visual data processing with minimal latency. This project aims to improve public safety and operational efficiency by providing reliable, high-speed detections to guide high-stakes decision-making, such as pedestrian-triggered braking.

---

## 🛠 Technical Approach

### Methodology
* **Dataset**: Use the **YOLO format** with the [MS COCO Dataset](https://cocodataset.org/), focusing on real-time inference and accuracy.
* **Architectures**: Deployment of [YOLOv8](https://github.com/ultralytics/ultralytics) and [YOLOv10](https://arxiv.org/abs/2405.14458) for efficient one-stage detection.
* **ML Morphism**: The system maps image tensors $X \in \mathbb{R}^{640 \times 640 \times 3}$ to bounding box vectors $Y = [x, y, w, h, conf, class]$ using a **CSPDarknet** backbone.

### Innovations & Optimization
* **Data Preprocessing**: Resizing with letterboxing and data augmentation (Mosaic/MixUp).
* **Quality Control**: Laplacian Variance checks to filter input frames with high motion blur.
* **Model Pruning**: Systematic removal of redundant neurons to optimize performance for resource-constrained edge-computing devices.
* **Latency Reduction**: Integration of YOLOv10 for NMS-free (Non-Maximum Suppression) inference.
## Processed Videos

https://drive.google.com/drive/folders/1PyxrEbPvkUkjqstuR1rltzoJ-FqmLucx?usp=drive_link

---
