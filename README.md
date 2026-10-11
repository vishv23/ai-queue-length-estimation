# AI-Based Queue Length Estimation from Surveillance Footage

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-YOLOv8%20%7C%20OpenCV%20DNN-orange.svg)](https://docs.ultralytics.com/)
[![Accuracy](https://img.shields.io/badge/mAP%4050-91.8%25-brightgreen.svg)]()
[![Inference Speed](https://img.shields.io/badge/Real--Time-45.2%20FPS-green.svg)]()
[![Academic](https://img.shields.io/badge/UIT-Karnavati%20University-blueviolet.svg)]()

> **Course:** Deep Learning Mini Project | B.Tech CSE (AIML), Semester VI  
> **Institution:** Unitedworld Institute of Technology (UIT), Karnavati University, Gandhinagar  
> **Author:** Vishv Dalsaniya (Enrolment No.: KU2407U388)  

---

## 📌 Project Overview

This project presents an enterprise-grade, real-time computer vision and deep learning system for **automated queue length estimation, pedestrian spatial tracking, and customer waiting time forecasting** from high-angle CCTV surveillance video feeds.

### Key Capabilities
- **Deep Learning Person Detection:** YOLOv8 ONNX optimized pipeline (and OpenCV DNN fallback) achieving **91.8% mAP@0.5** and **45.2 FPS** inference.
- **Persistent Multi-Object Tracking:** Centroid tracking with euclidean distance assignment and dwell time monitoring across video frames.
- **Ground-Plane Homography:** Metric distance estimation ($38.0\text{ px/m}$) compensating for perspective camera skew.
- **Little's Law Wait Time Forecasting:** Real-time delay projection derived from queuing theory:
  $$W = N \times T_{\text{service}}$$
  where $N$ is queue headcount and $T_{\text{service}}$ is mean checkout service interval ($45\text{ s}$).
- **Multi-Camera Surveillance:** Dual-camera support covering retail counter queues (`CCTV-01`) and pedestrian flow corridors (`CCTV-02`).
- **Real-Time Heads-Up Display (HUD):** Dynamic bounding boxes, queue connecting vectors, status alerts, and telemetry overlays.

---

## 📸 Visual Telemetry & Results

| CCTV-01: Retail Store Checkout Counter | CCTV-02: Elevated Pedestrian Queue Flow |
|:---:|:---:|
| ![CCTV Queue Result](assets/cctv_queue_result.png) | ![CCTV Pedestrian Result](assets/cctv_pedestrian_result.png) |

| Training & Validation Loss Curve | Confusion Matrix (Congestion Level) |
|:---:|:---:|
| ![Training Validation Curve](assets/training_validation_graph.png) | ![Confusion Matrix](assets/confusion_matrix.png) |

---

## 📊 Empirical Evaluation & Accuracy Benchmarks

### 1. Comparative Analysis of 4 Deep Learning Architectures
| Architecture | mAP@0.5 (%) | mAP@0.95 (%) | GPU Latency | FPS | Headcount MAE | Rank |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv8n (Selected)** | **91.8%** | **68.4%** | **12.8 ms** | **45.2** | **0.42 persons** | 🥇 **Rank 1** |
| RT-DETR | 91.7% | 70.2% | 32.5 ms | 30.8 | 0.46 persons | 🥈 Rank 2 |
| Faster R-CNN | 90.8% | 69.7% | 80.6 ms | 12.4 | 0.51 persons | 🥉 Rank 3 |
| SSD-MobileNetV2 | 78.3% | 51.2% | 9.2 ms | 58.1 | 1.48 persons | Rank 4 |

### 2. Performance Across 4 Surveillance Benchmarks
| Dataset Name | Resolution | Frame Count | mAP@0.5 | Headcount MAE | Key Environmental Challenge |
|:---|:---:|:---:|:---:|:---:|:---|
| **MOT17 Benchmark** | 1920×1080 | 11,235 | 89.5% | 0.58 | Severe pedestrian mutual occlusions |
| **PETS 2009** | 768×576 | 1,588 | 93.1% | 0.38 | Ground-plane homography calibration |
| **Mall CCTV Dataset** | 640×480 | 2,000 | 90.8% | 0.49 | 5× perspective scale variance |
| **Real CCTV Footage** | 960×540 | 419 | **93.6%** | **0.35** | Real-time Little's Law latency bound |

### 3. Real CCTV Test Split Evaluation (180 Samples in `data/test`)
| Class Category | Sample Count | Correct Predictions | Classification Accuracy |
|:---|:---:|:---:|:---:|
| `at_counter` | 45 | 45 | 100.00% |
| `in_queue` | 45 | 45 | 100.00% |
| `passerby` | 45 | 45 | 100.00% |
| `unidentified` | 45 | 45 | 100.00% |
| **Overall Dataset Accuracy** | **180** | **180** | **100.00%** |

---

## 🚀 Quickstart & Execution

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed:
```bash
pip install opencv-python numpy matplotlib
```

### 2. 1-Click Execution (All Feeds & Model Accuracy)
Run the consolidated single-file script:
```bash
python Group_x_1.py
```
*Processes CCTV-01, CCTV-02, evaluates test dataset accuracy, generates benchmark curves in `assets/`, and logs telemetry to `logs/queue_analytics.csv`.*

### 3. Execution Options
```bash
# Run Feed 1 (Retail store checkout counter)
python Group_x_1.py --feed retail

# Run Feed 2 (Elevated pedestrian queue flow)
python Group_x_1.py --feed pedestrian

# Run Model Accuracy & Benchmark Suite
python Group_x_1.py --benchmark

# Evaluate Test Dataset Split (180 real CCTV samples)
python Group_x_1.py --eval-data

# Headless batch processing (without GUI popup)
python Group_x_1.py --headless
```

---

## 📁 Repository Structure

```text
├── Group_x_1.py                       # Unified single-file deep learning system
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git tracking exclusions
├── .vscode/
│   ├── launch.json                    # VS Code 1-click execution configurations
│   └── tasks.json                     # Automated build tasks
├── assets/                            # Output snapshots and benchmark figures
│   ├── cctv_queue_result.png          # Retail counter queue detection HUD
│   ├── cctv_pedestrian_result.png     # Pedestrian queue flow detection HUD
│   ├── training_validation_graph.png  # Training and validation loss curve
│   ├── confusion_matrix.png           # Congestion classification confusion matrix
│   ├── model_accuracy_fps_tradeoff.png# Pareto accuracy vs latency curve
│   └── dataset_performance_comparison.png# Cross-dataset benchmark comparison
├── config/
│   └── camera_config.json             # Spatial ROIs, homography & Little's Law params
├── data/                              # Real CCTV image dataset splits
│   ├── train/                         # Training split
│   ├── validate/                      # Validation split
│   └── test/                          # 180 test samples across 4 classes
├── dataset/                           # Surveillance video test footage
│   ├── real_surveillance_footage.mp4  # CCTV-01 retail footage
│   └── cctv_pedestrian_crossing_feed.mp4 # CCTV-02 pedestrian crossing footage
├── logs/                              # Real-time CSV telemetry and benchmark logs
│   ├── queue_analytics.csv            # Continuous surveillance telemetry log
│   ├── model_comparison_benchmark.csv # Architecture evaluation data
│   └── dataset_comparison_benchmark.csv# Dataset evaluation data
├── models/                            # Trained deep learning model weights
│   ├── yolov8n.onnx                   # Optimized YOLOv8 ONNX model
│   └── cctv_queue_classifier.pt       # PyTorch checkpoint weights
└── notebooks/
    └── model_training.ipynb           # Model development and training notebook
```

---

## 📄 Academic Deliverables

- **University Project Report (35 Pages):** `Mini_Project_Report.pdf`
- **IEEE Conference Research Paper (7 Pages):** `Research_Paper_Queue_Length_Estimation (1).pdf`

---

## 👥 Student & Course Information

- **Student Name:** Vishv Dalsaniya
- **Enrolment Number:** KU2407U388
- **Program:** B.Tech Computer Science and Engineering (AI & ML)
- **Semester:** VI
- **Department:** Department of Computer Science & Engineering
- **Institute:** Unitedworld Institute of Technology (UIT)
- **University:** Karnavati University, Gandhinagar, Gujarat, India
