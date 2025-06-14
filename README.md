1. Dataset Preparation
Data Collection: Gathered from open datasets like BDD100K, UA-DETRAC, Indian traffic datasets, and custom CCTV/dashcam footage.
Preprocessing: Standardized image size to 640×640; removed duplicates, blurred frames, and ensured class balance.

Annotation:

Manual: Using LabelImg and CVAT.
Auto: With Roboflow, OpenCV, and weak detectors.
YOLO Format Conversion: All annotations converted to YOLO format for training compatibility.

2. Model Training
Model Used: YOLOv8n and YOLOv8s (for a balance of speed and accuracy).

Training Setup:
Trained on NVIDIA RTX 3060 GPU.
Image size: 640×640, batch size 16–32.
50–100 epochs with cosine learning rate decay and mixed precision.

Dataset Split: 70% training, 20% validation, 10% testing — with class balance.

Performance:
Achieved mAP@0.5 ≈ 72%, Precision: 75%, Recall: 68%.
Real-time performance: ~30–40 FPS.

3. Pilot Implementation (IT Parking Zone – RCOEM)
Location: RCOEM’s IT Parking Zone selected for its infrastructure and camera visibility.

Camera Setup:
2 HD CCTV cameras installed, covering the entire area.
Connected to an NVR (Network Video Recorder) managed by MIS department.

Integration:
NVR linked to system using secure Private Protocol (P-Protocol).
Real-time video streamed to YOLOv8 model for processing.

Features:
Vehicle Detection: Live detection of parked and moving vehicles.
Slot Availability Calculation: Automatic calculation of available/occupied slots.
Student Display: Real-time parking status shown on an LED screen at entrance.
Admin Dashboard: Admin-facing GUI for monitoring parking, detection, and camera feeds.

Outcome:
Successfully demonstrated real-time, automated smart parking management.
Enhanced efficiency and reduced manual effort in parking operations.
