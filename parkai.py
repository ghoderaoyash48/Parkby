import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load your trained YOLOv8 model
model = YOLO("D:/projectfolders/ParkAI/vehicle_detector/weights/best.pt")

# Class names for your model - these should match your training classes
class_names = ['bike', 'bus', 'car', 'truck']

# Define maximum parking capacity
MAX_PARKING_SPACES = 78

# Capture video from camera
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# For FPS calculation
prev_time = 0
new_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    new_time = time.time()
    fps = 1 / (new_time - prev_time) if prev_time > 0 else 0
    prev_time = new_time
    
    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Process detections
    vehicle_count = 0
    
    # Get the result of the first image (only one image per iteration)
    result = results[0]
    
    # Draw boxes and count vehicles
    for box in result.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get class and confidence
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Only count if confidence is high enough
        if conf > 0.5:  
            # All detected classes are vehicles in this case
            vehicle_count += 1
            
            # Draw bounding box
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate available parking spaces
    available_parking = MAX_PARKING_SPACES - vehicle_count
    available_parking = max(0, available_parking)  # Ensure it doesn't go negative
    
    # Display information
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Available Parking: {available_parking}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Parking Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()