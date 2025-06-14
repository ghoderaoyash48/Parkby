import cv2
import numpy as np
from ultralytics import YOLO
import time

def main():
    print("Initializing vehicle detection system with YOLOv8 and YOLOv3...")
    print("Focus mode: Bikes and motorcycles")
    print("\nControls:")
    print("  - Press 'q': Quit the application")
    print("  - Press '+': Increase YOLOv8 confidence threshold")
    print("  - Press '-': Decrease YOLOv8 confidence threshold")
    print("  - Press '.': Increase YOLOv3 confidence threshold")
    print("  - Press ',': Decrease YOLOv3 confidence threshold")
    print("  - Press 'b': Toggle bike/motorcycle focus mode")
    
    # Load YOLOv8 model
    yolov8_model = YOLO('yolov8n.pt')  # Can be replaced with your trained model
    
    # Load YOLOv3 model using OpenCV DNN
    yolov3_net = cv2.dnn.readNet("D:\\projectfolders\\YOLO\\yolov3.weights", "D:\\projectfolders\\YOLO\\yolov3.cfg")
    layer_names = yolov3_net.getLayerNames()
    unconnected_out_layers = yolov3_net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    
    # Load COCO class names for YOLOv3
    with open("D:\\projectfolders\\YOLO\\coco.names", "r") as f:
        yolov3_classes = [line.strip() for line in f.readlines()]
    
    # Define vehicle classes for both models
    vehicle_classes_yolov8 = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    vehicle_classes_yolov3 = ['car', 'truck', 'bus', 'motorbike', 'bicycle']
    
    # Focus classes (bikes and motorcycles)
    bike_classes_yolov8 = ['motorcycle', 'bicycle']
    bike_classes_yolov3 = ['motorbike', 'bicycle']
    
    # Define initial confidence thresholds (can be adjusted during runtime)
    conf_threshold_yolov8 = 0.25
    conf_threshold_yolov3 = 0.4
    nms_threshold = 0.4
    
    # Flag for bike focus mode
    bike_focus_mode = False
    
    # Define maximum parking capacity
    MAX_PARKING_SPACES = 78
    
    # Open video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # For FPS calculation
    prev_time = 0
    
    # Function to calculate IoU (Intersection over Union)
    def calculate_iou(box1, box2):
        # box format: (x1, y1, x2, y2)
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Get frame dimensions
        height, width, channels = frame.shape
        
        # Store all detected vehicles (to avoid duplicates)
        all_detected_vehicles = []
        
        # ------------------- YOLOv8 Detection --------------------
        # Determine which classes to focus on
        target_classes_yolov8 = bike_classes_yolov8 if bike_focus_mode else vehicle_classes_yolov8
        
        # Run YOLOv8 inference
        yolov8_results = yolov8_model(frame, conf=conf_threshold_yolov8, verbose=False)
        
        # Process YOLOv8 results
        for r in yolov8_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = yolov8_model.names[cls_id]
                
                # Apply lower threshold for bikes when in bike focus mode
                effective_threshold = conf_threshold_yolov8
                if bike_focus_mode and cls_name in bike_classes_yolov8:
                    effective_threshold *= 0.7  # Lower threshold for bikes in focus mode
                
                conf = box.conf[0].item()
                if cls_name in target_classes_yolov8 and conf >= effective_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Bikes get priority in bike focus mode
                    priority = 2 if bike_focus_mode and cls_name in bike_classes_yolov8 else 1
                    
                    # Add to detected vehicles list with model identifier and priority
                    all_detected_vehicles.append((cls_name, conf, (x1, y1, x2, y2), "YOLOv8", priority))
        
        # ------------------- YOLOv3 Detection --------------------
        # Determine which classes to focus on
        target_classes_yolov3 = bike_classes_yolov3 if bike_focus_mode else vehicle_classes_yolov3
        
        # Prepare image for YOLOv3
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolov3_net.setInput(blob)
        outs = yolov3_net.forward(output_layers)
        
        # Process YOLOv3 results
        yolov3_boxes = []
        yolov3_confidences = []
        yolov3_class_ids = []
        yolov3_priorities = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Apply lower threshold for bikes when in bike focus mode
                effective_threshold = conf_threshold_yolov3
                class_name = yolov3_classes[class_id] if class_id < len(yolov3_classes) else ""
                
                if bike_focus_mode and class_name in bike_classes_yolov3:
                    effective_threshold *= 0.7  # Lower threshold for bikes in focus mode
                
                if confidence > effective_threshold and class_name in target_classes_yolov3:
                    # Get coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Bikes get priority in bike focus mode
                    priority = 2 if bike_focus_mode and class_name in bike_classes_yolov3 else 1
                    
                    yolov3_boxes.append([x, y, w, h])
                    yolov3_confidences.append(float(confidence))
                    yolov3_class_ids.append(class_id)
                    yolov3_priorities.append(priority)
        
        # Apply non-maximum suppression to YOLOv3 detections
        indexes = cv2.dnn.NMSBoxes(yolov3_boxes, yolov3_confidences, effective_threshold, nms_threshold) if len(yolov3_boxes) > 0 else []
        
        # Convert indexes to a list if needed for Python versions where it returns a numpy array
        if isinstance(indexes, np.ndarray):
            indexes = indexes.flatten()
            
        for i in range(len(yolov3_boxes)):
            if i in indexes:
                x, y, w, h = yolov3_boxes[i]
                x2, y2 = x + w, y + h
                class_name = yolov3_classes[yolov3_class_ids[i]] if yolov3_class_ids[i] < len(yolov3_classes) else "unknown"
                confidence = yolov3_confidences[i]
                priority = yolov3_priorities[i]
                
                # Add to detected vehicles list with model identifier and priority
                all_detected_vehicles.append((class_name, confidence, (x, y, x2, y2), "YOLOv3", priority))
        
        # ------------- Process All Detections -------------
        # Sort by priority (highest first), then by confidence
        all_detected_vehicles.sort(key=lambda x: (x[4], x[1]), reverse=True)
        
        # Apply custom NMS across both models
        final_detections = []
        is_kept = [True] * len(all_detected_vehicles)
        
        for i in range(len(all_detected_vehicles)):
            if not is_kept[i]:
                continue
                
            for j in range(i + 1, len(all_detected_vehicles)):
                if not is_kept[j]:
                    continue
                    
                iou = calculate_iou(all_detected_vehicles[i][2], all_detected_vehicles[j][2])
                if iou > 0.5:  # IOU threshold
                    is_kept[j] = False  # Remove the lower priority/confidence detection
        
        # Keep only the non-suppressed detections
        for i, detection in enumerate(all_detected_vehicles):
            if is_kept[i]:
                final_detections.append(detection)
        
        # ------------- Display Results -------------
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # Draw boxes and labels
        for cls_name, conf, (x1, y1, x2, y2), model_name, priority in final_detections:
            # Choose color based on class and model
            if cls_name in ['motorcycle', 'bicycle', 'motorbike']:
                if model_name == "YOLOv8":
                    color = (0, 255, 255)  # Yellow for YOLOv8 bikes
                else:
                    color = (255, 255, 0)  # Cyan for YOLOv3 bikes
            else:
                if model_name == "YOLOv8":
                    color = (0, 255, 0)  # Green for YOLOv8
                else:
                    color = (0, 0, 255)  # Red for YOLOv3
                
            # Make box thicker for bikes in focus mode
            thickness = 3 if priority > 1 else 2
                
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with confidence and model
            label = f"{cls_name} {conf:.2f} ({model_name})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        # Calculate vehicle count and available parking
        vehicle_count = len(final_detections)
        bike_count = sum(1 for d in final_detections if d[0] in ['motorcycle', 'bicycle', 'motorbike'])
        car_count = vehicle_count - bike_count
        
        available_parking = MAX_PARKING_SPACES - vehicle_count
        available_parking = max(0, available_parking)  # Ensure it doesn't go negative
        
        # Display information on frame
        cv2.putText(frame, f"Vehicle Count: {vehicle_count} (Cars: {car_count}, Bikes: {bike_count})", 
                  (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Available Parking: {available_parking}", 
                  (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", 
                  (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display model stats and confidence thresholds
        yolov8_count = sum(1 for d in final_detections if d[3] == "YOLOv8")
        yolov3_count = sum(1 for d in final_detections if d[3] == "YOLOv3")
        
        cv2.putText(frame, f"YOLOv8: {yolov8_count} | YOLOv3: {yolov3_count}", 
                  (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Conf YOLOv8: {conf_threshold_yolov8:.2f} | YOLOv3: {conf_threshold_yolov3:.2f}", 
                  (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Display mode status
        mode_text = "BIKE FOCUS MODE" if bike_focus_mode else "NORMAL MODE"
        cv2.putText(frame, mode_text, 
                  (width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # Show the frame
        cv2.imshow('Vehicle Detection System', frame)
        
        # Handle key presses for adjusting thresholds
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('+'):
            conf_threshold_yolov8 = min(0.95, conf_threshold_yolov8 + 0.05)
            print(f"YOLOv8 confidence threshold increased to: {conf_threshold_yolov8:.2f}")
        elif key == ord('-'):
            conf_threshold_yolov8 = max(0.05, conf_threshold_yolov8 - 0.05)
            print(f"YOLOv8 confidence threshold decreased to: {conf_threshold_yolov8:.2f}")
        elif key == ord('.'):
            conf_threshold_yolov3 = min(0.95, conf_threshold_yolov3 + 0.05)
            print(f"YOLOv3 confidence threshold increased to: {conf_threshold_yolov3:.2f}")
        elif key == ord(','):
            conf_threshold_yolov3 = max(0.05, conf_threshold_yolov3 - 0.05)
            print(f"YOLOv3 confidence threshold decreased to: {conf_threshold_yolov3:.2f}")
        elif key == ord('b'):
            bike_focus_mode = not bike_focus_mode
            mode_str = "ENABLED" if bike_focus_mode else "DISABLED"
            print(f"Bike focus mode {mode_str}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Vehicle detection system stopped")

if __name__ == '__main__':
    main()