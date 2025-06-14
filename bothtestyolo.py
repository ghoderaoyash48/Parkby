import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import threading
import datetime

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
    print("  - Press 'e': Toggle image enhancement")
    print("  - Press 'r': Force reconnection to camera")
    print("  - Press 'n': Try next RTSP URL format")
    print("  - Press 'f': Toggle faster processing mode")
    print("  - Press 'd': Toggle dashboard view")
    
    # CCTV Camera information
    IP_ADDRESS = "172.16.101.193"  # Replace with your CCTV camera IP
    USERNAME = "admin"             # Replace with your CCTV username
    PASSWORD = "SMART@123"         # Replace with your CCTV password
    
    # Different RTSP URL formats to try
    RTSP_URLS = [
        f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/0",
        f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/Streaming/Channels/101",  # Hikvision format
        f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/cam/realmonitor?channel=1&subtype=0",  # Dahua format
        f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/live/ch0",  # Another common format
        f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/1",  # Simplified format
    ]
    
    # Start with first URL
    RTSP_URL = RTSP_URLS[0]
    rtsp_url_index = 0
    
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    yolov8_model = YOLO('yolov8n.pt')  # Can be replaced with your trained model
    
    # Define YOLOv3 model path
    MODEL_PATH = "D:\\projectfolders\\YOLO\\"
    CONFIG_FILE = os.path.join(MODEL_PATH, "yolov3.cfg")
    WEIGHTS_FILE = os.path.join(MODEL_PATH, "yolov3.weights")
    CLASSES_FILE = os.path.join(MODEL_PATH, "coco.names")
    
    # Check if YOLOv3 files exist
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        return
    if not os.path.exists(WEIGHTS_FILE):
        print(f"ERROR: Weights file not found: {WEIGHTS_FILE}")
        return
    if not os.path.exists(CLASSES_FILE):
        print(f"ERROR: Classes file not found: {CLASSES_FILE}")
        return
    
    print("All model files found. Loading YOLOv3 model...")
    
    # Load YOLOv3 model using OpenCV DNN
    try:
        yolov3_net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)
        # Explicitly set CPU backend to avoid CUDA issues
        yolov3_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        print("YOLOv3 network loaded successfully with CPU backend")
        
        layer_names = yolov3_net.getLayerNames()
        try:
            unconnected_out_layers = yolov3_net.getUnconnectedOutLayers()
            if isinstance(unconnected_out_layers, np.ndarray):
                output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
            else:
                output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        except:
            # Alternative approach for different OpenCV versions
            unconnected_out_layers = yolov3_net.getUnconnectedOutLayers()
            output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    except Exception as e:
        print(f"ERROR loading YOLOv3 network: {e}")
        return
    
    # Load COCO class names for YOLOv3
    with open(CLASSES_FILE, "r") as f:
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
    
    # Flag for bike focus mode and image enhancement
    bike_focus_mode = False
    enhance_mode = False
    fast_mode = False  # Flag for faster processing
    dashboard_mode = False  # Flag for dashboard view
    
    # Stream health monitoring
    stream_health_threshold = 10  # Number of consecutive bad frames to trigger reconnect
    bad_frame_counter = 0
    last_reconnect_time = time.time()
    min_reconnect_interval = 10  # Minimum seconds between reconnection attempts
    
    # Define maximum parking capacity
    MAX_PARKING_SPACES = 150
    
    # Setup buffer for threaded reading
    frame_buffer = None
    frame_buffer_lock = threading.Lock()
    frame_ready = False
    stop_thread = False
    
    # Data for dashboard
    dashboard_update_interval = 20  # Update dashboard every 20 seconds
    last_dashboard_update = time.time()
    current_vehicle_count = 0
    current_available_spaces = MAX_PARKING_SPACES
    parking_percentage = 0
    
        # Function to create dashboard
    def create_dashboard(width, height, status, available_slots, percentage_filled, date_str):
        """Create a dashboard frame based on the template provided"""
        # Create blank frame
        dashboard = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw border rectangle
        cv2.rectangle(dashboard, (0, 0), (width-1, height-1), (0, 0, 0), 2)
        
        # Draw IT PARKING header
        header_height = height // 8
        cv2.rectangle(dashboard, (0, 0), (width, header_height), (0, 125, 255), -1)  # Orange background
        cv2.putText(dashboard, "IT PARKING", (width//4, header_height-15), 
                   cv2.FONT_HERSHEY_DUPLEX, 2.5, (25, 25, 112), 4)  # Dark blue text
        
        # Status section
        status_section_height = height // 4
        cv2.rectangle(dashboard, (0, header_height), (width, header_height + status_section_height), 
                     (255, 255, 255), -1)  # White background
        
        # Yellow background for status text
        yellow_bg_margin = width // 6
        cv2.rectangle(dashboard, (yellow_bg_margin, header_height + 10), 
                     (width - yellow_bg_margin, header_height + status_section_height - 10), 
                     (255, 255, 160), -1)  # Light yellow
        
        # Status text (OPEN in green or CLOSE in red)
        if status == "OPEN":
            status_color = (0, 200, 0)  # Green
        else:
            status_color = (0, 0, 255)  # Red
            
        cv2.putText(dashboard, status, (width//3, header_height + status_section_height//2 + 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 5, status_color, 7)
        
        # Progress bar section
        progress_start_y = header_height + status_section_height
        progress_height = height // 8
        
        # Progress bar background
        cv2.rectangle(dashboard, (0, progress_start_y), (width, progress_start_y + progress_height), 
                     (255, 255, 255), -1)
        
        # Red part of the progress bar (used spaces)
        red_width = int((width * percentage_filled) / 100)
        cv2.rectangle(dashboard, (0, progress_start_y), (red_width, progress_start_y + progress_height), 
                     (0, 0, 255), -1)
        
        # Green part of the progress bar (available spaces) - only if it's open
        if status == "OPEN":
            green_start = red_width
            cv2.rectangle(dashboard, (green_start, progress_start_y), 
                         (width - width//8, progress_start_y + progress_height), (0, 255, 0), -1)
        
        # Percentage text
        percentage_box_width = width // 8
        cv2.rectangle(dashboard, (width - percentage_box_width, progress_start_y), 
                     (width, progress_start_y + progress_height), (255, 255, 255), -1)
        cv2.putText(dashboard, f"{percentage_filled}%", 
                   (width - percentage_box_width + 10, progress_start_y + progress_height - 15), 
                   cv2.FONT_HERSHEY_DUPLEX, 2, (255, 125, 0), 3)  # Orange percentage
        
        # Available slots text
        slots_text_y = progress_start_y + progress_height + 80
        cv2.putText(dashboard, f"AVALIABLE SLOTS : {available_slots}", 
                   (width//6, slots_text_y), cv2.FONT_HERSHEY_DUPLEX, 2, (25, 118, 210), 3)  # Blue text
        
        # Footer with date and time
        footer_start_y = height - height//8
        cv2.rectangle(dashboard, (0, footer_start_y), (width, height), 
                     (25, 25, 112), -1)  # Dark blue background
        
        # Current time
        current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
        cv2.putText(dashboard, current_time, (width//20, footer_start_y + height//16), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
        
        # Date
        cv2.putText(dashboard, date_str, (width//2 - 100, footer_start_y + height//16), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)  # Cyan date
        
        # Last update time
        cv2.putText(dashboard, f"Last Update: 10 sec ago", (width - 300, footer_start_y + height//16), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return dashboard
    # Create dashboard frame
    dashboard_width = 800
    dashboard_height = 600
    dashboard_frame = create_dashboard(dashboard_width, dashboard_height, 
                                      "OPEN", 27, 73, "21st April 2025")
    
    # Threading function for reading frames
    def read_frames_thread():
        nonlocal frame_buffer, frame_ready, stop_thread
        
        while not stop_thread:
            if cap.isOpened():
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    with frame_buffer_lock:
                        frame_buffer = frame.copy()
                        frame_ready = True
                else:
                    time.sleep(0.01)  # Small delay to avoid CPU hogging
            else:
                time.sleep(0.1)  # Larger delay if camera is not opened
    

    
    # Function to connect to CCTV
    def connect_to_cctv():
        nonlocal RTSP_URL, rtsp_url_index
        
        # Try each RTSP URL until one works
        for i in range(len(RTSP_URLS)):
            current_index = (rtsp_url_index + i) % len(RTSP_URLS)
            current_url = RTSP_URLS[current_index]
            
            print(f"Trying to connect to CCTV at {current_url}...")
            
            # Configure RTSP transport protocol - try TCP for more stability
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000"
            
            # Additional FFMPEG options for more stability with H.265/HEVC streams
            os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10"  # Number of read attempts
            
            try:
                # Try to capture with explicit FFMPEG backend
                cap = cv2.VideoCapture(current_url, cv2.CAP_FFMPEG)
                
                # Configure capture parameters
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increase buffer size for smoother video
                
                # Try to set lower resolution if supported
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Set timeouts
                try:
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)  # Reduced from 5000
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)  # Reduced from 5000
                except:
                    pass  # Some OpenCV versions don't support these properties
                
                if cap.isOpened():
                    # Try to read a few frames to ensure connection is stable
                    stable_connection = False
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            stable_connection = True
                        else:
                            stable_connection = False
                            break
                        time.sleep(0.1)
                    
                    if stable_connection:
                        print(f"✅ Successfully connected to CCTV camera with TCP: {current_url}")
                        RTSP_URL = current_url
                        rtsp_url_index = current_index
                        return cap
                    else:
                        print("Connected but stream is not stable")
                        cap.release()
                else:
                    print(f"❌ Failed to connect with TCP: {current_url}")
                
                # Try UDP if TCP failed
                print("Trying UDP transport...")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;500000"
                cap = cv2.VideoCapture(current_url, cv2.CAP_FFMPEG)
                
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ Successfully connected to CCTV camera with UDP: {current_url}")
                        RTSP_URL = current_url
                        rtsp_url_index = current_index
                        return cap
                    else:
                        print("Connected with UDP but couldn't get stable stream")
                        cap.release()
                else:
                    print(f"❌ Failed to connect with UDP: {current_url}")
                    
            except Exception as e:
                print(f"Error trying RTSP URL '{current_url}': {e}")
        
        # Move to the next URL for future attempts
        rtsp_url_index = (rtsp_url_index + 1) % len(RTSP_URLS)
        
        # If all RTSP URLs fail, try fallback to webcam
        print("All RTSP URLs failed. Falling back to webcam...")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Connected to webcam")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            else:
                print("❌ Failed to connect to webcam")
        except Exception as e:
            print(f"Error connecting to webcam: {e}")
        
        return None
    
    # Image enhancement function - optimized version
    def enhance_image(image):
        """Apply image enhancement to improve detection quality"""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back the channels
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
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
    
    # Connect to CCTV
    cap = connect_to_cctv()
    if cap is None:
        print("Failed to connect to any video source. Exiting.")
        return
    
    # Create window with trackbar for confidence
    cv2.namedWindow('Vehicle Detection System')
    cv2.namedWindow('Parking Dashboard')
    
    # Track vehicle detection history for stabilization
    vehicle_count_history = []
    
    # For FPS calculation
    prev_time = 0
    frame_count = 0
    start_time = time.time()
    processing_times = []
    
    # Start frame reading thread
    reader_thread = threading.Thread(target=read_frames_thread)
    reader_thread.daemon = True
    reader_thread.start()
    
    # Skip frame counter for faster processing
    frame_skip_count = 0
    
    # Wait for first frame
    time.sleep(1)
    
    while True:
        # Measure processing time
        loop_start = time.time()
        
        # Get frame from buffer (non-blocking)
        frame = None
        with frame_buffer_lock:
            if frame_ready:
                frame = frame_buffer.copy()
                frame_ready = False
        
        # If no frame is available yet, wait a bit
        if frame is None:
            time.sleep(0.01)
            continue
            
        frame_count += 1
        
        # Skip frames for faster processing when in fast mode
        if fast_mode and frame_skip_count % 2 != 0:
            frame_skip_count += 1
            cv2.imshow('Vehicle Detection System', frame)  # Show frame but skip processing
            
            # Always display the dashboard
            cv2.imshow('Parking Dashboard', dashboard_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                fast_mode = not fast_mode
                print(f"Fast mode {'enabled' if fast_mode else 'disabled'}")
            elif key == ord('d'):
                dashboard_mode = not dashboard_mode
                print(f"Dashboard mode {'enabled' if dashboard_mode else 'disabled'}")
            continue
        
        frame_skip_count += 1
        
        # Check for signs of corrupted/blurry frames
        try:
            # Calculate frame clarity using Laplacian variance (detects blurriness)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # If frame is extremely blurry, consider it corrupted
            if clarity < 5.0:  # Lower threshold indicates more blur
                bad_frame_counter += 1
                print(f"Blurry frame detected (clarity: {clarity:.2f}). Counter: {bad_frame_counter}/{stream_health_threshold}")
                
                if bad_frame_counter >= stream_health_threshold // 2:  # More aggressive for blurry frames
                    current_time = time.time()
                    if current_time - last_reconnect_time > min_reconnect_interval:
                        print("Stream quality degraded (too blurry). Reconnecting...")
                        stop_thread = True  # Stop reading thread
                        reader_thread.join(timeout=1.0)  # Wait for thread to end
                        cap.release()
                        time.sleep(1)
                        cap = connect_to_cctv()
                        last_reconnect_time = current_time
                        bad_frame_counter = 0
                        
                        # Reset and restart thread
                        if cap is not None:
                            stop_thread = False
                            reader_thread = threading.Thread(target=read_frames_thread)
                            reader_thread.daemon = True
                            reader_thread.start()
                            time.sleep(1)  # Wait for first frame
                        else:
                            print("Failed to reconnect. Exiting.")
                            break
                        continue
            else:
                # Reset counter for good frames
                bad_frame_counter = max(0, bad_frame_counter - 1)
        except Exception as e:
            print(f"Error checking frame quality: {e}")
        
        # Apply image enhancement if enabled
        if enhance_mode:
            frame = enhance_image(frame)
        
        # Get frame dimensions
        height, width, channels = frame.shape
        
        # Resize frame for faster processing in fast mode
        if fast_mode:
            processing_frame = cv2.resize(frame, (640, 360))  # Smaller size for processing
        else:
            processing_frame = frame
        
        # Store all detected vehicles (to avoid duplicates)
        all_detected_vehicles = []
        
        # ------------------- YOLOv8 Detection --------------------
        # Determine which classes to focus on
        target_classes_yolov8 = bike_classes_yolov8 if bike_focus_mode else vehicle_classes_yolov8
        
        # Run YOLOv8 inference
        yolov8_results = yolov8_model(processing_frame, conf=conf_threshold_yolov8, verbose=False)
        
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
                    
                    # Adjust coordinates if using resized frame
                    if fast_mode:
                        scale_x = width / 640
                        scale_y = height / 360
                        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                    
                    # Bikes get priority in bike focus mode
                    priority = 2 if bike_focus_mode and cls_name in bike_classes_yolov8 else 1
                    
                    # Add to detected vehicles list with model identifier and priority
                    all_detected_vehicles.append((cls_name, conf, (x1, y1, x2, y2), "YOLOv8", priority))
        
        # Skip YOLOv3 in fast mode for better FPS
        if not fast_mode:
            # ------------------- YOLOv3 Detection --------------------
            # Determine which classes to focus on
            target_classes_yolov3 = bike_classes_yolov3 if bike_focus_mode else vehicle_classes_yolov3
            
            # Prepare image for YOLOv3
            blob = cv2.dnn.blobFromImage(processing_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                        center_x = int(detection[0] * processing_frame.shape[1])
                        center_y = int(detection[1] * processing_frame.shape[0])
                        w = int(detection[2] * processing_frame.shape[1])
                        h = int(detection[3] * processing_frame.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Adjust coordinates if using resized frame
                        if fast_mode:
                            scale_x = width / 640
                            scale_y = height / 360
                            x, y = int(x * scale_x), int(y * scale_y)
                            w, h = int(w * scale_x), int(h * scale_y)
                        
                        # Bikes get priority in bike focus mode
                        priority = 2 if bike_focus_mode and class_name in bike_classes_yolov3 else 1
                        
                        yolov3_boxes.append([x, y, w, h])
                        yolov3_confidences.append(float(confidence))
                        yolov3_class_ids.append(class_id)
                        yolov3_priorities.append(priority)
            
            # Apply non-maximum suppression to YOLOv3 detections
            indexes = []
            if len(yolov3_boxes) > 0:
                try:
                    indexes = cv2.dnn.NMSBoxes(yolov3_boxes, yolov3_confidences, effective_threshold, nms_threshold)
                    # Convert indexes to a list if needed
                    if isinstance(indexes, np.ndarray):
                        indexes = indexes.flatten()
                except Exception as e:
                    print(f"Error in NMS: {e}")
                    indexes = []
                
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
        
        # Apply NMS to remove duplicate detections across models
        final_vehicles = []
        while all_detected_vehicles:
            best = all_detected_vehicles.pop(0)  # Get highest priority/confidence detection
            final_vehicles.append(best)
            
            # Remove any overlapping boxes with high IoU
            all_detected_vehicles = [
                det for det in all_detected_vehicles
                if calculate_iou(best[2], det[2]) < 0.45  # Lower threshold to avoid missing close vehicles
            ]
        
        # Count vehicles
        vehicle_count = len(final_vehicles)
        
        # Stabilize count with moving average
        vehicle_count_history.append(vehicle_count)
        if len(vehicle_count_history) > 10:  # Keep history of last 10 frames
            vehicle_count_history.pop(0)
        
        # Calculate stable count with slight emphasis on recent results
        stable_count = int(np.average(vehicle_count_history, weights=range(1, len(vehicle_count_history) + 1)))
        
        # Update vehicle count for dashboard
        current_vehicle_count = stable_count
        
        # Calculate available parking spaces
        current_available_spaces = max(0, MAX_PARKING_SPACES - current_vehicle_count)
        
        # Calculate parking percentage
        parking_percentage = min(100, int((current_vehicle_count / MAX_PARKING_SPACES) * 100))
        
        # Update dashboard if necessary
        current_time = time.time()
        if current_time - last_dashboard_update >= dashboard_update_interval or dashboard_frame is None:
            # Get current date as string
            current_date = datetime.datetime.now().strftime("%dst %B %Y")
            
            # Determine status based on available spaces
            status = "OPEN" if current_available_spaces > 0 else "CLOSE"
            
            # Update dashboard
            dashboard_frame = create_dashboard(dashboard_width, dashboard_height, 
                                             status, current_available_spaces, 
                                             parking_percentage, current_date)
            
            last_dashboard_update = current_time
        
        # Draw detection boxes on frame
        for (cls_name, confidence, bbox, model, _) in final_vehicles:
            x1, y1, x2, y2 = bbox
            
            # Different colors for different models
            color = (0, 255, 0) if model == "YOLOv8" else (0, 165, 255)  # Green for YOLOv8, Orange for YOLOv3
            
            # Special color for bikes in focus mode
            if bike_focus_mode and (cls_name in bike_classes_yolov8 or cls_name in bike_classes_yolov3):
                color = (0, 0, 255)  # Red for bikes in focus mode
            
            # Draw box with filled alpha transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # Apply transparency
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Format label text
            label = f"{cls_name}: {confidence:.2f}"
            
            # Draw background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate FPS
        current_time = time.time()
        frame_processing_time = current_time - loop_start
        processing_times.append(frame_processing_time)
        
        # Keep only recent processing times
        if len(processing_times) > 30:
            processing_times.pop(0)
        
        # Calculate average FPS over recent frames
        avg_processing_time = sum(processing_times) / len(processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        # Add information to frame
        # Create a semi-transparent overlay for the info panel
        info_overlay = frame.copy()
        cv2.rectangle(info_overlay, (0, 0), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(info_overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add system info
        cv2.putText(frame, f"Vehicle Detection System", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"YOLOv8 Conf: {conf_threshold_yolov8:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"YOLOv3 Conf: {conf_threshold_yolov3:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"Bikes Focus: {'ON' if bike_focus_mode else 'OFF'}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if bike_focus_mode else (255, 255, 255), 2)
        cv2.putText(frame, f"Enhancement: {'ON' if enhance_mode else 'OFF'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Fast Mode: {'ON' if fast_mode else 'OFF'}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add vehicle count
        cv2.putText(frame, f"Vehicles: {stable_count}", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Available: {current_available_spaces}/{MAX_PARKING_SPACES}", (width - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the processed frame
        cv2.imshow('Vehicle Detection System', frame)
        
        # Always display the dashboard
        cv2.imshow('Parking Dashboard', dashboard_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            conf_threshold_yolov8 = min(0.95, conf_threshold_yolov8 + 0.05)
            print(f"YOLOv8 confidence threshold increased to {conf_threshold_yolov8:.2f}")
        elif key == ord('-'):
            conf_threshold_yolov8 = max(0.05, conf_threshold_yolov8 - 0.05)
            print(f"YOLOv8 confidence threshold decreased to {conf_threshold_yolov8:.2f}")
        elif key == ord('.'):
            conf_threshold_yolov3 = min(0.95, conf_threshold_yolov3 + 0.05)
            print(f"YOLOv3 confidence threshold increased to {conf_threshold_yolov3:.2f}")
        elif key == ord(','):
            conf_threshold_yolov3 = max(0.05, conf_threshold_yolov3 - 0.05)
            print(f"YOLOv3 confidence threshold decreased to {conf_threshold_yolov3:.2f}")
        elif key == ord('b'):
            bike_focus_mode = not bike_focus_mode
            print(f"Bike/motorcycle focus mode {'enabled' if bike_focus_mode else 'disabled'}")
        elif key == ord('e'):
            enhance_mode = not enhance_mode
            print(f"Image enhancement {'enabled' if enhance_mode else 'disabled'}")
        elif key == ord('r'):
            print("Forcing camera reconnection...")
            stop_thread = True
            reader_thread.join(timeout=1.0)
            cap.release()
            time.sleep(1)
            cap = connect_to_cctv()
            last_reconnect_time = time.time()
            bad_frame_counter = 0
            
            if cap is not None:
                stop_thread = False
                reader_thread = threading.Thread(target=read_frames_thread)
                reader_thread.daemon = True
                reader_thread.start()
                time.sleep(1)
            else:
                print("Failed to reconnect. Exiting.")
                break
        elif key == ord('n'):
            # Try next RTSP URL format
            rtsp_url_index = (rtsp_url_index + 1) % len(RTSP_URLS)
            print(f"Switching to next RTSP URL format: {RTSP_URLS[rtsp_url_index]}")
            stop_thread = True
            reader_thread.join(timeout=1.0)
            cap.release()
            time.sleep(1)
            cap = connect_to_cctv()
            last_reconnect_time = time.time()
            if cap is not None:
                stop_thread = False
                reader_thread = threading.Thread(target=read_frames_thread)
                reader_thread.daemon = True
                reader_thread.start()
                time.sleep(1)
            else:
                print("Failed to reconnect. Exiting.")
                break
        elif key == ord('f'):
            fast_mode = not fast_mode
            print(f"Fast mode {'enabled' if fast_mode else 'disabled'}")
        elif key == ord('d'):
            dashboard_mode = not dashboard_mode
            print(f"Dashboard mode {'enabled' if dashboard_mode else 'disabled'}")
            # Force dashboard update
            last_dashboard_update = 0
    
    # Clean up
    stop_thread = True
    if reader_thread.is_alive():
        reader_thread.join(timeout=1.0)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Vehicle detection system shutdown successfully.")

if __name__ == "__main__":
    main()