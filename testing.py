import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import threading
import datetime

def main():
    print("Initializing dual camera vehicle detection system with YOLOv8 and YOLOv3...")
    print("Focus mode: Bikes and motorcycles")
    print("\nControls:")
    print(" - Press 'q': Quit the application")
    print(" - Press '+': Increase YOLOv8 confidence threshold for active camera")
    print(" - Press '-': Decrease YOLOv8 confidence threshold for active camera")
    print(" - Press '.': Increase YOLOv3 confidence threshold for active camera")
    print(" - Press ',': Decrease YOLOv3 confidence threshold for active camera")
    print(" - Press 'b': Toggle bike/motorcycle focus mode for active camera")
    print(" - Press 'e': Toggle image enhancement for active camera")
    print(" - Press 'r': Force reconnection to cameras")
    print(" - Press 'n': Try next RTSP URL format for active camera")
    print(" - Press 'f': Toggle faster processing mode for active camera")
    print(" - Press 'd': Toggle dashboard view")
    print(" - Press '1': Switch active control to Camera 1")
    print(" - Press '2': Switch active control to Camera 2")

    # CCTV Camera information for first camera
    IP_ADDRESS_1 = "172.16.101.196"  # First CCTV camera IP
    USERNAME_1 = "admin"  # Replace with your CCTV username
    PASSWORD_1 = "SMART@123"  # Replace with your CCTV password

    # CCTV Camera information for second camera
    IP_ADDRESS_2 = "172.16.101.193"  # Second CCTV camera IP
    USERNAME_2 = "admin"  # Replace with your CCTV username
    PASSWORD_2 = "SMART@123"  # Replace with your CCTV password

    # Different RTSP URL formats to try for camera 1
    RTSP_URLS_1 = [
        f"rtsp://{USERNAME_1}:{PASSWORD_1}@{IP_ADDRESS_1}/0",
        f"rtsp://{USERNAME_1}:{PASSWORD_1}@{IP_ADDRESS_1}:554/Streaming/Channels/101",  # Hikvision format
        f"rtsp://{USERNAME_1}:{PASSWORD_1}@{IP_ADDRESS_1}:554/cam/realmonitor?channel=1&subtype=0",  # Dahua format
        f"rtsp://{USERNAME_1}:{PASSWORD_1}@{IP_ADDRESS_1}/live/ch0",  # Another common format
        f"rtsp://{USERNAME_1}:{PASSWORD_1}@{IP_ADDRESS_1}:554/1",  # Simplified format
    ]

    # Different RTSP URL formats to try for camera 2
    RTSP_URLS_2 = [
        f"rtsp://{USERNAME_2}:{PASSWORD_2}@{IP_ADDRESS_2}/0",
        f"rtsp://{USERNAME_2}:{PASSWORD_2}@{IP_ADDRESS_2}:554/Streaming/Channels/101",  # Hikvision format
        f"rtsp://{USERNAME_2}:{PASSWORD_2}@{IP_ADDRESS_2}:554/cam/realmonitor?channel=1&subtype=0",  # Dahua format
        f"rtsp://{USERNAME_2}:{PASSWORD_2}@{IP_ADDRESS_2}/live/ch0",  # Another common format
        f"rtsp://{USERNAME_2}:{PASSWORD_2}@{IP_ADDRESS_2}:554/1",  # Simplified format
    ]

    # Start with first URL for both cameras
    RTSP_URL_1 = RTSP_URLS_1[0]
    rtsp_url_index_1 = 0
    RTSP_URL_2 = RTSP_URLS_2[0]
    rtsp_url_index_2 = 0

    # Set the active camera (for keyboard controls)
    active_camera = 1  # Start with camera 1 being active

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

    # Define initial confidence thresholds for each camera (can be adjusted during runtime)
    conf_threshold_yolov8_1 = 0.25
    conf_threshold_yolov3_1 = 0.4
    conf_threshold_yolov8_2 = 0.25
    conf_threshold_yolov3_2 = 0.4
    nms_threshold = 0.4

    # Flag for bike focus mode and image enhancement for each camera
    bike_focus_mode_1 = False
    enhance_mode_1 = False
    fast_mode_1 = False  # Flag for faster processing
    bike_focus_mode_2 = False
    enhance_mode_2 = False
    fast_mode_2 = False  # Flag for faster processing
    dashboard_mode = False  # Flag for dashboard view

    # Stream health monitoring for each camera
    stream_health_threshold = 10  # Number of consecutive bad frames to trigger reconnect
    bad_frame_counter_1 = 0
    bad_frame_counter_2 = 0
    last_reconnect_time_1 = time.time()
    last_reconnect_time_2 = time.time()
    min_reconnect_interval = 10  # Minimum seconds between reconnection attempts

    # Define maximum parking capacity (increased to 200)
    MAX_PARKING_SPACES = 200

    # Setup buffer for threaded reading for each camera
    frame_buffer_1 = None
    frame_buffer_lock_1 = threading.Lock()
    frame_ready_1 = False
    frame_buffer_2 = None
    frame_buffer_lock_2 = threading.Lock()
    frame_ready_2 = False
    stop_thread_1 = False
    stop_thread_2 = False

    # Data for dashboard
    dashboard_update_interval = 20  # Update dashboard every 20 seconds
    last_dashboard_update = time.time()
    current_vehicle_count_1 = 0
    current_vehicle_count_2 = 0
    total_vehicle_count = 0
    current_available_spaces = MAX_PARKING_SPACES
    parking_percentage = 0

    # Function to create dashboard
    def create_dashboard(width, height, status, available_slots, percentage_filled, date_str, count1, count2):
        """Create a dashboard frame with counts from both cameras"""
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
        slots_text_y = progress_start_y + progress_height + 60
        cv2.putText(dashboard, f"AVALIABLE SLOTS : {available_slots}", 
                   (width//6, slots_text_y), cv2.FONT_HERSHEY_DUPLEX, 2, (25, 118, 210), 3)  # Blue text
        
        # Display counts from both cameras
        cv2.putText(dashboard, f"Camera 1: {count1} vehicles", 
                   (width//6, slots_text_y + 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(dashboard, f"Camera 2: {count2} vehicles", 
                   (width//6, slots_text_y + 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
        
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
                   
        # Add active camera indicator
        active_cam_text = f"Active Camera: {active_camera}"
        cv2.putText(dashboard, active_cam_text, (width - 350, 50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
                   
        return dashboard

    # Create dashboard frame
    dashboard_width = 800
    dashboard_height = 600
    dashboard_frame = create_dashboard(dashboard_width, dashboard_height, 
                                     "OPEN", MAX_PARKING_SPACES, 0, 
                                     "8th May 2025", 0, 0)

    # Threading function for reading frames from camera 1
    def read_frames_thread_1():
        nonlocal frame_buffer_1, frame_ready_1, stop_thread_1
        while not stop_thread_1:
            if cap_1.isOpened():
                ret, frame = cap_1.read()
                if ret and frame is not None and frame.size > 0:
                    with frame_buffer_lock_1:
                        frame_buffer_1 = frame.copy()
                        frame_ready_1 = True
                else:
                    time.sleep(0.01)  # Small delay to avoid CPU hogging
            else:
                time.sleep(0.1)  # Larger delay if camera is not opened

    # Threading function for reading frames from camera 2
    def read_frames_thread_2():
        nonlocal frame_buffer_2, frame_ready_2, stop_thread_2
        while not stop_thread_2:
            if cap_2.isOpened():
                ret, frame = cap_2.read()
                if ret and frame is not None and frame.size > 0:
                    with frame_buffer_lock_2:
                        frame_buffer_2 = frame.copy()
                        frame_ready_2 = True
                else:
                    time.sleep(0.01)  # Small delay to avoid CPU hogging
            else:
                time.sleep(0.1)  # Larger delay if camera is not opened

    # Function to connect to camera 1
    def connect_to_cctv_1():
        nonlocal RTSP_URL_1, rtsp_url_index_1
        # Try each RTSP URL until one works
        for i in range(len(RTSP_URLS_1)):
            current_index = (rtsp_url_index_1 + i) % len(RTSP_URLS_1)
            current_url = RTSP_URLS_1[current_index]
            print(f"Trying to connect to Camera 1 at {current_url}...")
            
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
                        print(f"✅ Successfully connected to Camera 1 with TCP: {current_url}")
                        RTSP_URL_1 = current_url
                        rtsp_url_index_1 = current_index
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
                        print(f"✅ Successfully connected to Camera 1 with UDP: {current_url}")
                        RTSP_URL_1 = current_url
                        rtsp_url_index_1 = current_index
                        return cap
                    else:
                        print("Connected with UDP but couldn't get stable stream")
                        cap.release()
                else:
                    print(f"❌ Failed to connect with UDP: {current_url}")
            
            except Exception as e:
                print(f"Error trying RTSP URL '{current_url}': {e}")
                
        # Move to the next URL for future attempts
        rtsp_url_index_1 = (rtsp_url_index_1 + 1) % len(RTSP_URLS_1)
        
        # If all RTSP URLs fail, try fallback to webcam
        print("All RTSP URLs failed for Camera 1. Falling back to webcam...")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Connected to webcam for Camera 1")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            else:
                print("❌ Failed to connect to webcam for Camera 1")
        except Exception as e:
            print(f"Error connecting to webcam for Camera 1: {e}")
            
        return None

    # Function to connect to camera 2
    def connect_to_cctv_2():
        nonlocal RTSP_URL_2, rtsp_url_index_2
        # Try each RTSP URL until one works
        for i in range(len(RTSP_URLS_2)):
            current_index = (rtsp_url_index_2 + i) % len(RTSP_URLS_2)
            current_url = RTSP_URLS_2[current_index]
            print(f"Trying to connect to Camera 2 at {current_url}...")
            
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
                        print(f"✅ Successfully connected to Camera 2 with TCP: {current_url}")
                        RTSP_URL_2 = current_url
                        rtsp_url_index_2 = current_index
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
                        print(f"✅ Successfully connected to Camera 2 with UDP: {current_url}")
                        RTSP_URL_2 = current_url
                        rtsp_url_index_2 = current_index
                        return cap
                    else:
                        print("Connected with UDP but couldn't get stable stream")
                        cap.release()
                else:
                    print(f"❌ Failed to connect with UDP: {current_url}")
            
            except Exception as e:
                print(f"Error trying RTSP URL '{current_url}': {e}")
                
        # Move to the next URL for future attempts
        rtsp_url_index_2 = (rtsp_url_index_2 + 1) % len(RTSP_URLS_2)
        
        # If all RTSP URLs fail, try fallback to webcam
        print("All RTSP URLs failed for Camera 2. Falling back to webcam...")
        try:
            cap = cv2.VideoCapture(1)  # Try second webcam for camera 2
            if cap.isOpened():
                print("✅ Connected to webcam for Camera 2")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            else:
                print("❌ Failed to connect to webcam for Camera 2")
        except Exception as e:
            print(f"Error connecting to webcam for Camera 2: {e}")
            
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

    # Process frame function (to avoid code duplication)
    def process_frame(frame, camera_num, conf_threshold_yolov8, conf_threshold_yolov3, bike_focus_mode, enhance_mode, fast_mode):
        """Process a single frame with YOLOv8 and YOLOv3"""
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
        
        # ------- YOLOv8 Detection ---------
        # Determine which classes to focus on
        target_classes_yolov8 = bike_classes_yolov8 if bike_focus_mode else vehicle_classes_yolov8
        
        # Run YOLOv8 inference
        yolov8_results = yolov8_model(processing_frame, conf=conf_threshold_yolov8, verbose=False)
        
        # Process YOLOv8 results
        for r in yolov8_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = yolov8_model.names[cls_id]
                
                # Only process vehicle classes we're interested in
                if cls_name in target_classes_yolov8:
                    # Get coordinates (and scale back if using fast mode)
                    x1, y1, x2, y2 = box.xyxy[0]
                    if fast_mode:
                        scale_x, scale_y = width / 640, height / 360
                        x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence score
                    conf = float(box.conf[0])
                    
                    # Add to detected vehicles
                    all_detected_vehicles.append(("YOLOv8", (x1, y1, x2, y2), cls_name, conf))
        
        # ------- YOLOv3 Detection ---------
        # Skip YOLOv3 in fast mode to improve performance
        if not fast_mode:
            # Determine which classes to focus on
            target_classes_yolov3 = bike_classes_yolov3 if bike_focus_mode else vehicle_classes_yolov3
            
            # Create blob from image for YOLOv3
            blob = cv2.dnn.blobFromImage(processing_frame, 1/255.0, (416, 416), 
                                        swapRB=True, crop=False)
            
            # Set the input and get output
            yolov3_net.setInput(blob)
            try:
                layer_outputs = yolov3_net.forward(output_layers)
                
                # Process each detection
                for output in layer_outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence > conf_threshold_yolov3:
                            class_name = yolov3_classes[class_id]
                            
                            # Only process vehicle classes we're interested in
                            if class_name in target_classes_yolov3:
                                # Get coordinates
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                
                                # Calculate bounding box coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)
                                x2 = x + w
                                y2 = y + h
                                
                                # Add to detected vehicles
                                all_detected_vehicles.append(("YOLOv3", (x, y, x2, y2), class_name, confidence))
            except Exception as e:
                print(f"Error in YOLOv3 detection: {e}")
        
        # Apply non-maximum suppression to remove duplicates (from both models)
        final_vehicles = []
        
        # Sort by confidence (highest first)
        all_detected_vehicles.sort(key=lambda x: x[3], reverse=True)
        
        # Track which boxes have been selected
        used_boxes = []
        
        # Select boxes with NMS
        for model, box, cls_name, conf in all_detected_vehicles:
            # Check if this box overlaps significantly with any already used box
            is_duplicate = False
            for used_box in used_boxes:
                if calculate_iou(box, used_box) > 0.5:  # 0.5 is the IoU threshold
                    is_duplicate = True
                    break
            
            # If not a duplicate, add to final vehicles
            if not is_duplicate:
                final_vehicles.append((model, box, cls_name, conf))
                used_boxes.append(box)
        
        # Draw final detections on frame
        for model, (x1, y1, x2, y2), cls_name, conf in final_vehicles:
            # Choose color based on model (YOLOv8 = green, YOLOv3 = blue)
            color = (0, 255, 0) if model == "YOLOv8" else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"{cls_name}: {conf:.2f}"
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1-25), (x1+len(label)*11, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0, 0, 0), 2)
        
        # Add camera info and settings
        # Top-left: Frame resolution and camera number
        cv2.putText(frame, f"Camera {camera_num}: {width}x{height}", (10, 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top-right: Number of vehicles detected
        cv2.putText(frame, f"Vehicles: {len(final_vehicles)}", (width-200, 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Conf thresholds
        cv2.putText(frame, f"YOLOv8: {conf_threshold_yolov8:.2f}", (10, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"YOLOv3: {conf_threshold_yolov3:.2f}", (10, 75), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Modes
        mode_text = ""
        if bike_focus_mode:
            mode_text += "BIKE "
        if enhance_mode:
            mode_text += "ENHANCE "
        if fast_mode:
            mode_text += "FAST "
        if mode_text:
            cv2.putText(frame, mode_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.6, (0, 255, 255), 2)
        
        # Add active camera indicator if this is the active camera
        if camera_num == active_camera:
            cv2.putText(frame, "ACTIVE", (width-120, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame, len(final_vehicles)

# Main loop
print("Connecting to cameras...")
cap_1 = connect_to_cctv_1()
cap_2 = connect_to_cctv_2()

if cap_1 is None and cap_2 is None:
    print("Failed to connect to both cameras. Exiting...")
    return

# Start the reading threads
if cap_1 is not None:
    read_thread_1 = threading.Thread(target=read_frames_thread_1, daemon=True)
    read_thread_1.start()

if cap_2 is not None:
    read_thread_2 = threading.Thread(target=read_frames_thread_2, daemon=True)
    read_thread_2.start()

print("Starting main processing loop...")
while True:
    # Process camera 1 if available
    frame1 = None
    if cap_1 is not None:
        with frame_buffer_lock_1:
            if frame_ready_1 and frame_buffer_1 is not None:
                frame1 = frame_buffer_1.copy()
                frame_ready_1 = False
        
        if frame1 is not None:
            # Reset bad frame counter
            bad_frame_counter_1 = 0
            
            # Process the frame
            try:
                processed_frame1, count1 = process_frame(
                    frame1, 
                    1, 
                    conf_threshold_yolov8_1, 
                    conf_threshold_yolov3_1,
                    bike_focus_mode_1, 
                    enhance_mode_1, 
                    fast_mode_1
                )
                current_vehicle_count_1 = count1
            except Exception as e:
                print(f"Error processing frame from Camera 1: {e}")
                processed_frame1 = frame1
                current_vehicle_count_1 = 0
        else:
            # Increment bad frame counter
            bad_frame_counter_1 += 1
            
            # Create blank frame with message
            processed_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(processed_frame1, "Camera 1: No signal", (150, 240), 
                      cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            
            # Try to reconnect if too many bad frames
            current_time = time.time()
            if (bad_frame_counter_1 > stream_health_threshold and 
                current_time - last_reconnect_time_1 > min_reconnect_interval):
                print(f"Stream from Camera 1 is unhealthy (bad frames: {bad_frame_counter_1}). Attempting reconnection...")
                # Release current capture
                if cap_1 is not None:
                    cap_1.release()
                
                # Stop the thread
                stop_thread_1 = True
                if 'read_thread_1' in locals() and read_thread_1.is_alive():
                    read_thread_1.join(timeout=1.0)
                
                # Reset stop flag
                stop_thread_1 = False
                
                # Try to reconnect
                cap_1 = connect_to_cctv_1()
                
                # If reconnection successful, reset counters and restart thread
                if cap_1 is not None:
                    bad_frame_counter_1 = 0
                    last_reconnect_time_1 = current_time
                    frame_ready_1 = False
                    read_thread_1 = threading.Thread(target=read_frames_thread_1, daemon=True)
                    read_thread_1.start()
    else:
        # No camera connection
        processed_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(processed_frame1, "Camera 1: Not connected", (150, 240), 
                  cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        current_vehicle_count_1 = 0
    
    # Process camera 2 if available
    frame2 = None
    if cap_2 is not None:
        with frame_buffer_lock_2:
            if frame_ready_2 and frame_buffer_2 is not None:
                frame2 = frame_buffer_2.copy()
                frame_ready_2 = False
        
        if frame2 is not None:
            # Reset bad frame counter
            bad_frame_counter_2 = 0
            
            # Process the frame
            try:
                processed_frame2, count2 = process_frame(
                    frame2, 
                    2, 
                    conf_threshold_yolov8_2, 
                    conf_threshold_yolov3_2,
                    bike_focus_mode_2, 
                    enhance_mode_2, 
                    fast_mode_2
                )
                current_vehicle_count_2 = count2
            except Exception as e:
                print(f"Error processing frame from Camera 2: {e}")
                processed_frame2 = frame2
                current_vehicle_count_2 = 0
        else:
            # Increment bad frame counter
            bad_frame_counter_2 += 1
            
            # Create blank frame with message
            processed_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(processed_frame2, "Camera 2: No signal", (150, 240), 
                      cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            
            # Try to reconnect if too many bad frames
            current_time = time.time()
            if (bad_frame_counter_2 > stream_health_threshold and 
                current_time - last_reconnect_time_2 > min_reconnect_interval):
                print(f"Stream from Camera 2 is unhealthy (bad frames: {bad_frame_counter_2}). Attempting reconnection...")
                # Release current capture
                if cap_2 is not None:
                    cap_2.release()
                
                # Stop the thread
                stop_thread_2 = True
                if 'read_thread_2' in locals() and read_thread_2.is_alive():
                    read_thread_2.join(timeout=1.0)
                
                # Reset stop flag
                stop_thread_2 = False
                
                # Try to reconnect
                cap_2 = connect_to_cctv_2()
                
                # If reconnection successful, reset counters and restart thread
                if cap_2 is not None:
                    bad_frame_counter_2 = 0
                    last_reconnect_time_2 = current_time
                    frame_ready_2 = False
                    read_thread_2 = threading.Thread(target=read_frames_thread_2, daemon=True)
                    read_thread_2.start()
    else:
        # No camera connection
        processed_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(processed_frame2, "Camera 2: Not connected", (150, 240), 
                  cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        current_vehicle_count_2 = 0
    
    # Update dashboard data
    current_time = time.time()
    if current_time - last_dashboard_update > dashboard_update_interval:
        total_vehicle_count = current_vehicle_count_1 + current_vehicle_count_2
        # Calculate available spaces
        current_available_spaces = max(0, MAX_PARKING_SPACES - total_vehicle_count)
        # Calculate parking percentage
        parking_percentage = min(100, int((total_vehicle_count / MAX_PARKING_SPACES) * 100))
        # Parking status
        parking_status = "OPEN" if current_available_spaces > 0 else "FULL"
        # Current date
        current_date = datetime.datetime.now().strftime("%d %b %Y")
        
        # Update dashboard
        dashboard_frame = create_dashboard(
            dashboard_width, 
            dashboard_height, 
            parking_status, 
            current_available_spaces, 
            parking_percentage, 
            current_date,
            current_vehicle_count_1, 
            current_vehicle_count_2
        )
        
        last_dashboard_update = current_time
    
    # Choose which frames to display
    if dashboard_mode:
        # In dashboard mode, show the dashboard and one camera
        # Resize camera frame to fit half the dashboard
        dashboard_copy = dashboard_frame.copy()
        cam_height = dashboard_height // 2
        cam_width = int(cam_height * (16/9))  # Maintain aspect ratio
        
        if active_camera == 1:
            active_frame = cv2.resize(processed_frame1, (cam_width, cam_height))
        else:
            active_frame = cv2.resize(processed_frame2, (cam_width, cam_height))
        
        # Place the active camera feed in bottom right corner of dashboard
        h, w = active_frame.shape[:2]
        dashboard_copy[dashboard_height-h:dashboard_height, dashboard_width-w:dashboard_width] = active_frame
        
        # Add overlay text to indicate this is the active camera
        cv2.putText(dashboard_copy, f"Camera {active_camera} Live Feed", 
                  (dashboard_width-w+10, dashboard_height-h+25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show the dashboard with camera overlay
        cv2.imshow('IT Parking Detection System', dashboard_copy)
    else:
        # In regular mode, show both camera feeds side by side
        # Resize frames to same height for side-by-side display
        height1 = processed_frame1.shape[0]
        height2 = processed_frame2.shape[0]
        target_height = 480  # Fixed height for display
        
        # Resize both frames to same height
        aspect_ratio1 = processed_frame1.shape[1] / height1
        aspect_ratio2 = processed_frame2.shape[1] / height2
        
        width1 = int(target_height * aspect_ratio1)
        width2 = int(target_height * aspect_ratio2)
        
        resized_frame1 = cv2.resize(processed_frame1, (width1, target_height))
        resized_frame2 = cv2.resize(processed_frame2, (width2, target_height))
        
        # Combine frames side by side
        combined_frame = np.zeros((target_height, width1 + width2, 3), dtype=np.uint8)
        combined_frame[:, :width1] = resized_frame1
        combined_frame[:, width1:width1+width2] = resized_frame2
        
        # Show combined frame
        cv2.imshow('IT Parking Detection System', combined_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    # Quit on 'q'
    if key == ord('q'):
        break
    
    # Switch active camera
    elif key == ord('1'):
        active_camera = 1
        print(f"Active camera switched to Camera 1")
    elif key == ord('2'):
        active_camera = 2
        print(f"Active camera switched to Camera 2")
    
    # Toggle dashboard mode
    elif key == ord('d'):
        dashboard_mode = not dashboard_mode
        print(f"Dashboard mode: {'ON' if dashboard_mode else 'OFF'}")
    
    # Control active camera settings
    if active_camera == 1:
        # YOLOv8 confidence threshold for camera 1
        if key == ord('+'):
            conf_threshold_yolov8_1 = min(0.95, conf_threshold_yolov8_1 + 0.05)
            print(f"Camera 1 YOLOv8 confidence threshold: {conf_threshold_yolov8_1:.2f}")
        elif key == ord('-'):
            conf_threshold_yolov8_1 = max(0.05, conf_threshold_yolov8_1 - 0.05)
            print(f"Camera 1 YOLOv8 confidence threshold: {conf_threshold_yolov8_1:.2f}")
        
        # YOLOv3 confidence threshold for camera 1
        elif key == ord('.'):
            conf_threshold_yolov3_1 = min(0.95, conf_threshold_yolov3_1 + 0.05)
            print(f"Camera 1 YOLOv3 confidence threshold: {conf_threshold_yolov3_1:.2f}")
        elif key == ord(','):
            conf_threshold_yolov3_1 = max(0.05, conf_threshold_yolov3_1 - 0.05)
            print(f"Camera 1 YOLOv3 confidence threshold: {conf_threshold_yolov3_1:.2f}")
        
        # Toggle bike focus mode for camera 1
        elif key == ord('b'):
            bike_focus_mode_1 = not bike_focus_mode_1
            print(f"Camera 1 bike focus mode: {'ON' if bike_focus_mode_1 else 'OFF'}")
        
        # Toggle enhancement mode for camera 1
        elif key == ord('e'):
            enhance_mode_1 = not enhance_mode_1
            print(f"Camera 1 enhancement mode: {'ON' if enhance_mode_1 else 'OFF'}")
        
        # Toggle fast mode for camera 1
        elif key == ord('f'):
            fast_mode_1 = not fast_mode_1
            print(f"Camera 1 fast mode: {'ON' if fast_mode_1 else 'OFF'}")
        
        # Force reconnection for camera 1
        elif key == ord('r'):
            print("Forcing reconnection to Camera 1...")
            if cap_1 is not None:
                cap_1.release()
            
            # Stop the thread
            stop_thread_1 = True
            if 'read_thread_1' in locals() and read_thread_1.is_alive():
                read_thread_1.join(timeout=1.0)
            
            # Reset stop flag
            stop_thread_1 = False
            
            # Try to reconnect
            cap_1 = connect_to_cctv_1()
            
            # If reconnection successful, restart thread
            if cap_1 is not None:
                bad_frame_counter_1 = 0
                last_reconnect_time_1 = time.time()
                frame_ready_1 = False
                read_thread_1 = threading.Thread(target=read_frames_thread_1, daemon=True)
                read_thread_1.start()
        
        # Try next RTSP URL format for camera 1
        elif key == ord('n'):
            print("Trying next RTSP URL format for Camera 1...")
            rtsp_url_index_1 = (rtsp_url_index_1 + 1) % len(RTSP_URLS_1)
            
            if cap_1 is not None:
                cap_1.release()
            
            # Stop the thread
            stop_thread_1 = True
            if 'read_thread_1' in locals() and read_thread_1.is_alive():
                read_thread_1.join(timeout=1.0)
            
            # Reset stop flag
            stop_thread_1 = False
            
            # Try to reconnect with next URL
            cap_1 = connect_to_cctv_1()
            
            # If reconnection successful, restart thread
            if cap_1 is not None:
                bad_frame_counter_1 = 0
                last_reconnect_time_1 = time.time()
                frame_ready_1 = False
                read_thread_1 = threading.Thread(target=read_frames_thread_1, daemon=True)
                read_thread_1.start()
    else:  # active_camera == 2
        # YOLOv8 confidence threshold for camera 2
        if key == ord('+'):
            conf_threshold_yolov8_2 = min(0.95, conf_threshold_yolov8_2 + 0.05)
            print(f"Camera 2 YOLOv8 confidence threshold: {conf_threshold_yolov8_2:.2f}")
        elif key == ord('-'):
            conf_threshold_yolov8_2 = max(0.05, conf_threshold_yolov8_2 - 0.05)
            print(f"Camera 2 YOLOv8 confidence threshold: {conf_threshold_yolov8_2:.2f}")
        
        # YOLOv3 confidence threshold for camera 2
        elif key == ord('.'):
            conf_threshold_yolov3_2 = min(0.95, conf_threshold_yolov3_2 + 0.05)
            print(f"Camera 2 YOLOv3 confidence threshold: {conf_threshold_yolov3_2:.2f}")
        elif key == ord(','):
            conf_threshold_yolov3_2 = max(0.05, conf_threshold_yolov3_2 - 0.05)
            print(f"Camera 2 YOLOv3 confidence threshold: {conf_threshold_yolov3_2:.2f}")
        
        # Toggle bike focus mode for camera 2
        elif key == ord('b'):
            bike_focus_mode_2 = not bike_focus_mode_2
            print(f"Camera 2 bike focus mode: {'ON' if bike_focus_mode_2 else 'OFF'}")
        
        # Toggle enhancement mode for camera 2
        elif key == ord('e'):
            enhance_mode_2 = not enhance_mode_2
            print(f"Camera 2 enhancement mode: {'ON' if enhance_mode_2 else 'OFF'}")
        
        # Toggle fast mode for camera 2
        elif key == ord('f'):
            fast_mode_2 = not fast_mode_2
            print(f"Camera 2 fast mode: {'ON' if fast_mode_2 else 'OFF'}")
        
        # Force reconnection for camera 2
        elif key == ord('r'):
            print("Forcing reconnection to Camera 2...")
            if cap_2 is not None:
                cap_2.release()
            
            # Stop the thread
            stop_thread_2 = True
            if 'read_thread_2' in locals() and read_thread_2.is_alive():
                read_thread_2.join(timeout=1.0)
            
            # Reset stop flag
            stop_thread_2 = False
            
            # Try to reconnect
            cap_2 = connect_to_cctv_2()
            
            # If reconnection successful, restart thread
            if cap_2 is not None:
                bad_frame_counter_2 = 0
                last_reconnect_time_2 = time.time()
                frame_ready_2 = False
                read_thread_2 = threading.Thread(target=read_frames_thread_2, daemon=True)
                read_thread_2.start()
        
        # Try next RTSP URL format for camera 2
        elif key == ord('n'):
            print("Trying next RTSP URL format for Camera 2...")
            rtsp_url_index_2 = (rtsp_url_index_2 + 1) % len(RTSP_URLS_2)
            
            if cap_2 is not None:
                cap_2.release()
            
            # Stop the thread
            stop_thread_2 = True
            if 'read_thread_2' in locals() and read_thread_2.is_alive():
                read_thread_2.join(timeout=1.0)
            
            # Reset stop flag
            stop_thread_2 = False
            
            # Try to reconnect with next URL
            cap_2 = connect_to_cctv_2()
            
            # If reconnection successful, restart thread
            if cap_2 is not None:
                bad_frame_counter_2 = 0
                last_reconnect_time_2 = time.time()
                frame_ready_2 = False
                read_thread_2 = threading.Thread(target=read_frames_thread_2, daemon=True)
                read_thread_2.start()

# Clean up
print("Cleaning up...")
stop_thread_1 = True
stop_thread_2 = True

# Release camera captures
if cap_1 is not None:
    cap_1.release()
if cap_2 is not None:
    cap_2.release()

# Wait for threads to stop
if 'read_thread_1' in locals() and read_thread_1.is_alive():
    read_thread_1.join(timeout=1.0)
if 'read_thread_2' in locals() and read_thread_2.is_alive():
    read_thread_2.join(timeout=1.0)

# Close all windows
cv2.destroyAllWindows()
print("Application closed successfully.")

if __name__ == "__main__":
    main()

