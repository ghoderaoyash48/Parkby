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
    print("  - Press 'q': Quit the application")
    print("  - Press '+': Increase YOLOv8 confidence threshold")
    print("  - Press '-': Decrease YOLOv8 confidence threshold")
    print("  - Press '.': Increase YOLOv3 confidence threshold")
    print("  - Press ',': Decrease YOLOv3 confidence threshold")
    print("  - Press 'b': Toggle bike/motorcycle focus mode")
    print("  - Press 'e': Toggle image enhancement")
    print("  - Press 'r': Force reconnection to cameras")
    print("  - Press 'n': Try next RTSP URL format")
    print("  - Press 'f': Toggle faster processing mode")
    print("  - Press 'd': Toggle dashboard view")
    print("  - Press '1': Switch active control to Camera 1")
    print("  - Press '2': Switch active control to Camera 2")
   
    # CCTV Camera information for first camera
    IP_ADDRESS_1 = "172.16.101.196"  # First CCTV camera IP
    USERNAME_1 = "admin"             # Replace with your CCTV username
    PASSWORD_1 = "SMART@123"         # Replace with your CCTV password
   
    # CCTV Camera information for second camera
    IP_ADDRESS_2 = "172.16.101.193"  # Second CCTV camera IP
    USERNAME_2 = "admin"             # Replace with your CCTV username
    PASSWORD_2 = "SMART@123"         # Replace with your CCTV password
   
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
   
    # Define initial confidence thresholds (can be adjusted during runtime)
    conf_threshold_yolov8_1 = 0.25  # YOLOv8 threshold for camera 1
    conf_threshold_yolov3_1 = 0.4   # YOLOv3 threshold for camera 1
    conf_threshold_yolov8_2 = 0.25  # YOLOv8 threshold for camera 2
    conf_threshold_yolov3_2 = 0.4   # YOLOv3 threshold for camera 2
    nms_threshold = 0.4
   
    # Flag for bike focus mode and image enhancement - shared between cameras
    bike_focus_mode = False
    enhance_mode = False
    fast_mode = False  # Flag for faster processing
    dashboard_mode = True  # Flag for dashboard view
   
    # Stream health monitoring for each camera
    stream_health_threshold = 10  # Number of consecutive bad frames to trigger reconnect
    bad_frame_counter_1 = 0
    bad_frame_counter_2 = 0
    last_reconnect_time_1 = time.time()
    last_reconnect_time_2 = time.time()
    min_reconnect_interval = 10  # Minimum seconds between reconnection attempts
   
     # Define maximum parking capacity for each camera
    MAX_PARKING_SPACES = 200
   
    # Setup buffer for threaded reading - Camera 1
    frame_buffer_1 = None
    frame_buffer_lock_1 = threading.Lock()
    frame_ready_1 = False
    stop_thread_1 = False
   
    # Setup buffer for threaded reading - Camera 2
    frame_buffer_2 = None
    frame_buffer_lock_2 = threading.Lock()
    frame_ready_2 = False
    stop_thread_2 = False
   
   
    # 3. Add these variables for combined dashboard
    last_dashboard_update = time.time()
    current_vehicle_count_total = 0
    current_available_spaces_total = MAX_PARKING_SPACES
    parking_percentage_total = 0
    combined_dashboard_frame = None

    # Function to create dashboard
    def create_combined_dashboard(width, height, status, available_slots, percentage_filled, date_str):
       """Create a dashboard frame showing combined data from both cameras"""
    # Create blank frame
    dashboard = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw border rectangle
    cv2.rectangle(dashboard, (0, 0), (width-1, height-1), (0, 0, 0), 2)
    
    # Draw IT PARKING header
    header_height = height // 8
    cv2.rectangle(dashboard, (0, 0), (width, header_height), (0, 125, 255), -1)  # Orange background
    cv2.putText(dashboard, "IT PARKING - COMBINED VIEW", (width//6, header_height-15), 
                cv2.FONT_HERSHEY_DUPLEX, 1.8, (25, 25, 112), 4)  # Dark blue text
    
    # Status section
    status_section_height = height // 4
    cv2.rectangle(dashboard, (0, header_height), (width, header_height + status_section_height), 
                  (255, 255, 255), -1)  # White background
    
    # Yellow background for status text
    yellow_bg_margin = width // 6
    cv2.rectangle(dashboard, (yellow_bg_margin, header_height + 10), 
                  (width - yellow_bg_margin, header_height + status_section_height - 10),
                  (255, 255, 160), -1)  # Light yellow
    
    # Status text (OPEN in green or FULL in red)
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
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 125, 0), 3)  # Orange percentage
    
    # Available slots text
    slots_text_y = progress_start_y + progress_height + 80
    cv2.putText(dashboard, f"AVAILABLE SLOTS : {available_slots}", 
                (width//6, slots_text_y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (25, 118, 210), 3)  # Blue text
    
    # Camera info
    camera_text_y = slots_text_y + 60
    cv2.putText(dashboard, "MONITORING: BOTH CAMERAS", 
                (width//6, camera_text_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (50, 50, 50), 2)
    
    # Footer with date and time
    footer_start_y = height - height//8
    cv2.rectangle(dashboard, (0, footer_start_y), (width, height), 
                  (25, 25, 112), -1)  # Dark blue background
    
    # Current time
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
    cv2.putText(dashboard, current_time, (width//20, footer_start_y + height//16), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # Date
    cv2.putText(dashboard, date_str, (width//2 - 100, footer_start_y + height//16), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)  # Cyan date
    
    # Last update time
    cv2.putText(dashboard, f"Last Update: 10 sec ago", (width - 300, footer_start_y + height//16), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return dashboard
   
     # Function to combine two dashboards side by side
    def combine_dashboards(dashboard1, dashboard2, width=1600, height=600):
        # Create a combined dashboard with the two dashboards side by side
        combined = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
       
        # Resize both dashboards to fit half the width each
        half_width = width // 2
        dashboard1_resized = cv2.resize(dashboard1, (half_width, height))
        dashboard2_resized = cv2.resize(dashboard2, (half_width, height))
       
        # Place dashboards side by side
        combined[:, 0:half_width] = dashboard1_resized
        combined[:, half_width:width] = dashboard2_resized
       
        # Add a dividing line
        cv2.line(combined, (half_width, 0), (half_width, height), (0, 0, 0), 2)
       
        # Add camera labels at top
        cv2.putText(combined, "Camera 1", (half_width//2 - 70, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(combined, "Camera 2", (half_width + half_width//2 - 70, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
       
        # Add active camera indicator
        if active_camera == 1:
            cv2.putText(combined, "* ACTIVE *", (half_width//2 - 70, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(combined, "* ACTIVE *", (half_width + half_width//2 - 70, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                   
        return combined
   
    # Create initial dashboard frames
    dashboard_width = 800
    dashboard_height = 600
    dashboard_frame_1 = create_dashboard(dashboard_width, dashboard_height,
                                        "OPEN", 27, 73, "21st April 2025", "Camera 1")
    dashboard_frame_2 = create_dashboard(dashboard_width, dashboard_height,
                                        "OPEN", 27, 73, "21st April 2025", "Camera 2")
   
    # Combined dashboard
    combined_dashboard = combine_dashboards(dashboard_frame_1, dashboard_frame_2)
   
    # Threading function for reading frames from camera 1
    def read_frames_thread_1():
        nonlocal frame_buffer_1, frame_ready_1, stop_thread_1
       
        while not stop_thread_1:
            if cap1.isOpened():
                ret, frame = cap1.read()
               
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
            if cap2.isOpened():
                ret, frame = cap2.read()
               
                if ret and frame is not None and frame.size > 0:
                    with frame_buffer_lock_2:
                        frame_buffer_2 = frame.copy()
                        frame_ready_2 = True
                else:
                    time.sleep(0.01)  # Small delay to avoid CPU hogging
            else:
                time.sleep(0.1)  # Larger delay if camera is not opened
   
    # Function to connect to CCTV - camera 1
    def connect_to_cctv_1():
        nonlocal RTSP_URL_1, rtsp_url_index_1
       
        # Try each RTSP URL until one works
        for i in range(len(RTSP_URLS_1)):
            current_index = (rtsp_url_index_1 + i) % len(RTSP_URLS_1)
            current_url = RTSP_URLS_1[current_index]
           
            print(f"Camera 1: Trying to connect at {current_url}...")
           
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
                        print(f"✅ Camera 1: Successfully connected with TCP: {current_url}")
                        RTSP_URL_1 = current_url
                        rtsp_url_index_1 = current_index
                        return cap
                    else:
                        print("Camera 1: Connected but stream is not stable")
                        cap.release()
                else:
                    print(f"❌ Camera 1: Failed to connect with TCP: {current_url}")
               
                # Try UDP if TCP failed
                print("Camera 1: Trying UDP transport...")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;500000"
                cap = cv2.VideoCapture(current_url, cv2.CAP_FFMPEG)
               
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ Camera 1: Successfully connected with UDP: {current_url}")
                        RTSP_URL_1 = current_url
                        rtsp_url_index_1 = current_index
                        return cap
                    else:
                        print("Camera 1: Connected with UDP but couldn't get stable stream")
                        cap.release()
                else:
                    print(f"❌ Camera 1: Failed to connect with UDP: {current_url}")
                   
            except Exception as e:
                print(f"Camera 1: Error trying RTSP URL '{current_url}': {e}")
       
        # Move to the next URL for future attempts
        rtsp_url_index_1 = (rtsp_url_index_1 + 1) % len(RTSP_URLS_1)
       
        # If all RTSP URLs fail, try fallback to webcam
        print("Camera 1: All RTSP URLs failed. Falling back to webcam...")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Camera 1: Connected to webcam")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            else:
                print("❌ Camera 1: Failed to connect to webcam")
        except Exception as e:
            print(f"Camera 1: Error connecting to webcam: {e}")
       
        return None
   
    # Function to connect to CCTV - camera 2
    def connect_to_cctv_2():
        nonlocal RTSP_URL_2, rtsp_url_index_2
       
        # Try each RTSP URL until one works
        for i in range(len(RTSP_URLS_2)):
            current_index = (rtsp_url_index_2 + i) % len(RTSP_URLS_2)
            current_url = RTSP_URLS_2[current_index]
           
            print(f"Camera 2: Trying to connect at {current_url}...")
           
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
                        print(f"✅ Camera 2: Successfully connected with TCP: {current_url}")
                        RTSP_URL_2 = current_url
                        rtsp_url_index_2 = current_index
                        return cap
                    else:
                        print("Camera 2: Connected but stream is not stable")
                        cap.release()
                else:
                    print(f"❌ Camera 2: Failed to connect with TCP: {current_url}")
               
                # Try UDP if TCP failed
                print("Camera 2: Trying UDP transport...")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;500000"
                cap = cv2.VideoCapture(current_url, cv2.CAP_FFMPEG)
               
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ Camera 2: Successfully connected with UDP: {current_url}")
                        RTSP_URL_2 = current_url
                        rtsp_url_index_2 = current_index
                        return cap
                    else:
                        print("Camera 2: Connected with UDP but couldn't get stable stream")
                        cap.release()
                else:
                    print(f"❌ Camera 2: Failed to connect with UDP: {current_url}")
                   
            except Exception as e:
                print(f"Camera 2: Error trying RTSP URL '{current_url}': {e}")
       
        # Move to the next URL for future attempts
        rtsp_url_index_2 = (rtsp_url_index_2 + 1) % len(RTSP_URLS_2)
       
        # If all RTSP URLs fail, try fallback to webcam
        print("Camera 2: All RTSP URLs failed. Falling back to webcam...")
        try:
            cap = cv2.VideoCapture(1)  # Try second webcam
            if cap.isOpened():
                print("✅ Camera 2: Connected to webcam")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            else:
                print("❌ Camera 2: Failed to connect to webcam")
        except Exception as e:
            print(f"Camera 2: Error connecting to webcam: {e}")
       
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
   
    # Function to process a frame and detect vehicles - optimized for YOLOv8
    def process_frame_yolov8(frame, model, conf_threshold, focus_classes=None):
        """Process a frame with YOLOv8 and return vehicle detections"""
        if frame is None or frame.size == 0:
            return frame, [], 0
       
        # Copy the frame to avoid modifying the original
        result_frame = frame.copy()
       
        # Detect objects in the frame
        results = model(frame, conf=conf_threshold, verbose=False)
       
        # Get bounding boxes, confidences, and class IDs
        boxes = []
        vehicle_count = 0
       
        if len(results) > 0:
            # Extract detection results
            for r in results:
                boxes_data = r.boxes
               
                for box in boxes_data:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                   
                    # Get class name
                    cls_name = model.names[cls_id]
                   
                    # Check if the detected object is a vehicle
                    if cls_name in vehicle_classes_yolov8:
                        # If in focus mode, only count bikes/motorcycles
                        if bike_focus_mode and cls_name not in bike_classes_yolov8:
                            continue
                           
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                       
                        # Add box to list
                        boxes.append((x1, y1, x2, y2, conf, cls_name))
                        vehicle_count += 1
                       
                        # Draw bounding box and label on the frame
                        color = (0, 255, 0) if cls_name in bike_classes_yolov8 else (0, 0, 255)
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                       
                        # Add label with class name and confidence
                        label = f"{cls_name}: {conf:.2f}"
                        cv2.putText(result_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       
        return result_frame, boxes, vehicle_count
   
    # Function to process a frame and detect vehicles - YOLOv3
    def process_frame_yolov3(frame, net, output_layers, classes, conf_threshold, focus_classes=None):
        """Process a frame with YOLOv3 and return vehicle detections"""
        if frame is None or frame.size == 0:
            return frame, [], 0
       
        # Copy the frame to avoid modifying the original
        result_frame = frame.copy()
       
        # Get dimensions of the frame
        height, width, _ = frame.shape
       
        # Prepare the frame for YOLOv3 (resize, normalize, and create blob)
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
           
            # Forward pass
            layer_outputs = net.forward(output_layers)
        except Exception as e:
            print(f"Error during YOLOv3 processing: {e}")
            return result_frame, [], 0
       
        # Initialize lists for detected objects
        boxes = []
        vehicle_count = 0
       
        # Process each output layer
        for output in layer_outputs:
            # Process each detection
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
               
                # Get class name
                if class_id < len(classes):
                    class_name = classes[class_id]
                else:
                    class_name = "unknown"
               
                # Only process vehicles above confidence threshold
                if confidence > conf_threshold and class_name in vehicle_classes_yolov3:
                    # If in focus mode, only count bikes/motorcycles
                    if bike_focus_mode and class_name not in bike_classes_yolov3:
                        continue
                       
                    # Get bounding box coordinates (normalized)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                   
                    # Calculate top-left corner of bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                   
                    # Add box to list with format (x1, y1, x2, y2, conf, class_name)
                    boxes.append((x, y, x + w, y + h, confidence, class_name))
                    vehicle_count += 1
                   
                    # Draw bounding box and label on the frame
                    color = (0, 255, 0) if class_name in bike_classes_yolov3 else (0, 0, 255)
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                   
                    # Add label with class name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(result_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       
        return result_frame, boxes, vehicle_count
   
    # Main execution block
    try:
        # Connect to cameras
        print("Connecting to Camera 1...")
        cap1 = connect_to_cctv_1()
        if cap1 is None:
            print("ERROR: Could not connect to Camera 1. Exiting...")
            return
       
        print("Connecting to Camera 2...")
        cap2 = connect_to_cctv_2()
        if cap2 is None:
            print("ERROR: Could not connect to Camera 2. Exiting...")
            cap1.release()
            return
       
        # Start threads for reading frames
        thread1 = threading.Thread(target=read_frames_thread_1)
        thread1.daemon = True
        thread1.start()
       
        thread2 = threading.Thread(target=read_frames_thread_2)
        thread2.daemon = True
        thread2.start()
       
        # Window setup for combined view
        cv2.namedWindow("Dual Camera Vehicle Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dual Camera Vehicle Detection", 1600, 600)
       
        # Window for combined dashboard
        cv2.namedWindow("Parking Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Dashboard", 1600, 600)
       
        # Create initial processing counters
        frame_counter = 0
        skip_frames = 2  # Process every 3rd frame for better performance
       
        print("\nRunning vehicle detection system on both cameras. Press 'q' to quit.")
       
        # Main loop
        while True:
            # Get current time for this frame
            current_time = time.time()
            current_date = datetime.datetime.now().strftime("%d %B %Y")
           
            # Initialize frame variables
            cam1_frame = None
            cam2_frame = None
           
            # Get frame from Camera 1 if available
            with frame_buffer_lock_1:
                if frame_ready_1 and frame_buffer_1 is not None:
                    cam1_frame = frame_buffer_1.copy()
                    frame_ready_1 = False
           
            # Get frame from Camera 2 if available
            with frame_buffer_lock_2:
                if frame_ready_2 and frame_buffer_2 is not None:
                    cam2_frame = frame_buffer_2.copy()
                    frame_ready_2 = False
           
            # Skip frames to improve performance if needed
            frame_counter += 1
            process_this_frame = (frame_counter % (skip_frames + 1) == 0)
           
            # Process Camera 1 frame
            if cam1_frame is not None:
                # Reset bad frame counter
                bad_frame_counter_1 = 0
               
                # Apply image enhancement if enabled
                if enhance_mode:
                    cam1_frame = enhance_image(cam1_frame)
               
                # Process frame with YOLOv8 and YOLOv3 if needed
                if process_this_frame or not fast_mode:
                    # Process with YOLOv8
                    cam1_frame, boxes_yolov8_1, count_yolov8_1 = process_frame_yolov8(
                        cam1_frame, yolov8_model, conf_threshold_yolov8_1)
                   
                    # Process with YOLOv3
                    cam1_frame, boxes_yolov3_1, count_yolov3_1 = process_frame_yolov3(
                        cam1_frame, yolov3_net, output_layers, yolov3_classes, conf_threshold_yolov3_1)
                   
                    # Combine counts (avoiding double counting)
                    vehicle_count_1 = count_yolov8_1  # Use YOLOv8 count as primary
                   
                    # Update dashboard data for Camera 1
                    if current_time - last_dashboard_update_1 > dashboard_update_interval:
                        last_dashboard_update_1 = current_time
                        current_vehicle_count_1 = vehicle_count_1
                        current_available_spaces_1 = max(0, MAX_PARKING_SPACES_1 - current_vehicle_count_1)
                        parking_percentage_1 = min(100, int((current_vehicle_count_1 / MAX_PARKING_SPACES_1) * 100))
                       
                        # Update dashboard frame
                        status_1 = "OPEN" if current_available_spaces_1 > 0 else "FULL"
                        dashboard_frame_1 = create_dashboard(dashboard_width, dashboard_height,
                                                           status_1, current_available_spaces_1,
                                                           parking_percentage_1, current_date, "Camera 1")
               
                # Add FPS and settings info to frame 1
                fps_text = f"Processing: {'Normal' if not fast_mode else 'Fast'}"
                cv2.putText(cam1_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               
                # Add confidence thresholds
                conf_text = f"YOLOv8: {conf_threshold_yolov8_1:.2f}, YOLOv3: {conf_threshold_yolov3_1:.2f}"
                cv2.putText(cam1_frame, conf_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               
                # Add mode indicators
                mode_text = "Mode: "
                mode_text += "Bike Focus, " if bike_focus_mode else "All Vehicles, "
                mode_text += "Enhanced" if enhance_mode else "Normal"
                cv2.putText(cam1_frame, mode_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               
                # Add active camera indicator
                if active_camera == 1:
                    cv2.putText(cam1_frame, "* ACTIVE *", (cam1_frame.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Increment bad frame counter
                bad_frame_counter_1 += 1
               
                # Create blank frame if no frame available
                cam1_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(cam1_frame, "Camera 1: No Signal", (50, 240),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
               
                # Try to reconnect if bad frames exceed threshold
                if (bad_frame_counter_1 > stream_health_threshold and
                    current_time - last_reconnect_time_1 > min_reconnect_interval):
                    print("Camera 1: Stream unhealthy. Attempting to reconnect...")
                   
                    # Stop thread and release camera
                    stop_thread_1 = True
                    if thread1.is_alive():
                        thread1.join(1.0)  # Wait up to 1 second for thread to finish
                   
                    if cap1 is not None:
                        cap1.release()
                   
                    # Reconnect
                    cap1 = connect_to_cctv_1()
                   
                    # Start new thread
                    stop_thread_1 = False
                    thread1 = threading.Thread(target=read_frames_thread_1)
                    thread1.daemon = True
                    thread1.start()
                   
                    # Reset counters
                    bad_frame_counter_1 = 0
                    last_reconnect_time_1 = current_time
           
            # Process Camera 2 frame
            if cam2_frame is not None:
                # Reset bad frame counter
                bad_frame_counter_2 = 0
               
                # Apply image enhancement if enabled
                if enhance_mode:
                    cam2_frame = enhance_image(cam2_frame)
               
                # Process frame with YOLOv8 and YOLOv3 if needed
                if process_this_frame or not fast_mode:
                    # Process with YOLOv8
                    cam2_frame, boxes_yolov8_2, count_yolov8_2 = process_frame_yolov8(
                        cam2_frame, yolov8_model, conf_threshold_yolov8_2)
                   
                    # Process with YOLOv3
                    cam2_frame, boxes_yolov3_2, count_yolov3_2 = process_frame_yolov3(
                        cam2_frame, yolov3_net, output_layers, yolov3_classes, conf_threshold_yolov3_2)

                   
                    # Combine counts (avoiding double counting)
                    vehicle_count_2 = count_yolov8_2  # Use YOLOv8 count as primary
                   
                    # Update dashboard data for Camera 2
                    if current_time - last_dashboard_update_2 > dashboard_update_interval:
                        last_dashboard_update_2 = current_time
                        current_vehicle_count_2 = vehicle_count_2
                        current_available_spaces_2 = max(0, MAX_PARKING_SPACES_2 - current_vehicle_count_2)
                        parking_percentage_2 = min(100, int((current_vehicle_count_2 / MAX_PARKING_SPACES_2) * 100))
                       
                        # Update dashboard frame
                        status_2 = "OPEN" if current_available_spaces_2 > 0 else "FULL"
                        dashboard_frame_2 = create_dashboard(dashboard_width, dashboard_height,
                                                           status_2, current_available_spaces_2,
                                                           parking_percentage_2, current_date, "Camera 2")
               
                # Add FPS and settings info to frame 2
                fps_text = f"Processing: {'Normal' if not fast_mode else 'Fast'}"
                cv2.putText(cam2_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               
                # Add confidence thresholds
                conf_text = f"YOLOv8: {conf_threshold_yolov8_2:.2f}, YOLOv3: {conf_threshold_yolov3_2:.2f}"
                cv2.putText(cam2_frame, conf_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               
                # Add mode indicators
                mode_text = "Mode: "
                mode_text += "Bike Focus, " if bike_focus_mode else "All Vehicles, "
                mode_text += "Enhanced" if enhance_mode else "Normal"
                cv2.putText(cam2_frame, mode_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               
                # Add active camera indicator
                if active_camera == 2:
                    cv2.putText(cam2_frame, "* ACTIVE *", (cam2_frame.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Increment bad frame counter
                bad_frame_counter_2 += 1
               
                # Create blank frame if no frame available
                cam2_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(cam2_frame, "Camera 2: No Signal", (50, 240),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
               
                # Try to reconnect if bad frames exceed threshold
                if (bad_frame_counter_2 > stream_health_threshold and
                    current_time - last_reconnect_time_2 > min_reconnect_interval):
                    print("Camera 2: Stream unhealthy. Attempting to reconnect...")
                   
                    # Stop thread and release camera
                    stop_thread_2 = True
                    if thread2.is_alive():
                        thread2.join(1.0)  # Wait up to 1 second for thread to finish
                   
                    if cap2 is not None:
                        cap2.release()
                   
                    # Reconnect
                    cap2 = connect_to_cctv_2()
                   
                    # Start new thread
                    stop_thread_2 = False
                    thread2 = threading.Thread(target=read_frames_thread_2)
                    thread2.daemon = True
                    thread2.start()
                   
                    # Reset counters
                    bad_frame_counter_2 = 0
                    last_reconnect_time_2 = current_time
            # Calculate combined stats
              if cam1_frame is not None and cam2_frame is not None:
            # Update combined dashboard data
               if current_time - last_dashboard_update > dashboard_update_interval:
                last_dashboard_update = current_time
            # Combine vehicle counts, avoiding duplicates as much as possible
            # We'll use a simple approach - take the higher count between cameras and add ~30% of the lower count
            # This is an approximation since we can't perfectly deduplicate without advanced tracking
              if vehicle_count_1 > vehicle_count_2:
            higher_count = vehicle_count_1
            lower_count = vehicle_count_2
        else:
            higher_count = vehicle_count_2
            lower_count = vehicle_count_1
        
        # Calculate the combined count - add about 30% of the lower count to avoid double counting
        current_vehicle_count_total = higher_count + int(lower_count * 0.3)
        # Ensure we don't exceed the maximum
        current_vehicle_count_total = min(current_vehicle_count_total, MAX_PARKING_SPACES)
        # Calculate available spaces and percentage
        current_available_spaces_total = max(0, MAX_PARKING_SPACES - current_vehicle_count_total)
        parking_percentage_total = min(100, int((current_vehicle_count_total / MAX_PARKING_SPACES) * 100))
        
        # Update combined dashboard frame
        status_total = "OPEN" if current_available_spaces_total > 0 else "FULL"
        combined_dashboard_frame = create_combined_dashboard(
            dashboard_width, dashboard_height,
            status_total, current_available_spaces_total,
            parking_percentage_total, current_date)       
            # Create combined display
            # Make sure both frames are the same size
            if cam1_frame.shape != cam2_frame.shape:
                cam1_height, cam1_width = cam1_frame.shape[:2]
                cam2_height, cam2_width = cam2_frame.shape[:2]
                target_height = max(cam1_height, cam2_height)
                target_width = max(cam1_width, cam2_width)
               
                # Resize both frames to the same dimensions
                cam1_frame = cv2.resize(cam1_frame, (target_width, target_height))
                cam2_frame = cv2.resize(cam2_frame, (target_width, target_height))
           
            # Combine frames side by side
            combined_frame = np.hstack((cam1_frame, cam2_frame))
           
            # Update combined dashboard
            combined_dashboard = combine_dashboards(dashboard_frame_1, dashboard_frame_2)
           
            # Display combined view and dashboard
            cv2.imshow("Dual Camera Vehicle Detection", combined_frame)
           
            # Only show dashboard if dashboard mode is enabled
            if dashboard_mode:
                cv2.imshow("Parking Dashboard", combined_dashboard)
            else:
                cv2.destroyWindow("Parking Dashboard")
           
             # Process key presses
            key = cv2.waitKey(1) & 0xFF
           
            if key == ord('q'):
                print("Quitting application...")
                break
            elif key == ord('+'):
                # Increase YOLOv8 confidence threshold for active camera
                if active_camera == 1:
                    conf_threshold_yolov8_1 = min(0.95, conf_threshold_yolov8_1 + 0.05)
                    print(f"Camera 1 YOLOv8 confidence threshold increased to {conf_threshold_yolov8_1:.2f}")
                else:
                    conf_threshold_yolov8_2 = min(0.95, conf_threshold_yolov8_2 + 0.05)
                    print(f"Camera 2 YOLOv8 confidence threshold increased to {conf_threshold_yolov8_2:.2f}")
            elif key == ord('-'):
                # Decrease YOLOv8 confidence threshold for active camera
                if active_camera == 1:
                    conf_threshold_yolov8_1 = max(0.05, conf_threshold_yolov8_1 - 0.05)
                    print(f"Camera 1 YOLOv8 confidence threshold decreased to {conf_threshold_yolov8_1:.2f}")
                else:
                    conf_threshold_yolov8_2 = max(0.05, conf_threshold_yolov8_2 - 0.05)
                    print(f"Camera 2 YOLOv8 confidence threshold decreased to {conf_threshold_yolov8_2:.2f}")
            elif key == ord('.'):
                # Increase YOLOv3 confidence threshold for active camera
                if active_camera == 1:
                    conf_threshold_yolov3_1 = min(0.95, conf_threshold_yolov3_1 + 0.05)
                    print(f"Camera 1 YOLOv3 confidence threshold increased to {conf_threshold_yolov3_1:.2f}")
                else:
                    conf_threshold_yolov3_2 = min(0.95, conf_threshold_yolov3_2 + 0.05)
                    print(f"Camera 2 YOLOv3 confidence threshold increased to {conf_threshold_yolov3_2:.2f}")
            elif key == ord(','):
                # Decrease YOLOv3 confidence threshold for active camera
                if active_camera == 1:
                    conf_threshold_yolov3_1 = max(0.05, conf_threshold_yolov3_1 - 0.05)
                    print(f"Camera 1 YOLOv3 confidence threshold decreased to {conf_threshold_yolov3_1:.2f}")
                else:
                    conf_threshold_yolov3_2 = max(0.05, conf_threshold_yolov3_2 - 0.05)
                    print(f"Camera 2 YOLOv3 confidence threshold decreased to {conf_threshold_yolov3_2:.2f}")

               
                # Camera 1
                stop_thread_1 = True
                if thread1.is_alive():
                    thread1.join(1.0)
                if cap1 is not None:
                    cap1.release()
               
                # Camera 2
                stop_thread_2 = True
                if thread2.is_alive():
                    thread2.join(1.0)
                if cap2 is not None:
                    cap2.release()
               
                # Reconnect both cameras
                cap1 = connect_to_cctv_1()
                cap2 = connect_to_cctv_2()
               
                # Restart threads
                stop_thread_1 = False
                thread1 = threading.Thread(target=read_frames_thread_1)
                thread1.daemon = True
                thread1.start()
               
                stop_thread_2 = False
                thread2 = threading.Thread(target=read_frames_thread_2)
                thread2.daemon = True
                thread2.start()
               
                # Reset counters
                bad_frame_counter_1 = 0
                bad_frame_counter_2 = 0
                last_reconnect_time_1 = current_time
                last_reconnect_time_2 = current_time
            elif key == ord('n'):
                # Try next RTSP URL format for active camera
                if active_camera == 1:
                    print("Trying next RTSP URL format for Camera 1...")
                    rtsp_url_index_1 = (rtsp_url_index_1 + 1) % len(RTSP_URLS_1)
                   
                    # Stop thread and release camera
                    stop_thread_1 = True
                    if thread1.is_alive():
                        thread1.join(1.0)
                    if cap1 is not None:
                        cap1.release()
                   
                    # Reconnect with new URL
                    cap1 = connect_to_cctv_1()
                   
                    # Restart thread
                    stop_thread_1 = False
                    thread1 = threading.Thread(target=read_frames_thread_1)
                    thread1.daemon = True
                    thread1.start()
                   
                    bad_frame_counter_1 = 0
                    last_reconnect_time_1 = current_time
                else:
                    print("Trying next RTSP URL format for Camera 2...")
                    rtsp_url_index_2 = (rtsp_url_index_2 + 1) % len(RTSP_URLS_2)
                   
                    # Stop thread and release camera
                    stop_thread_2 = True
                    if thread2.is_alive():
                        thread2.join(1.0)
                    if cap2 is not None:
                        cap2.release()
                   
                    # Reconnect with new URL
                    cap2 = connect_to_cctv_2()
                   
                    # Restart thread
                    stop_thread_2 = False
                    thread2 = threading.Thread(target=read_frames_thread_2)
                    thread2.daemon = True
                    thread2.start()
                   
                    bad_frame_counter_2 = 0
                    last_reconnect_time_2 = current_time
            elif key == ord('d'):
                # Toggle dashboard view
                dashboard_mode = not dashboard_mode
                print(f"Dashboard view: {'ON' if dashboard_mode else 'OFF'}")
            elif key == ord('1'):
                # Switch active control to Camera 1
                active_camera = 1
                print("Active control switched to Camera 1")
            elif key == ord('2'):
                # Switch active control to Camera 2
                active_camera = 2
                print("Active control switched to Camera 2")
       
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up resources
        print("Cleaning up resources...")
       
        # Stop threads
        stop_thread_1 = True
        stop_thread_2 = True
       
        # Wait for threads to finish
        if 'thread1' in locals() and thread1.is_alive():
            thread1.join(1.0)
        if 'thread2' in locals() and thread2.is_alive():
            thread2.join(1.0)
       
        # Release camera captures
        if 'cap1' in locals() and cap1 is not None:
            cap1.release()
        if 'cap2' in locals() and cap2 is not None:
            cap2.release()
       
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Application terminated successfully")

if __name__ == "__main__":
    main()