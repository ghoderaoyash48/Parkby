import cv2
import numpy as np
import time
import os

# Print OpenCV version for debugging
print(f"OpenCV Version: {cv2.__version__}")

# Define paths
MODEL_PATH = "D:\\projectfolders\\YOLO\\"
CONFIG_FILE = os.path.join(MODEL_PATH, "yolov3.cfg")
WEIGHTS_FILE = os.path.join(MODEL_PATH, "yolov3.weights")
CLASSES_FILE = os.path.join(MODEL_PATH, "coco.names")

# Check if files exist
if not os.path.exists(CONFIG_FILE):
    print(f"ERROR: Config file not found: {CONFIG_FILE}")
    exit()
if not os.path.exists(WEIGHTS_FILE):
    print(f"ERROR: Weights file not found: {WEIGHTS_FILE}")
    exit()
if not os.path.exists(CLASSES_FILE):
    print(f"ERROR: Classes file not found: {CLASSES_FILE}")
    exit()

print("All model files found. Loading YOLO model...")

# Load class names
with open(CLASSES_FILE, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Load YOLO network
print("Loading neural network...")
try:
    net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)
    print("Network loaded successfully")
except Exception as e:
    print(f"ERROR loading network: {e}")
    exit()

# Get output layer names
try:
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    print(f"Output layers: {output_layers}")
except Exception as e:
    print(f"Error getting output layers: {e}")
    try:
        unconnected_out_layers = net.getUnconnectedOutLayers()
        if isinstance(unconnected_out_layers, np.ndarray):
            output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:
            output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        print(f"Alternative method output layers: {output_layers}")
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")
        exit()

# Define the vehicle classes we're interested in
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

# Define maximum parking capacity
MAX_PARKING_SPACES = 100

# Try all available camera indices
def try_cameras():
    print("Testing available cameras...")
    for i in range(10):  # Try indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} available - resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
            else:
                print(f"Camera {i} open but no frames")
                cap.release()
        else:
            print(f"Camera {i} not available")
    
    # Default to camera 1 (external webcam) if available, otherwise use 0
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("External webcam (index 1) not available, trying laptop camera (index 0)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No cameras available. Exiting.")
            exit()
    
    return cap

# Main function
def main():
    # Open camera
    cap = try_cameras()
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting vehicle detection...")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break
        
        frame_count += 1
        
        # Only process every 2nd frame for performance
        if frame_count % 2 != 0:
            continue
        
        # Prepare image for detection
        height, width, channels = frame.shape
        
        # Create a blob from the image
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        except Exception as e:
            print(f"Error creating blob: {e}")
            continue
        
        # Set the blob as input to the network
        try:
            net.setInput(blob)
        except Exception as e:
            print(f"Error setting input: {e}")
            continue
            
        # Run forward pass
        try:
            # Use this for more memory-efficient processing
            detections = net.forward(output_layers)
        except cv2.error as e:
            print(f"OpenCV error during detection: {e}")
            print("This may be due to insufficient memory. Try using YOLOv3-tiny instead.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and CLASSES[class_id] in VEHICLE_CLASSES:
                    # Object detected is a vehicle with good confidence
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        try:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        except:
            indices = []
        
        # Draw bounding boxes
        vehicle_count = 0
        
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            
            for i in indices:
                x, y, w, h = boxes[i]
                label = CLASSES[class_ids[i]]
                confidence = confidences[i]
                
                if label in VEHICLE_CLASSES:
                    vehicle_count += 1
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, 
                        f"{label} {confidence:.2f}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2
                    )
        
        # Calculate available parking
        available_parking = MAX_PARKING_SPACES - vehicle_count
        available_parking = max(0, available_parking)
        
        # Display counts
        cv2.putText(
            frame,
            f"Vehicle Count: {vehicle_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Available Parking: {available_parking}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Calculate and display FPS
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
        
        # Display the resulting frame
        cv2.imshow("Parking Detection", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()