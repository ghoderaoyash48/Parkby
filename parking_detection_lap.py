import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("D:\\projectfolders\\YOLO\\yolov3.weights", "D:\\projectfolders\\YOLO\\yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load COCO names for class labels
with open("D:\\projectfolders\\YOLO\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Capture video from camera
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Define maximum parking capacity
MAX_PARKING_SPACES = 78

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vehicle_count = 0

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in ["car", "truck", "bus", "motorbike"]:
                vehicle_count += 1
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate available parking spaces
    available_parking = MAX_PARKING_SPACES - vehicle_count
    available_parking = max(0, available_parking)  # Ensure it doesn't go negative

    # Display the vehicle count and parking availability
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Available Parking: {available_parking}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()