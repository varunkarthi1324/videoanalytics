import cv2
import numpy as np
import os

# File paths (update these as needed)
yolo_weights = r"D:\fifth project\divya\video\yolov4-tiny.weights"
yolo_config = r"D:\fifth project\divya\video\yolov4-tiny.cfg"
coco_names = r"D:\fifth project\divya\video\coco.names"

# Check if the files exist
if not all(os.path.exists(path) for path in [yolo_weights, yolo_config, coco_names]):
    raise FileNotFoundError("One or more required files (weights, config, or coco.names) are missing. Check the file paths.")

# Load class names
with open(coco_names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Focus only on fruit-related classes
fruit_classes = {"banana", "apple", "orange", "pear", "watermelon", "grapes", "pineapple", 
                 "peach", "kiwi", "mango", "blueberry", "cherry", "plum", "strawberry", "lemon"}

# Load YOLO network
net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open the video file
video_path = r"D:\fifth project\divya\video\Sample2.mp4"  # Replace with your video path
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Input video file not found at {video_path}")

cap = cv2.VideoCapture(video_path)

# Get video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = r"D:\fifth project\divya\video\output_video.avi"
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Analyzing the outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in fruit_classes:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write the processed frame
    out.write(frame)
    cv2.imshow("Fruit Detection", frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at {output_path}")
