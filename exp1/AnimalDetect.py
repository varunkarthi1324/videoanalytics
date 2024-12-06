import cv2
import numpy as np
import os

# File paths
yolo_weights = r"D:\fifth project\me\video\yolov4-tiny.weights"  # Update the path
yolo_config = r"D:\fifth project\me\video\yolov4-tiny.cfg"      # Update the path
coco_names = r"D:\fifth project\me\video\coco.names"             # Update the path
video_path = r"D:\fifth project\me\video\sample.mp4"             # Update the path
output_path = r"D:\fifth project\me\video\output_video_animals.avi"  # Update the path

# Check if the files exist
if not all(os.path.exists(path) for path in [yolo_weights, yolo_config, coco_names, video_path]):
    raise FileNotFoundError("One or more required files (weights, config, coco.names, or video) are missing. Check the file paths.")

# Load class names
with open(coco_names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Focus on relevant animal-related classes
lion_king_classes = {"lion", "giraffe", "zebra", "bird", "elephant"}

# Load YOLO network
net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open the video file
cap = cv2.VideoCapture(video_path)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
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

    boxes = []
    confidences = []
    class_ids = []

    # Analyze the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)

            # Validate class_id
            if class_id < len(classes):
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in lion_king_classes:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Debugging: Print valid detections
                    print(f"Detected {classes[class_id]} with confidence {confidence}")
            else:
                # Debugging: Invalid class_id detected
                print(f"Invalid class_id: {class_id}")

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write the processed frame
    out.write(frame)
    cv2.imshow("Animal Detection", frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at {output_path}")
