
import cv2
import numpy as np

# Load YOLO model and configuration files
yolo_net = cv2.dnn.readNet('darknet-master/model/yolov3.weights', 'darknet-master/cfg/yolov3.cfg')
yolo_classes = []
with open('darknet-master/data/coco.names', 'r') as f:
    yolo_classes = [line.strip() for line in f.readlines()]

# Initialize a video capture object (use 0 for webcam or a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO network
    yolo_net.setInput(blob)

    # Get the output layer names
    output_layers = yolo_net.getUnconnectedOutLayersNames()

    # Run forward pass to get output from YOLO
    outs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(yolo_classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # BGR color for the bounding box (here, green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
