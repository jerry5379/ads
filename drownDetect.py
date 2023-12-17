import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = ["person"]  # For simplicity, we're detecting only people

# Open the video file
video_path = "videos/both2.mp4"
cap = cv2.VideoCapture(video_path)

# Define pool area coordinates (x, y, width, height)
pool_area = (100, 100, 300, 200)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 corresponds to person class
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Check if the detected person is inside the pool area
                x, y = center_x - w // 2, center_y - h // 2
                if pool_area[0] < x < pool_area[0] + pool_area[2] and pool_area[1] < y < pool_area[1] + pool_area[3]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangle for drowning person
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle for other people

    # Display the frame
    cv2.imshow("Drowning Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
