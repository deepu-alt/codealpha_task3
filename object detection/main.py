import cv2
from ultralytics import YOLO
from utils.detector import detect_objects
from utils.tracker import initialize_tracker, update_tracker
from utils.drawing import draw_tracks

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = initialize_tracker()

# Webcam
cap = cv2.VideoCapture(0)
# For video file use:
# cap = cv2.VideoCapture("videos/sample_video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    detections = detect_objects(model, frame)

    # Track objects
    tracks = update_tracker(tracker, detections, frame)

    # Draw boxes and IDs
    draw_tracks(frame, tracks)

    # Show output
    cv2.imshow("Object Detection and Tracking", frame)

    # Exit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()