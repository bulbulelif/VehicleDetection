import cv2
import time
import os
import sys
import yaml
import torch  # Ensure torch is imported for device check

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.yolov8_detector import YOLOv8Detector
from src.tracking.tracker import VehicleTracker
from src.counting import VehicleCounter
from src.utils import draw_counting_line, check_gpu
from config import (
    YOLOV8_WEIGHTS_PATH,
    INPUT_VIDEO_PATH,
    VIDEOS_DIR,
    COUNTING_LINE_COORDS,
    VEHICLE_CLASSES,
    DATA_YAML_PATH
)


def run_real_time_counting(video_path=None, webcam_id=0):
    """
    Runs the real-time object detection, tracking, and counting system.

    Args:
        video_path (str, optional): Path to a video file. If None, uses webcam.
        webcam_id (int): ID of the webcam to use (e.g., 0 for default).
    """
    device = check_gpu()

    # Load class names from data.yaml
    class_names = {}
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            data_yaml = yaml.safe_load(f)
            class_names_list = data_yaml.get('names', [])
            class_names = {i: name for i, name in enumerate(class_names_list)}
            if not class_names:
                print(f"Warning: No 'names' found in {DATA_YAML_PATH}. Using default VEHICLE_CLASSES from config.py.")
                class_names = {i: name for i, name in enumerate(VEHICLE_CLASSES)}  # Fallback
            else:
                print(f"Loaded class names from data.yaml: {class_names}")
    except FileNotFoundError:
        print(f"Warning: {DATA_YAML_PATH} not found. Using default VEHICLE_CLASSES from config.py.")
        class_names = {i: name for i, name in enumerate(VEHICLE_CLASSES)}  # Fallback
    except Exception as e:
        print(f"Error loading {DATA_YAML_PATH}: {e}. Using default VEHICLE_CLASSES from config.py.")
        class_names = {i: name for i, name in enumerate(VEHICLE_CLASSES)}  # Fallback

    detector = YOLOv8Detector(model_weights=YOLOV8_WEIGHTS_PATH, device=device)
    tracker = VehicleTracker()
    counter = VehicleCounter(class_names)

    cap = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video from: {video_path}")
    else:
        print(f"Video file '{video_path}' not found or not specified. Attempting to use webcam {webcam_id}.")
        cap = cv2.VideoCapture(webcam_id)
        if not cap.isOpened():
            print(f"Error: Could not open webcam {webcam_id}.")
            return

    if not cap or not cap.isOpened():
        print("Error: Could not open video stream (neither video file nor webcam).")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        frame_count += 1

        # 1. Object Detection
        detections = detector.detect(frame)  # Returns list of {'box', 'conf', 'cls'}

        xyxy_detections = [d['box'] for d in detections]
        confidences = [d['conf'] for d in detections]
        class_ids = [d['cls'] for d in detections]

        # 2. Object Tracking
        tracked_objects = tracker.update(xyxy_detections, confidences, class_ids, frame)

        # 3. Vehicle Counting
        current_counts = counter.update(tracked_objects)

        # Display results on frame
        frame = draw_counting_line(frame)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, class_id = obj

            # Draw bounding box and track ID
            color = (0, 255, 0)  # Green for tracked objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Get class name
            class_name = class_names.get(class_id, f"Class {class_id}")

            text = f"ID: {track_id} {class_name}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display counts
        y_offset = 30
        for vehicle_type, count in current_counts.items():
            count_text = f"{vehicle_type.capitalize()} Count: {count}"
            cv2.putText(frame, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

        # Calculate and display FPS
        end_time = time.time()
        # Avoid division by zero if start_time and end_time are the same
        if (end_time - start_time) > 0:
            fps = frame_count / (end_time - start_time)
        else:
            fps = 0.0  # Or some other default, or handle as error.
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Real-Time Vehicle Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Ensure the 'videos' directory exists.
    if not os.path.exists(VIDEOS_DIR):
        os.makedirs(VIDEOS_DIR)

    # Use the video path from config.py
    target_video_path = INPUT_VIDEO_PATH

    # To run with the video file specified in config.py:
    if os.path.exists(target_video_path):
        run_real_time_counting(video_path=target_video_path)
    else:
        print(f"Video file not found at {target_video_path}. Attempting to use webcam.")
        run_real_time_counting(webcam_id=0)  # Fallback to webcam if video not found