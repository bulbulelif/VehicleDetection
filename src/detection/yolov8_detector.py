import torch
from ultralytics import YOLO
from config import YOLOV8_WEIGHTS_PATH, EPOCHS, BATCH_SIZE, IMG_SIZE, DATA_YAML_PATH
import os
from src.utils import check_gpu

class YOLOv8Detector:
    def __init__(self, model_weights=YOLOV8_WEIGHTS_PATH, device=None):
        """
        Initializes the YOLOv8 detector.
        [cite: 6]
        """
        if device is None:
            self.device = check_gpu()
        else:
            self.device = device
        self.model = YOLO('yolov8n.pt') # Load pre-trained YOLOv8n model [cite: 6]
        if os.path.exists(model_weights):
            print(f"Loading custom weights from {model_weights}")
            self.model = YOLO(model_weights)
        self.model.to(self.device)
        print(f"YOLOv8 model loaded on {self.device}")

    def train(self, data_yaml_path=DATA_YAML_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
        """
        Trains the YOLOv8 model.
        [cite: 6, 11]
        """
        print(f"Starting YOLOv8 training for {epochs} epochs...")
        self.model.train(data=data_yaml_path,
                         epochs=epochs,
                         imgsz=img_size,
                         batch=batch_size,
                         val=True, # Enable validation monitoring [cite: 11]
                         device=self.device,
                         patience=20, # Early stopping patience
                         # Augmentation parameters can be added here
                         # e.g., mixup=0.1, mosaic=1.0 etc. to apply regularization/augmentation [cite: 11]
                         )
        self.model.save(YOLOV8_WEIGHTS_PATH)
        print(f"YOLOv8 model trained and saved to {YOLOV8_WEIGHTS_PATH}")

    def detect(self, image):
        """
        Performs object detection on a single image.
        Args:
            image (numpy.ndarray): Input image (H, W, C) in BGR format.
        Returns:
            list of dict: Each dict contains 'box' (xyxy), 'conf' (confidence), 'cls' (class ID).
        """
        results = self.model(image, verbose=False, device=self.device) # verbose=False to suppress output per frame
        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'conf': conf,
                        'cls': cls
                    })
        return detections

if __name__ == '__main__':
    # Example training
    detector = YOLOv8Detector()
    # Ensure your dataset is ready and data.yaml is configured correctly before running training.
    # detector.train()

    # Example detection
    import cv2
    dummy_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'dataset', 'val', 'images', 'some_image.jpg')
    if os.path.exists(dummy_image_path):
        img = cv2.imread(dummy_image_path)
        if img is not None:
            print(f"Performing detection on {dummy_image_path}")
            detections = detector.detect(img)
            print(f"Detections: {detections}")
            # Optionally, draw bounding boxes on the image and display
            for det in detections:
                x1, y1, x2, y2 = det['box']
                conf = det['conf']
                cls = det['cls']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Class {cls}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Detection Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Error: Could not load image from {dummy_image_path}. Please check path and file integrity.")
    else:
        print(f"Dummy image not found at {dummy_image_path}. Please create one or provide a valid path for testing detection.")