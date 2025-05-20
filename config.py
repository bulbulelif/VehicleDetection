import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Citation references are included in comments, e.g., [cite: 11], [cite: 3]
# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
DATA_YAML_PATH = os.path.join(DATA_DIR, 'data.yaml')
VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')
INPUT_VIDEO_PATH = os.path.join(VIDEOS_DIR, 'Traffic.mp4')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
YOLOV8_MODEL_DIR = os.path.join(MODELS_DIR, 'yolov8')
YOLOV8_WEIGHTS_PATH = os.path.join(YOLOV8_MODEL_DIR, 'best.pt')

# Source paths
SRC_DIR = os.path.join(BASE_DIR, 'src')
DATA_LOADING_SCRIPT = os.path.join(SRC_DIR, 'data_loading.py')
YOLOV8_DETECTOR_SCRIPT = os.path.join(SRC_DIR, 'detection', 'yolov8_detector.py')
TRACKER_SCRIPT = os.path.join(SRC_DIR, 'tracking', 'tracker.py')
COUNTING_SCRIPT = os.path.join(SRC_DIR, 'counting.py')
REAL_TIME_COUNTING_SCRIPT = os.path.join(SRC_DIR, 'real_time_counting.py')
UTILS_SCRIPT = os.path.join(SRC_DIR, 'utils.py')

# Training parameters
EPOCHS = 50 
BATCH_SIZE = 16 # Example value, adjust as needed
IMG_SIZE = 640 # Example value, adjust as needed

# Vehicle classes (adjust based on your dataset)
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

# Tracking parameters
MAX_AGE = 30 # Maximum number of frames to keep a track without new detections
MIN_HITS = 3 # Minimum number of hits required to establish a track

# Counting Line (example coordinates - adjust for your video)
# Format: (x1, y1, x2, y2)
COUNTING_LINE_COORDS = (100, 300, 1180, 300)