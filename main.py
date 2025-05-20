from ultralytics import YOLO
from IPython.display import display, Image
from IPython.display import Image as show_image

model = YOLO('models/yolov8/yolov8n.pt')

results = model.train(data='data.yaml', epochs = 100, imgsz = 640)

metrics = model.val()