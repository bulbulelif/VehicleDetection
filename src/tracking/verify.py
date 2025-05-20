import torch
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ckpt.t7")

print(f"Checking model at: {model_path}")

try:
    # Try to load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print("✅ Model loaded successfully!")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print("Model type:", type(model))
except Exception as e:
    print(f"❌ Error loading model: {e}")