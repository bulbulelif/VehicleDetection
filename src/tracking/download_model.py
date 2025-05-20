import os
import urllib.request
import sys

def download_deepsort_model():
    # Get the project root directory (2 levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define model path at the project root
    model_dir = os.path.join(project_root, "models")
    model_path = os.path.join(model_dir, "ckpt.t7")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Several alternative URLs for the DeepSORT model
    urls = [
        "https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/raw/master/deep_sort/deep/checkpoint/ckpt.t7",
        "https://github.com/ZQPei/deep_sort_pytorch/raw/master/deep_sort/deep/checkpoint/ckpt.t7",
        "https://drive.google.com/uc?export=download&id=1_qwTWdzT9dWNXum2OQ0dEqZadRzpkW8h"
    ]
    
    print(f"Project root: {project_root}")
    print(f"Attempting to download DeepSORT model to: {model_path}")
    
    success = False
    for url in urls:
        try:
            print(f"Trying URL: {url}")
            # Download the file
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Download successful! Model saved to: {model_path}")
            print(f"   File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            success = True
            break
        except Exception as e:
            print(f"❌ Download failed: {e}")
    
    if not success:
        print("\nAll download attempts failed. Please try manual download:")
        print("1. Go to: https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch")
        print("2. Navigate to: deep_sort/deep/checkpoint/ckpt.t7")
        print("3. Download the file and place it at:", model_path)
        
    return success

if __name__ == "__main__":
    success = download_deepsort_model()
    sys.exit(0 if success else 1)