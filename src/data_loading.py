import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import DATASET_DIR, DATA_YAML_PATH, TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE
import yaml

class VehicleDataset(Dataset):
    """
    Custom Dataset class for loading vehicle images and their annotations.
    Assumes YOLO format annotations (txt files with normalized bounding boxes).
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load annotations (assuming YOLO format: class_id x_center y_center width height)
        label_path = os.path.join(self.img_dir.replace('images', 'labels'), img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    labels.append(int(parts[0]))
                    boxes.append(parts[1:]) # Normalized x_center, y_center, width, height

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels, img_name

def get_transforms():
    """
    Defines image preprocessing and augmentation transforms.
    [cite: 5]
    """
    # Define transformations for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Data augmentation
        transforms.RandomRotation(10), # Data augmentation
        transforms.RandomHorizontalFlip(), # Data augmentation
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # Simulate noise
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def load_datasets(batch_size):
    """
    Loads training and validation datasets.
    [cite: 3, 4]
    """
    train_transform, val_transform = get_transforms()

    train_dataset = VehicleDataset(os.path.join(TRAIN_DIR, 'images'), transform=train_transform)
    val_dataset = VehicleDataset(os.path.join(VAL_DIR, 'images'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # You might want to explore the dataset here, understand structure and class diversity [cite: 4]
    # For example, print some stats or visualize samples.
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    # Load class names from data.yaml
    with open(DATA_YAML_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names']
        print(f"Detected classes: {class_names}")

    return train_loader, val_loader, class_names

if __name__ == '__main__':
    # Example usage:
    train_loader, val_loader, class_names = load_datasets(batch_size=4)
    for images, boxes, labels, img_names in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch boxes (first image) shape: {boxes[0].shape}")
        print(f"Batch labels (first image) shape: {labels[0].shape}")
        break