import os
import random
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

# Paths and Parameters
dataset_path = "./dataset"  # Your main dataset folder
output_path = "processed_dataset"  # Folder to save split datasets
img_size = 128  # Resize to 128x128
random.seed(42)

# Create directories for train, validation, and test sets
def create_directories(base_path, categories):
    for category in categories:
        os.makedirs(os.path.join(base_path, "train", category), exist_ok=True)
        os.makedirs(os.path.join(base_path, "val", category), exist_ok=True)
        os.makedirs(os.path.join(base_path, "test", category), exist_ok=True)

# Resize and normalize image
def preprocess_image(image_path, img_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize pixel values
    return img

# Split dataset into train, validation, and test
def split_dataset(dataset_path, output_path, img_size, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    categories = os.listdir(dataset_path)
    create_directories(output_path, categories)

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith((".jpg", ".png", ".jpeg"))]
        
        # Shuffle and split
        train_imgs, test_imgs = train_test_split(images, test_size=(1 - train_ratio))
        val_imgs, test_imgs = train_test_split(test_imgs, test_size=test_ratio / (test_ratio + val_ratio))

        # Save images to respective folders
        for train_img in train_imgs:
            shutil.copy(train_img, os.path.join(output_path, "train", category))
        for val_img in val_imgs:
            shutil.copy(val_img, os.path.join(output_path, "val", category))
        for test_img in test_imgs:
            shutil.copy(test_img, os.path.join(output_path, "test", category))
    
    print("Dataset splitting and preprocessing complete!")

# Process images
split_dataset(dataset_path, output_path, img_size)
