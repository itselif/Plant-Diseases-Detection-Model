import cv2
import os
import random
import numpy as np
import albumentations as A
from pathlib import Path
from albumentations import RandomRotate90, HorizontalFlip, VerticalFlip, ColorJitter, RandomBrightnessContrast

# Augmentation processes
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),        # Horizontal flip
    A.VerticalFlip(p=0.5),         # Vertical flip
    A.RandomRotate90(p=0.5),       # Random 90-degree rotation
    A.RandomBrightnessContrast(p=0.2), # Brightness and contrast adjustment
    A.HueSaturationValue(p=0.2),   # Hue and saturation adjustment
    A.Resize(width=256, height=256, p=1.0) # Resizing
])

#%%

def augment_image(image_path, augment_count):
    """
    Applies augmentation to a specific image and saves the results.
    """
    image = cv2.imread(image_path)  # Read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Apply augmentation to each image
    for i in range(augment_count):
        augmented = augmentation(image=image)
        augmented_image = augmented['image']

        # Save the augmented image in the original folder
        image_name = os.path.basename(image_path).replace(".jpg", f"_aug_{i}.jpg")
        output_path = os.path.join(os.path.dirname(image_path), image_name)  # Save in the same folder
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        cv2.imwrite(output_path, augmented_image)
        print(f"Augmented image saved to {output_path}")

# Main folder path
input_directory = r'E:/new/New Plant Diseases Dataset(Augmented)/ekle/Wheat_SeptoriaBlotch'

# Get all image files in the folder
image_files = [f for f in os.listdir(input_directory) if f.endswith(".jpg")]

# Calculate how many augmentations to apply per image to reach a total of 1500 augmentations
augment_count = 1500 // len(image_files)  # Number of augmentations per image
extra_augments = 1500 % len(image_files)  # Remaining augmentations

# Apply augmentation to the image files
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(input_directory, image_file)
    
    # If there are extra augmentations, add +1 to some images
    extra = 1 if idx < extra_augments else 0
    
    augment_image(image_path, augment_count + extra)
    

#%%
# FOR TESTING

import os
import cv2

def augment_image(image_path, augment_count, output_directory):
    """
    Applies augmentation to the given image and saves the results to the specified folder.
    """
    image = cv2.imread(image_path)  # Read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Apply augmentation to each image
    for i in range(augment_count):
        augmented = augmentation(image=image)
        augmented_image = augmented['image']

        # Save the augmented image to the specified folder
        image_name = os.path.basename(image_path).replace(".jpg", f"_aug_{i}.jpg")
        output_path = os.path.join(output_directory, image_name)  # Save to the output folder
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        cv2.imwrite(output_path, augmented_image)
        print(f"Augmented image saved to {output_path}")

# Output folder path
output_directory = r'E:/new/New Plant Diseases Dataset(Augmented)/ekle/test/Wheat_SeptoriaBlotch'
os.makedirs(output_directory, exist_ok=True)  # Create the output folder if it doesn't exist

# Get all image files in the folder
image_files = [f for f in os.listdir(input_directory) if f.endswith(".jpg")]

# Calculate how many augmentations to apply per image to reach a total of 500 augmentations
augment_count = 500 // len(image_files)  # Number of augmentations per image
extra_augments = 500 % len(image_files)  # Remaining augmentations

# Apply augmentation to the image files
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(input_directory, image_file)
    
    # If there are extra augmentations, add +1 to some images
    extra = 1 if idx < extra_augments else 0
    
    augment_image(image_path, augment_count + extra, output_directory)
