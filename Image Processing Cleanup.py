# Image Processing Pipeline for Cleaning and Preparing Dataset
# This script processes images by detecting corrupted files, normalizing formats, and ensuring compatibility for training.

from PIL import Image
import os

dataset_path = "E:/Plant Diseases Dataset(Augmented)/train" 


# Step 1: Detect Corrupted Files
# Traverse the dataset directory to identify and list corrupted files.

def detect_corrupted_files(dataset_path):
    corrupted_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Check for corrupted files
            except (IOError, SyntaxError) as e:
                print(f"Corrupted file found: {file_path}")
                corrupted_files.append(file_path)
    print(f"Total corrupted files: {len(corrupted_files)}")
    return corrupted_files

# Step 2: Fix RGBA Images
# Convert RGBA images to RGB to avoid issues during training.

def fix_rgba_images(dataset_path="E:/Plant Diseases Dataset(Augmented)/train"):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                    img.save(file_path, "JPEG")  # Save as JPEG
            except (IOError, SyntaxError) as e:
                print(f"Error processing file: {file_path}")


# Step 3: Convert Image Formats to JPG
# Convert all images in the dataset to JPG format for consistency.

def convert_to_jpg(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.jpeg', '.png')):  # Target specific formats
                old_path = os.path.join(root, file)
                new_path = os.path.splitext(old_path)[0] + ".jpg"  # Change extension to .jpg
                try:
                    img = Image.open(old_path).convert("RGB")  # Ensure RGB mode for JPEG
                    img.save(new_path, "JPEG")
                    os.remove(old_path)  # Delete the original file
                    print(f"Converted and saved: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error processing file {old_path}: {e}")


# Step 4: Standardize File Extensions to Lowercase
# Ensure all file extensions are in lowercase to avoid inconsistencies.

def standardize_file_extensions(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            old_path = os.path.join(root, file)
            name, ext = os.path.splitext(file)
            if ext.isupper():
                new_path = os.path.join(root, name + ext.lower())
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")



