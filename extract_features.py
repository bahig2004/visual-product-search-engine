"""
Feature Extraction Script
Extracts CNN features (ResNet50) from product images.
Saves feature vectors to HDF5 for efficient retrieval.
"""

import os
from pathlib import Path
import csv
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from tqdm import tqdm
import traceback

# Configuration
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
PROCESSED_DIR = BASE_DIR / "processed"
METADATA_DIR = PROCESSED_DIR / "metadata"
FEATURES_DIR = PROCESSED_DIR / "features"

# Create features directory
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_HDF5 = FEATURES_DIR / "image_features.h5"
FEATURES_MAPPING = FEATURES_DIR / "feature_mapping.json"
EXTRACTION_LOG = FEATURES_DIR / "extraction_log.txt"

BATCH_SIZE = 32
FEATURE_DIM = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


def get_model():
    """Load ResNet50 model for feature extraction."""
    print("Loading ResNet50 model...")
    model = resnet50(weights="ResNet50_Weights.DEFAULT")
    
    # Remove the classification layer to get features from avgpool
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(DEVICE)
    model.eval()
    
    return model


def get_transform():
    """Define image preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(image_path):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_image_paths():
    """Get all image paths from the local images directory."""
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    categories = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    
    for category_dir in sorted(categories):
        category_name = category_dir.name
        for img_file in category_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_paths.append({
                    "path": str(img_file),
                    "category": category_name,
                    "filename": img_file.name
                })
    
    return image_paths


def extract_features(model, image_batch, transform, device):
    """Extract features from a batch of images."""
    with torch.no_grad():
        images_tensor = torch.stack(image_batch).to(device)
        features = model(images_tensor)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, 2048)
        return features.cpu().numpy()


def main():
    """Main feature extraction pipeline."""
    
    print("=" * 80)
    print("FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    
    # Initialize model and transforms
    model = get_model()
    transform = get_transform()
    
    # Get all image paths
    print("\nScanning images directory...")
    image_list = get_image_paths()
    total_images = len(image_list)
    print(f"Found {total_images} images")
    
    if total_images == 0:
        print("No images found!")
        return
    
    # Initialize HDF5 file
    print(f"\nInitializing HDF5 file: {FEATURES_HDF5}")
    with h5py.File(FEATURES_HDF5, "w") as f:
        features_dataset = f.create_dataset(
            "features",
            shape=(total_images, FEATURE_DIM),
            dtype=np.float32,
            compression="gzip"
        )
        
        # Process images in batches
        feature_mapping = []
        current_batch = []
        batch_indices = []
        processed_count = 0
        error_count = 0
        
        print("\nExtracting features...")
        pbar = tqdm(total=total_images, desc="Extracting features")
        
        for idx, img_info in enumerate(image_list):
            img_path = img_info["path"]
            
            # Load image
            img = load_image(img_path)
            if img is None:
                error_count += 1
                pbar.update(1)
                continue
            
            # Preprocess image
            try:
                img_tensor = transform(img)
            except Exception as e:
                print(f"Error transforming image {img_path}: {e}")
                error_count += 1
                pbar.update(1)
                continue
            
            # Add to batch
            current_batch.append(img_tensor)
            batch_indices.append(idx)
            
            # Process batch when full or at end
            if len(current_batch) == BATCH_SIZE or idx == total_images - 1:
                features = extract_features(model, current_batch, transform, DEVICE)
                
                # Store features and metadata
                for batch_idx, feature_idx in enumerate(batch_indices):
                    features_dataset[feature_idx] = features[batch_idx]
                    feature_mapping.append({
                        "feature_idx": int(feature_idx),
                        "image_path": image_list[feature_idx]["path"],
                        "category": image_list[feature_idx]["category"],
                        "filename": image_list[feature_idx]["filename"]
                    })
                
                processed_count += len(current_batch)
                current_batch = []
                batch_indices = []
            
            pbar.update(1)
        
        pbar.close()
    
    # Save feature mapping
    print(f"\nSaving feature mapping: {FEATURES_MAPPING}")
    with open(FEATURES_MAPPING, "w") as f:
        json.dump(feature_mapping, f, indent=2)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total images processed: {processed_count}")
    print(f"Total errors: {error_count}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Features saved to: {FEATURES_HDF5}")
    print(f"Mapping saved to: {FEATURES_MAPPING}")
    
    # Log summary
    with open(EXTRACTION_LOG, "w") as log:
        log.write("FEATURE EXTRACTION LOG\n")
        log.write("=" * 80 + "\n")
        log.write(f"Device: {DEVICE}\n")
        log.write(f"Model: ResNet50\n")
        log.write(f"Feature dimension: {FEATURE_DIM}\n")
        log.write(f"Batch size: {BATCH_SIZE}\n")
        log.write(f"Total images found: {total_images}\n")
        log.write(f"Total images processed: {processed_count}\n")
        log.write(f"Total errors: {error_count}\n")
        log.write(f"Success rate: {100 * processed_count / total_images:.2f}%\n")
        log.write(f"Features file: {FEATURES_HDF5}\n")
        log.write(f"Mapping file: {FEATURES_MAPPING}\n")


if __name__ == "__main__":
    main()
