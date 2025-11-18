#!/usr/bin/env python3
"""
Split dataset into training and validation directories
Creates separate folders and labels.txt files
"""

import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

# Configuration
SOURCE_DIR = '2025F_imgs'
SOURCE_LABELS = '2025F_imgs/labels.txt'
TRAIN_DIR = 'train'
VAL_DIR = 'val'
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

def load_labels(labels_file):
    """Load image names and labels from labels.txt"""
    image_names = []
    labels = []

    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                img_name, label = line.split(',')
                img_name = img_name.strip()
                label = int(label.strip())

                # Check if image exists
                img_path = os.path.join(SOURCE_DIR, f"{img_name}.png")
                if os.path.exists(img_path):
                    image_names.append(img_name)
                    labels.append(label)

    return image_names, labels

def create_directories():
    """Create train and val directories"""
    for directory in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(directory):
            print(f"Removing existing {directory}/ directory...")
            shutil.rmtree(directory)
        os.makedirs(directory)
        print(f"✓ Created {directory}/ directory")

def split_and_copy_data(image_names, labels):
    """Split data and copy images to train/val directories"""

    # Stratified split to maintain class distribution
    train_names, val_names, train_labels, val_labels = train_test_split(
        image_names, labels,
        train_size=TRAIN_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels  # Ensures each class is proportionally represented
    )

    print(f"\nSplitting {len(image_names)} images:")
    print(f"  Training: {len(train_names)} images")
    print(f"  Validation: {len(val_names)} images")

    # Copy training images
    print(f"\nCopying training images to {TRAIN_DIR}/...")
    for img_name in train_names:
        src = os.path.join(SOURCE_DIR, f"{img_name}.png")
        dst = os.path.join(TRAIN_DIR, f"{img_name}.png")
        shutil.copy2(src, dst)

    # Copy validation images
    print(f"Copying validation images to {VAL_DIR}/...")
    for img_name in val_names:
        src = os.path.join(SOURCE_DIR, f"{img_name}.png")
        dst = os.path.join(VAL_DIR, f"{img_name}.png")
        shutil.copy2(src, dst)

    print("✓ Images copied successfully")

    return train_names, train_labels, val_names, val_labels

def write_labels_file(directory, image_names, labels):
    """Write labels.txt file for a directory"""
    labels_path = os.path.join(directory, 'labels.txt')

    with open(labels_path, 'w') as f:
        for img_name, label in zip(image_names, labels):
            f.write(f"{img_name}, {label}\n")

    print(f"✓ Created {labels_path}")

def print_statistics(train_labels, val_labels):
    """Print dataset statistics"""
    class_names = ['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']

    print("\n" + "="*60)
    print("DATASET SPLIT STATISTICS")
    print("="*60)

    print("\nClass Distribution:")
    print(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Total':<10}")
    print("-" * 60)

    train_counts = np.bincount(train_labels, minlength=6)
    val_counts = np.bincount(val_labels, minlength=6)

    for i, name in enumerate(class_names):
        total = train_counts[i] + val_counts[i]
        print(f"{name:<15} {train_counts[i]:<10} {val_counts[i]:<10} {total:<10}")

    print("-" * 60)
    print(f"{'TOTAL':<15} {len(train_labels):<10} {len(val_labels):<10} {len(train_labels) + len(val_labels):<10}")

    print("\nPercentage Split:")
    total = len(train_labels) + len(val_labels)
    train_pct = 100 * len(train_labels) / total
    val_pct = 100 * len(val_labels) / total
    print(f"  Training: {train_pct:.1f}%")
    print(f"  Validation: {val_pct:.1f}%")

    print("\n" + "="*60)

def main():
    print("="*60)
    print("DATASET SPLITTING SCRIPT")
    print("="*60)

    # Load data
    print(f"\nLoading data from {SOURCE_LABELS}...")
    image_names, labels = load_labels(SOURCE_LABELS)
    print(f"✓ Loaded {len(image_names)} images")

    # Create directories
    print("\nCreating directories...")
    create_directories()

    # Split and copy data
    train_names, train_labels, val_names, val_labels = split_and_copy_data(
        image_names, labels
    )

    # Write labels files
    print("\nCreating labels files...")
    write_labels_file(TRAIN_DIR, train_names, train_labels)
    write_labels_file(VAL_DIR, val_names, val_labels)

    # Print statistics
    print_statistics(train_labels, val_labels)

    print("\n✓ Dataset split complete!")
    print(f"\nYou can now train using:")
    print(f"  python3 train_mobilenetv4.py")
    print(f"\nThe training script will automatically use:")
    print(f"  - {TRAIN_DIR}/ for training")
    print(f"  - {VAL_DIR}/ for validation")

if __name__ == '__main__':
    main()
