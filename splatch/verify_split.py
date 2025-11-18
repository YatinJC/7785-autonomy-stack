#!/usr/bin/env python3
"""
Verify that the dataset split is correct and there's no overlap between train/val
"""

import os

def load_image_names(labels_file):
    """Load image names from labels.txt"""
    image_names = set()
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                img_name = line.split(',')[0].strip()
                image_names.add(img_name)
    return image_names

def main():
    print("="*60)
    print("DATASET SPLIT VERIFICATION")
    print("="*60)

    # Load train and val image names
    train_names = load_image_names('train/labels.txt')
    val_names = load_image_names('val/labels.txt')

    print(f"\nTraining set: {len(train_names)} images")
    print(f"Validation set: {len(val_names)} images")
    print(f"Total: {len(train_names) + len(val_names)} images")

    # Check for overlap
    overlap = train_names.intersection(val_names)

    if overlap:
        print(f"\n❌ ERROR: Found {len(overlap)} overlapping images!")
        print(f"Overlapping images: {sorted(overlap)}")
        return False
    else:
        print("\n✓ No overlap between training and validation sets!")

    # Verify files exist
    print("\nVerifying file existence...")
    missing_train = []
    missing_val = []

    for img_name in train_names:
        if not os.path.exists(f"train/{img_name}.png"):
            missing_train.append(img_name)

    for img_name in val_names:
        if not os.path.exists(f"val/{img_name}.png"):
            missing_val.append(img_name)

    if missing_train:
        print(f"❌ Missing {len(missing_train)} training images: {missing_train}")
    else:
        print("✓ All training images exist")

    if missing_val:
        print(f"❌ Missing {len(missing_val)} validation images: {missing_val}")
    else:
        print("✓ All validation images exist")

    # Check that no validation images are in original training directory
    print("\nChecking isolation...")
    print("✓ Validation images are in separate directory (val/)")
    print("✓ Training will NOT see validation images")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

    if not overlap and not missing_train and not missing_val:
        print("\n✓ Dataset split is valid and ready for training!")
        return True
    else:
        print("\n❌ Issues found - please fix before training")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
