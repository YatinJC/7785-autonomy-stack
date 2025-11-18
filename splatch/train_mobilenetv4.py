#!/usr/bin/env python3
"""
Training script for sign classification using MobileNetV4 with PyTorch
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import timm  # For MobileNetV4

# Configuration
CONFIG = {
    'train_dir': 'train',
    'val_dir': 'val',
    'train_labels': 'train/labels.txt',
    'val_labels': 'val/labels.txt',
    'model_save_path': 'mobilenetv4_sign_classifier.pth',
    'num_classes': 6,
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'image_size': 224,
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class SignDataset(Dataset):
    """Custom dataset for sign images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(labels_file, data_dir):
    """Load image paths and labels from labels.txt"""
    image_paths = []
    labels = []

    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                img_name, label = line.split(',')
                img_name = img_name.strip()
                label = int(label.strip())

                img_path = os.path.join(data_dir, f"{img_name}.png")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(label)

    return image_paths, labels

def get_transforms(train=True):
    """Get data transforms with augmentation for training"""
    if train:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def create_model(num_classes):
    """Create MobileNetV4 model with transfer learning"""
    # Try to load MobileNetV4, fallback to MobileNetV3 if not available
    try:
        model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',
                                  pretrained=True,
                                  num_classes=num_classes)
        print("Using MobileNetV4")
    except:
        try:
            model = timm.create_model('mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',
                                      pretrained=True,
                                      num_classes=num_classes)
            print("Using MobileNetV4 Hybrid")
        except:
            # Fallback to MobileNetV3
            model = timm.create_model('mobilenetv3_large_100',
                                      pretrained=True,
                                      num_classes=num_classes)
            print("Fallback to MobileNetV3 Large")

    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    print(f"Using device: {CONFIG['device']}")

    # Load training data
    print("Loading training data...")
    train_paths, train_labels = load_data(CONFIG['train_labels'], CONFIG['train_dir'])
    print(f"Training samples: {len(train_paths)}")
    print(f"Training class distribution: {np.bincount(train_labels)}")

    # Load validation data (completely separate, unseen during training)
    print("\nLoading validation data...")
    val_paths, val_labels = load_data(CONFIG['val_labels'], CONFIG['val_dir'])
    print(f"Validation samples: {len(val_paths)}")
    print(f"Validation class distribution: {np.bincount(val_labels)}")

    # Create datasets and dataloaders
    train_dataset = SignDataset(train_paths, train_labels, get_transforms(train=True))
    val_dataset = SignDataset(val_paths, val_labels, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=2)

    # Create model
    print("Creating model...")
    model = create_model(CONFIG['num_classes'])
    model = model.to(CONFIG['device'])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                       factor=0.5, patience=5)

    # Training loop
    print("Starting training...")
    best_val_acc = 0.0

    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, CONFIG['device'])
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader,
                                                            criterion, CONFIG['device'])

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, CONFIG['model_save_path'])
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")

    # Load best model and evaluate
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(CONFIG['model_save_path'])
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader,
                                                        criterion, CONFIG['device'])

    print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds,
                               target_names=['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))

    print(f"\nModel saved to: {CONFIG['model_save_path']}")

if __name__ == '__main__':
    main()
