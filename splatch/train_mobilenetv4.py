#!/usr/bin/env python3
"""
Training script for sign classification using MobileNetV4 with PyTorch
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    Focuses training on hard misclassified examples.

    FL(pt) = -α(1-pt)^γ * log(pt)

    Args:
        alpha: Weighting factor for class balance (default: 1.0)
        gamma: Focusing parameter for hard examples (default: 2.0)
        reduction: Specifies reduction to apply to output
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
        # Open as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Canny Edge Detection
        # Thresholds (100, 200) may need tuning based on your lighting
        edges = cv2.Canny(image, 100, 200)
        
        # Convert to PIL Image for transforms
        # Model input will now be 1-channel, not 3-channel
        image = Image.fromarray(edges) 
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
            # Removed RandomHorizontalFlip to preserve left/right orientation
            transforms.RandomRotation(degrees=8),  # Reduced from 15 to avoid confusing directional signs
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Simulate different viewing angles
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add robustness to blur
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Handle partial occlusions
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
    model = timm.create_model('mobilenetv3_small_100', pretrained=True, in_chans=1, num_classes=6)
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
    criterion = FocalLoss(alpha=1.0, gamma=2.0)  # Using Focal Loss for better handling of hard examples
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
