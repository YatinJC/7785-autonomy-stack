# Dataset Split Information

## Overview
The dataset has been split into separate **training** and **validation** directories to ensure the model never sees validation images during training.

## Directory Structure

```
splatch/
├── 2025F_imgs/          # Original images (kept for reference)
│   ├── labels.txt
│   └── *.png (158 images)
├── train/               # Training set (model sees these)
│   ├── labels.txt
│   └── *.png (126 images, 80%)
└── val/                 # Validation set (model NEVER sees these during training)
    ├── labels.txt
    └── *.png (32 images, 20%)
```

## Split Statistics

### Total Distribution
- **Training**: 126 images (79.7%)
- **Validation**: 32 images (20.3%)
- **Total**: 158 images

### Per-Class Distribution
| Class         | Train | Val | Total |
|---------------|-------|-----|-------|
| empty (0)     | 14    | 4   | 18    |
| left (1)      | 32    | 8   | 40    |
| right (2)     | 32    | 8   | 40    |
| do_not_enter (3) | 16 | 4   | 20    |
| stop (4)      | 16    | 4   | 20    |
| goal (5)      | 16    | 4   | 20    |

## Key Features

### ✓ Stratified Split
- Each class is proportionally represented in train and validation sets
- Ensures no class bias in evaluation

### ✓ No Overlap
- Training and validation sets are completely separate
- Validation images are never seen during training
- True measure of model generalization

### ✓ Reproducible
- Split uses random seed 42 for reproducibility
- Same split every time you run `split_dataset.py`

## How to Use

### 1. Create the split (already done)
```bash
python3 split_dataset.py
```

### 2. Verify the split
```bash
python3 verify_split.py
```

### 3. Train the model
```bash
conda activate splatch
python3 train_mobilenetv4.py
```
The training script automatically uses:
- `train/` for training
- `val/` for validation

### 4. Test on validation set
```bash
python3 model_grader.py --data_path ./val --model_path ./mobilenetv4_sign_classifier.pth
```

## Re-splitting the Dataset

If you need to re-split (e.g., different train/val ratio):

1. Edit `TRAIN_SPLIT` in `split_dataset.py`
2. Run: `python3 split_dataset.py`
3. Verify: `python3 verify_split.py`

## Important Notes

⚠️ **Do not manually modify train/ or val/ directories**
- Always use `split_dataset.py` to ensure proper stratification

⚠️ **Keep original 2025F_imgs/ directory**
- Serves as backup and source for re-splitting

✓ **Validation set isolation ensures honest evaluation**
- Your reported accuracy on validation set is trustworthy
- Prevents data leakage and overfitting

## Comparison: Before vs After

### Before (Programmatic Split)
```python
# In training script
train_test_split(images, labels, train_size=0.8)
# Risk: Might accidentally use validation data
```

### After (Physical Separation)
```python
# Load from separate directories
train_data = load_data('train/labels.txt', 'train/')
val_data = load_data('val/labels.txt', 'val/')
# Guarantee: Impossible to use validation data in training
```

## Files

| File | Purpose |
|------|---------|
| `split_dataset.py` | Create train/val split |
| `verify_split.py` | Verify no overlap exists |
| `train_mobilenetv4.py` | Training script (uses split) |
| `model_grader.py` | Evaluation script |

---

**Last Updated**: Dataset split created on 2025-11-11
**Random Seed**: 42 (for reproducibility)
