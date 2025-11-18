# Lab 6 Vision Checkpoint - Submission Checklist

## 🎯 Submission Results

✅ **Validation Accuracy: 100% (32/32 images)**

Confusion Matrix (Perfect Classification):
```
         empty  left  right  do_not_enter  stop  goal
empty      4      0     0         0         0     0
left       0      8     0         0         0     0
right      0      0     8         0         0     0
do_not_enter 0    0     0         4         0     0
stop       0      0     0         0         4     0
goal       0      0     0         0         0     4
```

## 📦 Submission Files (Required)

Your zip file must contain exactly these 5 files at the root level:

```
LastName1_LastName2_Lab6.zip
├── train_mobilenetv4.py                 ✓ (Training script)
├── mobilenetv4_sign_classifier.pth      ✓ (29MB - Trained model)
├── model_grader.py                      ✓ (Grading script - MODIFIED)
├── requirements.txt                     ✓ (Dependencies)
└── readme.txt                           ✓ (Instructions)
```

## ✅ Pre-Submission Verification

### Step 1: Verify all files exist
```bash
cd /home/yc/turtlebot3_ws/src/7785-autonomy-stack/splatch

# Check all required files
ls -lh train_mobilenetv4.py
ls -lh mobilenetv4_sign_classifier.pth
ls -lh model_grader.py
ls -lh requirements.txt
ls -lh readme.txt
```

### Step 2: Test the grader works
```bash
conda activate splatch
python3 model_grader.py --data_path ./val --model_path ./mobilenetv4_sign_classifier.pth
```

Expected output: **Total accuracy: 1.0** (100%)

### Step 3: Verify requirements.txt
```bash
cat requirements.txt
```

Should include:
- numpy
- opencv-python ← IMPORTANT (for grader)
- Pillow
- scikit-learn
- timm
- torch
- torchvision

### Step 4: Check paths are relative
```bash
grep -n "home/yc" train_mobilenetv4.py model_grader.py
```

Should return **nothing** (no hardcoded absolute paths)

## 📝 Create Submission Zip

Replace `YourLastName1` and `YourLastName2` with your actual last names:

```bash
cd /home/yc/turtlebot3_ws/src/7785-autonomy-stack/splatch

zip YourLastName1_YourLastName2_Lab6.zip \
    train_mobilenetv4.py \
    mobilenetv4_sign_classifier.pth \
    model_grader.py \
    requirements.txt \
    readme.txt

# Verify the zip contents
unzip -l YourLastName1_YourLastName2_Lab6.zip
```

## 🧪 Test Your Submission (Recommended)

Simulate the grading environment:

```bash
# Create test directory
mkdir -p /tmp/test_submission
cd /tmp/test_submission

# Extract your zip
unzip ~/turtlebot3_ws/src/7785-autonomy-stack/splatch/YourLastName1_YourLastName2_Lab6.zip

# Install dependencies
pip install -r requirements.txt

# Test with validation data (copy val directory to test)
cp -r ~/turtlebot3_ws/src/7785-autonomy-stack/splatch/val .

# Run grader
python3 model_grader.py --data_path ./val --model_path ./mobilenetv4_sign_classifier.pth
```

Expected: **Total accuracy: 1.0**

## 📤 Submit

1. Upload `YourLastName1_YourLastName2_Lab6.zip` to Canvas
2. Submission location: **Assignments → Lab 6**
3. Deadline: **November 14th, 2025 at 11:59 PM**

## ⚠️ Common Issues to Avoid

❌ **DO NOT include:**
- `train/` directory (graders have their own data)
- `val/` directory (graders have their own data)
- `2025F_imgs/` directory
- Helper scripts (split_dataset.py, verify_split.py, etc.)
- `__pycache__/` directories
- `.pyc` files

❌ **DO NOT:**
- Use absolute paths like `/home/yc/...`
- Modify function signatures in model_grader.py
- Change the grading script below the "DO NOT MODIFY" line

✅ **DO:**
- Use relative paths only
- Test the grader before submission
- Include all dependencies in requirements.txt
- Follow the exact file structure

## 📊 Expected Grade

Your grade = accuracy on hidden test set

With 100% on validation set, you should expect:
- **Best case**: 95-100% (if hidden set is similar)
- **Realistic**: 90-95% (accounting for some variation)
- **Model**: MobileNetV4 with transfer learning and data augmentation

## 🔑 Key Implementation Details

**Model**: MobileNetV4 (pretrained on ImageNet)
**Training**: 50 epochs, Adam optimizer, learning rate 0.001
**Data Split**: 80% train (126), 20% val (32), stratified by class
**Augmentation**: Random flip, rotation, color jitter, affine transforms
**Preprocessing**: Resize to 224x224, normalize with ImageNet stats

**Important**: The grader uses OpenCV (BGR format) but our model expects RGB - automatic conversion is handled in `predict()` function.

---

**Good luck!** 🎉
