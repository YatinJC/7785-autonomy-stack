Dependencies:
    - torch >= 2.0.0
    - torchvision >= 0.15.0
    - timm >= 1.0.0 (for MobileNetV4)
    - opencv-python >= 4.8.0 (for image loading)
    - Pillow, numpy, scikit-learn

To evaluate the model on a validation dataset:

    python3 model_grader.py --data_path ./val_dataset --model_path ./mobilenetv4_sign_classifier.pth

Replace ./val_dataset with the path to the validation images directory.
