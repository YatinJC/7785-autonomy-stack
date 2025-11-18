#!/usr/bin/env python3
"""
Quick test script to verify setup before training
"""

import sys
import os

def check_imports():
    """Check if all required packages are installed"""
    print("Checking imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not found. Install: pip install torch torchvision")
        return False

    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError:
        print("✗ TorchVision not found. Install: pip install torchvision")
        return False

    try:
        import timm
        print(f"✓ timm {timm.__version__}")
    except ImportError:
        print("✗ timm not found. Install: pip install timm")
        return False

    try:
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError:
        print("✗ Pillow not found. Install: pip install Pillow")
        return False

    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not found. Install: pip install numpy")
        return False

    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn not found. Install: pip install scikit-learn")
        return False

    return True

def check_data():
    """Check if data directory exists"""
    print("\nChecking data...")

    if not os.path.exists('2025F_imgs'):
        print("✗ Data directory '2025F_imgs' not found!")
        return False
    print("✓ Data directory found")

    if not os.path.exists('2025F_imgs/labels.txt'):
        print("✗ labels.txt not found!")
        return False
    print("✓ labels.txt found")

    # Count images
    images = [f for f in os.listdir('2025F_imgs') if f.endswith('.png')]
    print(f"✓ Found {len(images)} PNG images")

    return True

def check_mobilenetv4():
    """Check if MobileNetV4 is available in timm"""
    print("\nChecking MobileNetV4 availability...")

    try:
        import timm

        # Try to list available models
        available_models = timm.list_models('mobilenetv4*')

        if available_models:
            print(f"✓ MobileNetV4 models available:")
            for model in available_models:
                print(f"  - {model}")
        else:
            print("! MobileNetV4 not found, will fallback to MobileNetV3")
            mobilev3_models = timm.list_models('mobilenetv3*')
            if mobilev3_models:
                print(f"✓ MobileNetV3 models available (fallback):")
                for model in mobilev3_models[:3]:
                    print(f"  - {model}")

        return True
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False

def test_model_creation():
    """Test if we can create the model"""
    print("\nTesting model creation...")

    try:
        import torch
        import timm

        # Try MobileNetV4
        try:
            model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',
                                      pretrained=False,
                                      num_classes=6)
            print("✓ Successfully created MobileNetV4 model")
            return True
        except:
            pass

        # Try alternative MobileNetV4
        try:
            model = timm.create_model('mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',
                                      pretrained=False,
                                      num_classes=6)
            print("✓ Successfully created MobileNetV4 Hybrid model")
            return True
        except:
            pass

        # Fallback to MobileNetV3
        model = timm.create_model('mobilenetv3_large_100',
                                  pretrained=False,
                                  num_classes=6)
        print("✓ Successfully created MobileNetV3 model (fallback)")
        return True

    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

def main():
    print("="*60)
    print("SIGN CLASSIFICATION SETUP VERIFICATION")
    print("="*60)

    all_good = True

    if not check_imports():
        all_good = False

    if not check_data():
        all_good = False

    if not check_mobilenetv4():
        all_good = False

    if not test_model_creation():
        all_good = False

    print("\n" + "="*60)
    if all_good:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou're ready to train. Run:")
        print("  python3 train_mobilenetv4.py")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before training.")
    print("="*60)

if __name__ == '__main__':
    main()
