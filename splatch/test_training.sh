#!/bin/bash
# Quick test to verify training script can start

source $(conda info --base)/etc/profile.d/conda.sh
conda activate splatch

echo "Testing training script with new directory structure..."
timeout 15 python3 train_mobilenetv4.py 2>&1 | head -30

echo ""
echo "Test complete! If you see 'Loading training data...' above, the script is working correctly."
