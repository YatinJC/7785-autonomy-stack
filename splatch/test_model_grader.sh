#!/bin/bash
# Test the renamed model_grader.py

source $(conda info --base)/etc/profile.d/conda.sh
conda activate splatch

echo "Testing model_grader.py..."
python3 model_grader.py --data_path ./val --model_path ./mobilenetv4_sign_classifier.pth 2>&1 | tail -15
