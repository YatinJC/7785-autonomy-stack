#!/bin/bash
# Test model_grader2.py

source $(conda info --base)/etc/profile.d/conda.sh
conda activate splatch

echo "Testing model_grader2.py on validation set..."
python3 model_grader2.py --data_path ./val --model_path ./mobilenetv4_sign_classifier.pth
