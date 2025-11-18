#!/bin/bash
# Quick activation script for splatch environment

echo "Activating splatch conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate splatch

echo "✓ Environment activated!"
echo ""
echo "Available commands:"
echo "  python3 quick_test.py                     - Verify setup"
echo "  python3 train_mobilenetv4.py              - Train the model"
echo "  python3 model_grader.py --data_path ...   - Test the model"
echo ""
echo "To deactivate: conda deactivate"
