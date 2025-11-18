#!/bin/bash
# Test generate_requirements.py

source $(conda info --base)/etc/profile.d/conda.sh
conda activate splatch

# Backup current requirements.txt
cp requirements.txt requirements.txt.backup

echo "Testing generate_requirements.py..."
python3 generate_requirements.py

echo ""
echo "Generated requirements.txt:"
cat requirements.txt

echo ""
echo "Restoring original requirements.txt..."
mv requirements.txt.backup requirements.txt

echo "✓ Test complete"
