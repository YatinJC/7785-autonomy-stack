#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate splatch

echo "Testing pipreqs module..."
python3 -m pipreqs.pipreqs --help 2>&1 | head -10

echo ""
echo "Trying alternative command..."
pipreqs --help 2>&1 | head -10

echo ""
echo "Testing on temp directory with a simple file..."
mkdir -p /tmp/test_pipreqs
echo "import numpy" > /tmp/test_pipreqs/test.py
python3 -m pipreqs.pipreqs --encoding=utf-8 /tmp/test_pipreqs 2>&1
cat /tmp/test_pipreqs/requirements.txt 2>&1
rm -rf /tmp/test_pipreqs
