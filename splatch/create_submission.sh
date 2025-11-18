#!/bin/bash
# Helper script to create Lab 6 submission zip

echo "================================================"
echo "Lab 6 Vision Checkpoint - Submission Creator"
echo "================================================"

# Get last names
echo ""
echo "Enter your last names (e.g., Smith_Johnson):"
read -p "LastName1_LastName2: " names

if [ -z "$names" ]; then
    echo "Error: No names provided!"
    exit 1
fi

ZIPFILE="${names}_Lab6.zip"

echo ""
echo "Creating submission: $ZIPFILE"
echo ""

# Check required files exist
echo "Checking required files..."
files=(
    "train_mobilenetv4.py"
    "mobilenetv4_sign_classifier.pth"
    "model_grader.py"
    "requirements.txt"
    "readme.txt"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file (MISSING)"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "Error: Some required files are missing!"
    exit 1
fi

# Create zip
echo ""
echo "Creating zip file..."
zip -q "$ZIPFILE" \
    train_mobilenetv4.py \
    mobilenetv4_sign_classifier.pth \
    model_grader.py \
    requirements.txt \
    readme.txt

if [ $? -eq 0 ]; then
    echo "✓ Zip file created successfully!"
    echo ""
    echo "================================================"
    echo "Submission Details:"
    echo "================================================"
    echo "File: $ZIPFILE"
    echo "Size: $(du -h "$ZIPFILE" | cut -f1)"
    echo ""
    echo "Contents:"
    unzip -l "$ZIPFILE"
    echo ""
    echo "================================================"
    echo "Next Steps:"
    echo "================================================"
    echo "1. Review SUBMISSION_CHECKLIST.md"
    echo "2. Test: python3 model_grader.py --data_path ./val --model_path ./mobilenetv4_sign_classifier.pth"
    echo "3. Upload $ZIPFILE to Canvas"
    echo "4. Deadline: November 14th, 2025 at 11:59 PM"
    echo "================================================"
else
    echo "Error: Failed to create zip file!"
    exit 1
fi
