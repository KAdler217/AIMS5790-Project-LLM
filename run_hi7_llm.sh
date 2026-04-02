#!/bin/bash
# Run LLM-based prediction for Hitachi HDS722020ALA330 (hi7)
# Example: Processing 10-day data

# Disk model
DISK_MODEL="hi7"

# Configuration
START_DATE="2015-01-30"
ITERATIONS=10
LABEL_DAYS=6

# Paths
DATASET_PATH="~/trace/smart/all/"
TRAIN_PATH="./pyloader/${DISK_MODEL}_train/"
TEST_PATH="./pyloader/${DISK_MODEL}_test/"
OUTPUT_DIR="./${DISK_MODEL}_llm/"

# LLM Configuration
MODEL="deepseek-ai/DeepSeek-V3.2"
THRESHOLD=0.5
BATCH_SIZE=10
VALIDATION_WINDOW=30

echo "=========================================="
echo "LLM-based Disk Failure Prediction"
echo "Disk Model: Hitachi HDS722020ALA330 (hi7)"
echo "=========================================="

# Step 1: Python Preprocessing (using existing pyloader)
echo ""
echo "Step 1: Running Python preprocessing..."
cd pyloader

python run.py \
    -s "$START_DATE" \
    -a "$LABEL_DAYS" \
    -p "$DATASET_PATH" \
    -r "../${TRAIN_PATH}" \
    -e "../${TEST_PATH}" \
    -c "features_erg/hi7_all.txt" \
    -d "HDS722020ALA330" \
    -i "$ITERATIONS" \
    -t "sliding" \
    -w 30 \
    -L 7 \
    -V 30

cd ..

# Step 2: LLM-based Prediction (replacement for MOA Java)
echo ""
echo "Step 2: Running LLM-based prediction..."

# Check if API key is set
if [ -z "$SILICONFLOW_API_KEY" ]; then
    echo "Error: SILICONFLOW_API_KEY environment variable is not set"
    echo "Please set it with: export SILICONFLOW_API_KEY='your-api-key'"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python llm_predictor/predictor.py \
    -s "$START_DATE" \
    -p "$TRAIN_PATH" \
    -t "$TEST_PATH" \
    -i "$ITERATIONS" \
    -m "$MODEL" \
    --threshold "$THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    -o "${OUTPUT_DIR}/results.json" \
    2>&1 | tee "${OUTPUT_DIR}/prediction.log"

# Step 3: Parse results
echo ""
echo "Step 3: Parsing results..."
python llm_predictor/parse_results.py "${OUTPUT_DIR}/results.json"

echo ""
echo "=========================================="
echo "Completed!"
echo "Results directory: $OUTPUT_DIR"
echo "=========================================="
