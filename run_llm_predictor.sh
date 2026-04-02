#!/bin/bash
# Run LLM-based disk failure prediction
# Replaces the MOA Java simulation

# Configuration
TRAIN_PATH="./pyloader/train/"
TEST_PATH="./pyloader/test/"
START_DATE="2015-01-30"
ITERATIONS=30
MODEL="deepseek-ai/DeepSeek-V3.2"
THRESHOLD=0.5
BATCH_SIZE=10

# API Key - set via environment variable for security
# export SILICONFLOW_API_KEY="your-api-key-here"

# Check if API key is set
if [ -z "$SILICONFLOW_API_KEY" ]; then
    echo "Error: SILICONFLOW_API_KEY environment variable is not set"
    echo "Please set it with: export SILICONFLOW_API_KEY='your-api-key'"
    exit 1
fi

echo "=========================================="
echo "LLM-based Disk Failure Prediction"
echo "=========================================="
echo "Model: $MODEL"
echo "Start Date: $START_DATE"
echo "Iterations: $ITERATIONS"
echo "Train Path: $TRAIN_PATH"
echo "Test Path: $TEST_PATH"
echo "=========================================="

# Run predictor
python llm_predictor/predictor.py \
    -s "$START_DATE" \
    -p "$TRAIN_PATH" \
    -t "$TEST_PATH" \
    -i "$ITERATIONS" \
    -m "$MODEL" \
    --threshold "$THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    -o "llm_output/results.json"

echo "=========================================="
echo "Prediction completed!"
echo "Results saved to: llm_output/results.json"
echo "=========================================="
