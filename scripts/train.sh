#!/bin/bash

# Training script with error handling

set -e

echo "Starting Whisper fine-tuning..."

# Check if config exists
if [ ! -f "config/config.yaml" ]; then
    echo "Error: config/config.yaml not found!"
    exit 1
fi

# Check if CSV file exists
CSV_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('config/config.yaml'))['data']['csv_path'])")
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file $CSV_PATH not found!"
    exit 1
fi

# Run training
python3 train.py \
    --config config/config.yaml \
    --log-level INFO

echo "Training completed!"
