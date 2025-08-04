#!/bin/bash

# Inference script

set -e

MODEL_PATH=${1:-"models/whisper-finetuned"}
AUDIO_PATH=${2:-"data/test_audio"}

echo "Running inference with model: $MODEL_PATH"
echo "Audio path: $AUDIO_PATH"

if [ -f "$AUDIO_PATH" ]; then
    # Single file
    python3 inference.py \
        --model-path "$MODEL_PATH" \
        --audio-path "$AUDIO_PATH" \
        --output-file "results/transcription.json"
elif [ -d "$AUDIO_PATH" ]; then
    # Directory
    python3 inference.py \
        --model-path "$MODEL_PATH" \
        --audio-dir "$AUDIO_PATH" \
        --output-file "results/transcriptions.json"
else
    echo "Error: Audio path $AUDIO_PATH not found!"
    exit 1
fi

echo "Inference completed!"
