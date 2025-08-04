#!/bin/bash

# Setup script for Whisper fine-tuning project

echo "Setting up Whisper fine-tuning environment..."

# Create virtual environment
python3 -m venv whisper_env
source whisper_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create directory structure
mkdir -p data models logs results config

# Download sample config if not exists
if [ ! -f "config/config.yaml" ]; then
    echo "Config file created. Please update with your settings."
fi

echo "Setup complete!"
echo "To activate the environment, run: source whisper_env/bin/activate"
echo "To start training, run: python train.py --config config/config.yaml"
