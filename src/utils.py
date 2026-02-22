"""Utility functions."""

import logging
import os
import json
import dataclasses
import re
import yaml
from pathlib import Path
from typing import Dict, Any

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _json_default(value: Any):
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for fair WER/CER comparison."""
    if not text:
        return ""

    normalized = text
    normalized = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", normalized)
    normalized = normalized.replace("ـ", "")
    normalized = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    normalized = normalized.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_directory_structure():
    """Create necessary directories."""
    directories = [
        "data",
        "models",
        "logs",
        "results",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_environment():
    """Validate environment setup."""
    import torch
    import transformers
    import datasets
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Datasets version: {datasets.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
