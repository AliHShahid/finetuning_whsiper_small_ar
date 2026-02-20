"""Configuration management for Whisper fine-tuning."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    name: str = "openai/whisper-small"
    max_length: int = 225
    language: str = "en"
    task: str = "transcribe"

@dataclass
class DataConfig:
    source: str = "local_csv"
    csv_path: str = "./data/audio_list.csv"
    metadata_path: str = ""
    kaggle_dataset: str = ""
    kaggle_file_path: str = ""
    audio_column: str = "FilePath"
    text_column: str = "Transcript"
    duration_column: str = ""
    readerlist_path: str = ""
    allowed_readers: Optional[list] = None
    sampling_rate: int = 16000
    max_duration: float = 30.0
    min_duration: float = 1.0
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class TrainingConfig:
    output_dir: str = "./models/whisper-finetuned"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    seed: int = 42
    lr_scheduler_type: str = "linear"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    eval_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 200
    save_total_limit: int = 3
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    report_to: Optional[list] = None
    # report_to: list = field(default_factory=lambda: ["tensorboard"])


@dataclass
class HuggingFaceConfig:
    push_to_hub: bool = False
    hub_model_id: str = ""
    hub_private_repo: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)

def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Using default configuration.")
        return Config()
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dict to dataclass
    model_config = ModelConfig(**config_dict.get('model', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    huggingface_config = HuggingFaceConfig(**config_dict.get('huggingface', {}))
    
    return Config(
        model=model_config,
        data=data_config,
        training=training_config,
        huggingface=huggingface_config
    )

def get_device():
    """Get the appropriate device for training."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
