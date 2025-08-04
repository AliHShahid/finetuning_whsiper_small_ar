"""Data processing utilities for Whisper fine-tuning."""

import pandas as pd
import librosa
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, DatasetDict
from transformers import WhisperProcessor
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class WhisperDataProcessor:
    """Data processor for Whisper fine-tuning."""
    
    def __init__(self, processor: WhisperProcessor, config):
        self.processor = processor
        self.config = config
        self.sampling_rate = config.data.sampling_rate
        self.max_duration = config.data.max_duration
        self.min_duration = config.data.min_duration
        
    def load_and_validate_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and validate the CSV file."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows")
            
            # Validate required columns
            required_cols = [self.config.data.audio_column, self.config.data.class_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate file paths
            valid_files = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating audio files"):
                file_path = row[self.config.data.audio_column]
                if Path(file_path).exists():
                    valid_files.append(idx)
                else:
                    logger.warning(f"File not found: {file_path}")
            
            df_valid = df.iloc[valid_files].reset_index(drop=True)
            logger.info(f"Found {len(df_valid)} valid audio files out of {len(df)}")
            
            return df_valid
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def extract_audio_features(self, batch: Dict) -> Dict:
        """Extract audio features from file paths."""
        try:
            file_path = batch[self.config.data.audio_column]
            
            # Load audio
            waveform, sr = librosa.load(file_path, sr=self.sampling_rate)
            
            # Validate duration
            duration = len(waveform) / sr
            if duration < self.min_duration or duration > self.max_duration:
                logger.warning(f"Audio duration {duration:.2f}s outside valid range for {file_path}")
                # Pad or truncate as needed
                target_length = int(self.sampling_rate * min(duration, self.max_duration))
                if len(waveform) > target_length:
                    waveform = waveform[:target_length]
                elif len(waveform) < int(self.sampling_rate * self.min_duration):
                    # Pad with zeros
                    pad_length = int(self.sampling_rate * self.min_duration) - len(waveform)
                    waveform = np.pad(waveform, (0, pad_length), mode='constant')
            
            batch["audio"] = waveform
            batch["sampling_rate"] = self.sampling_rate
            return batch
            
        except Exception as e:
            logger.error(f"Error processing audio file {batch.get(self.config.data.audio_column, 'unknown')}: {e}")
            # Return empty audio for failed files
            batch["audio"] = np.zeros(int(self.sampling_rate * self.min_duration))
            batch["sampling_rate"] = self.sampling_rate
            return batch
    
    def prepare_features(self, batch: Dict) -> Dict:
        """Prepare features for model input."""
        try:
            # Process audio
            audio = batch["audio"]
            input_features = self.processor.feature_extractor(
                audio, 
                sampling_rate=batch["sampling_rate"], 
                return_tensors="pt"
            ).input_features[0]
            
            batch["input_features"] = input_features
            
            # For speaker identification, we can use the class as a label
            # For transcription, you would need transcription text
            speaker_class = batch[self.config.data.class_column]
            
            # Create a simple transcription prompt with speaker info
            # This is a placeholder - you might want actual transcriptions
            transcription = f"Speaker: {speaker_class}"
            
            # Tokenize labels
            labels = self.processor.tokenizer(
                transcription,
                return_tensors="pt",
                padding="max_length",
                max_length=self.config.model.max_length,
                truncation=True
            ).input_ids[0]
            
            # Replace padding token id with -100 for loss calculation
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            
            return batch
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return dummy features for failed samples
            batch["input_features"] = torch.zeros((80, 3000))
            batch["labels"] = torch.full((self.config.model.max_length,), -100)
            return batch
    
    def create_dataset(self, csv_path: str) -> DatasetDict:
        """Create train/validation/test datasets."""
        logger.info("Creating dataset from CSV...")
        
        # Load and validate data
        df = self.load_and_validate_csv(csv_path)
        
        # Create dataset
        dataset = Dataset.from_pandas(df)
        
        # Split dataset
        train_test_split = dataset.train_test_split(
            test_size=1 - self.config.data.train_split,
            seed=42
        )
        
        remaining_split = self.config.data.val_split / (self.config.data.val_split + self.config.data.test_split)
        val_test_split = train_test_split['test'].train_test_split(
            test_size=1 - remaining_split,
            seed=42
        )
        
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
        
        logger.info(f"Dataset splits - Train: {len(dataset_dict['train'])}, "
                   f"Val: {len(dataset_dict['validation'])}, Test: {len(dataset_dict['test'])}")
        
        # Process audio features
        logger.info("Extracting audio features...")
        dataset_dict = dataset_dict.map(
            self.extract_audio_features,
            desc="Extracting audio features",
            num_proc=1  # Use single process to avoid multiprocessing issues with audio
        )
        
        # Prepare model inputs
        logger.info("Preparing model features...")
        dataset_dict = dataset_dict.map(
            self.prepare_features,
            desc="Preparing features",
            remove_columns=[col for col in dataset_dict['train'].column_names 
                          if col not in ['input_features', 'labels']],
            num_proc=1
        )
        
        return dataset_dict

class DataCollator:
    """Custom data collator for Whisper training."""
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        # Extract input features and labels
        input_features = [torch.tensor(feature["input_features"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]
        
        # Pad sequences
        input_features = torch.stack(input_features)
        
        # Pad labels
        max_label_length = max(len(label) for label in labels)
        padded_labels = []
        for label in labels:
            padded_label = torch.full((max_label_length,), -100, dtype=label.dtype)
            padded_label[:len(label)] = label
            padded_labels.append(padded_label)
        
        labels = torch.stack(padded_labels)
        
        return {
            "input_features": input_features,
            "labels": labels
        }
