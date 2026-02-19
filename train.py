#!/usr/bin/env python3
"""Main training script for Whisper fine-tuning."""
import os

import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.data_processor import WhisperDataProcessor
from src.trainer import WhisperTrainer
from src.hyperparameter_tuning import HyperparameterTuner
from src.utils import setup_logging, create_directory_structure, save_results

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--csv-path", type=str, help="Path to CSV file (overrides config)")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--tune-hyperparameters", action="store_true", 
                       help="Enable hyperparameter tuning")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup
    create_directory_structure()
    setup_logging(args.log_level, "logs/training.log")
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.csv_path:
        config.data.csv_path = args.csv_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    logger.info("Starting Whisper fine-tuning")
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize processor and data processor
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(config.model.name)
        data_processor = WhisperDataProcessor(processor, config)
        
        # Create dataset
        logger.info("Creating dataset...")
        dataset = data_processor.create_dataset(
            config.data.csv_path if config.data.source == "local_csv" else None
        )
        
        # Hyperparameter tuning
        if args.tune_hyperparameters:
            logger.info("Starting hyperparameter tuning...")
            tuner = HyperparameterTuner(config, dataset)
            best_params = tuner.tune()
            
            # Update config with best parameters
            for param, value in best_params.items():
                setattr(config.training, param, value)
            
            logger.info(f"Using best parameters: {best_params}")
        
        # Train model
        trainer_instance = WhisperTrainer(config)
        trainer = trainer_instance.train(dataset)
        
        # Evaluate model
        eval_results = trainer_instance.evaluate(dataset, trainer)
        
        # Save results
        results = {
            "config": config.__dict__,
            "evaluation_results": eval_results,
            "model_path": config.training.output_dir
        }
        
        save_results(results, f"{config.training.output_dir}/training_results.json")
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config.training.output_dir}")
        logger.info(f"Final evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
