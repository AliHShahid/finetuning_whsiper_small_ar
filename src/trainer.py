"""Training utilities for Whisper fine-tuning."""

import torch
import logging
import inspect
from pathlib import Path
from typing import Dict, Optional
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from datasets import DatasetDict
import evaluate
import numpy as np
from .data_processor import DataCollator
from .config import Config

logger = logging.getLogger(__name__)

class WhisperTrainer:
    """Whisper model trainer."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(config.model.name)
        self.model = WhisperForConditionalGeneration.from_pretrained(config.model.name)
        self.model.to(self.device)
        
        # Initialize metrics
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        
        # Data collator
        self.data_collator = DataCollator(self.processor)
        
        logger.info(f"Initialized trainer with model: {config.model.name}")
        logger.info(f"Using device: {self.device}")
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels with pad token id
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)
        
        # Compute WER
        wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute CER
        cer = self.cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {
            "wer": wer,
            "cer": cer
        }
             
    def setup_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Setup training arguments."""
        return Seq2SeqTrainingArguments(
            output_dir=str(self.config.training.output_dir),
            per_device_train_batch_size=int(self.config.training.per_device_train_batch_size),
            per_device_eval_batch_size=int(self.config.training.per_device_eval_batch_size),
            gradient_accumulation_steps=int(self.config.training.gradient_accumulation_steps),
            learning_rate=float(self.config.training.learning_rate),
            weight_decay=float(self.config.training.weight_decay),
            warmup_steps=int(self.config.training.warmup_steps),
            max_steps=int(self.config.training.max_steps),
            eval_steps=int(self.config.training.eval_steps),
            logging_steps=int(self.config.training.logging_steps),
            save_steps=int(self.config.training.save_steps),
            save_total_limit=int(self.config.training.save_total_limit),
            fp16=bool(self.config.training.fp16),
            gradient_checkpointing=bool(self.config.training.gradient_checkpointing),
            dataloader_num_workers=int(self.config.training.dataloader_num_workers),
            remove_unused_columns=bool(self.config.training.remove_unused_columns),
            load_best_model_at_end=bool(self.config.training.load_best_model_at_end),
            metric_for_best_model=str(self.config.training.metric_for_best_model),
            greater_is_better=bool(self.config.training.greater_is_better),
            eval_strategy="steps",
            save_strategy="steps",
            predict_with_generate=True,
            generation_max_length=int(self.config.model.max_length),
            report_to=list(self.config.training.report_to if self.config.training.report_to is not None else []),
            # report_to=list(self.config.training.report_to),
            push_to_hub=bool(self.config.huggingface.push_to_hub),
            hub_model_id=str(self.config.huggingface.hub_model_id) if self.config.huggingface.hub_model_id else None,
            hub_private_repo=bool(self.config.huggingface.hub_private_repo),
        ) 

    
    # def setup_training_arguments(self) -> Seq2SeqTrainingArguments:
    #     """Setup training arguments."""
    #     return Seq2SeqTrainingArguments(
    #         output_dir=self.config.training.output_dir,
    #         per_device_train_batch_size=self.config.training.per_device_train_batch_size,
    #         per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
    #         gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
    #         learning_rate=self.config.training.learning_rate,
    #         weight_decay=self.config.training.weight_decay,
    #         warmup_steps=self.config.training.warmup_steps,
    #         max_steps=self.config.training.max_steps,
    #         eval_steps=self.config.training.eval_steps,
    #         logging_steps=self.config.training.logging_steps,
    #         save_steps=self.config.training.save_steps,
    #         save_total_limit=self.config.training.save_total_limit,
    #         fp16=self.config.training.fp16,
    #         gradient_checkpointing=self.config.training.gradient_checkpointing,
    #         dataloader_num_workers=self.config.training.dataloader_num_workers,
    #         remove_unused_columns=self.config.training.remove_unused_columns,
    #         load_best_model_at_end=self.config.training.load_best_model_at_end,
    #         metric_for_best_model=self.config.training.metric_for_best_model,
    #         greater_is_better=self.config.training.greater_is_better,
    #         eval_strategy="steps",
    #         # evaluation_strategy="steps",
    #         save_strategy="steps",
    #         predict_with_generate=True,
    #         generation_max_length=self.config.model.max_length,
    #         report_to=self.config.training.report_to,
    #         push_to_hub=True,  # Handle separately
    #     )
    
    def train(self, dataset: DatasetDict) -> Seq2SeqTrainer:
        """Train the model."""
        logger.info("Starting training...")
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Create trainer
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": dataset["train"],
            "eval_dataset": dataset["validation"],
            "data_collator": self.data_collator,
            "compute_metrics": self.compute_metrics,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
        }
        trainer_kwargs.update(self._get_trainer_processing_kwargs())

        trainer = Seq2SeqTrainer(**trainer_kwargs)
        
        # Train
        trainer.train()
        
        # Save model
        self.save_model(trainer)
        
        return trainer

    def _get_trainer_processing_kwargs(self) -> Dict[str, object]:
        """Return compatible tokenizer/processor kwargs for Seq2SeqTrainer."""
        try:
            sig = inspect.signature(Seq2SeqTrainer.__init__)
        except (TypeError, ValueError):
            return {}

        if "tokenizer" in sig.parameters:
            return {"tokenizer": self.processor.tokenizer}
        if "processing_class" in sig.parameters:
            return {"processing_class": self.processor}
        return {}
    
    def save_model(self, trainer: Seq2SeqTrainer):
        """Save the trained model and processor."""
        logger.info(f"Saving model to {self.config.training.output_dir}")
        
        # Create output directory
        Path(self.config.training.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model and processor
        trainer.save_model()
        self.processor.save_pretrained(self.config.training.output_dir)
        
        logger.info("Model saved successfully")
    
    def evaluate(self, dataset: DatasetDict, trainer: Seq2SeqTrainer) -> Dict:
        """Evaluate the model on test set."""
        logger.info("Evaluating model on test set...")
        
        eval_results = trainer.evaluate(eval_dataset=dataset["test"])
        
        logger.info(f"Test Results: {eval_results}")
        return eval_results
