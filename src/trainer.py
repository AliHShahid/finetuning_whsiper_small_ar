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
from datasets import Dataset, DatasetDict, IterableDataset
import evaluate
import numpy as np
from .data_processor import DataCollator
from .config import Config
from .utils import normalize_arabic_text

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)

class WhisperTrainer:
    """Whisper model trainer."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(
            config.model.name,
            language="ar",
            task="transcribe",
        )
        self.processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
        logger.info("Forced Arabic language and transcription task tokens in tokenizer")
        self.model = WhisperForConditionalGeneration.from_pretrained(config.model.name)
        self.model.to(self.device)

        # Apply LoRA if enabled
        if config.model.lora.enabled:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT is enabled in config but peft library is not installed.")
            
            logger.info("Applying LoRA to the model...")
            lora_config = LoraConfig(
                r=config.model.lora.r,
                lora_alpha=config.model.lora.alpha,
                target_modules=config.model.lora.target_modules,
                lora_dropout=config.model.lora.dropout,
                bias=config.model.lora.bias,
                task_type="SEQ_2_SEQ_LM"
            )
            
            # If using gradient checkpointing with PEFT, we need this
            if config.training.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            # Freeze the encoder ONLY if NOT using LoRA
            # (LoRA freezes the whole model and then unfreezes only the adapter layers)
            self.model.model.encoder.requires_grad_(False)
            logger.info("Encoder has been frozen (requires_grad=False)")

        # Enable SpecAugment for better robustness
        if hasattr(self.processor.feature_extractor, "apply_spec_augment"):
            self.processor.feature_extractor.apply_spec_augment = True
            logger.info("SpecAugment enabled in feature extractor")

        self.model.generation_config.language = "ar"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        
        # Relaxed decoding parameters for Quranic text
        self.model.generation_config.num_beams = 5
        self.model.generation_config.repetition_penalty = 1.1 # Reduced from 1.2
        self.model.generation_config.no_repeat_ngram_size = 0 # Disabled to allow repetitive verses
        self.model.generation_config.length_penalty = 0.8
        self.model.generation_config.max_length = int(config.model.max_length)
        self.model.generation_config.early_stopping = True
        
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
        decoded_preds_raw = self.processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels with pad token id
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels_raw = self.processor.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [normalize_arabic_text(text) for text in decoded_preds_raw]
        decoded_labels = [normalize_arabic_text(text) for text in decoded_labels_raw]
        
        # Log a few samples for visibility (raw vs normalized)
        for i in range(min(3, len(decoded_preds))):
            logger.info(f"Sample {i}:")
            logger.info(f"  Ref (Raw):  {decoded_labels_raw[i]}")
            logger.info(f"  Ref (Norm): {decoded_labels[i]}")
            logger.info(f"  Pred (Raw):  {decoded_preds_raw[i]}")
            logger.info(f"  Pred (Norm): {decoded_preds[i]}")
        
        # Compute WER
        wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute CER
        cer = self.cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        metrics = {
            "wer": wer,
            "cer": cer
        }
        logger.info(f"Computed metrics: {metrics}")
        return metrics
             
    def setup_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Setup training arguments."""
        # Force evaluation strategy to 'steps' if we have verification data
        eval_strategy = "steps"
        load_best_model_at_end = bool(self.config.training.load_best_model_at_end)
        
        return Seq2SeqTrainingArguments(
            output_dir=str(self.config.training.output_dir),
            per_device_train_batch_size=int(self.config.training.per_device_train_batch_size),
            per_device_eval_batch_size=int(self.config.training.per_device_eval_batch_size),
            gradient_accumulation_steps=int(self.config.training.gradient_accumulation_steps),
            learning_rate=float(self.config.training.learning_rate),
            weight_decay=float(self.config.training.weight_decay),
            warmup_steps=int(self.config.training.warmup_steps),
            num_train_epochs=float(self.config.training.num_train_epochs) if self.config.training.num_train_epochs is not None else 3.0,
            max_steps=int(self.config.training.max_steps) if self.config.training.max_steps is not None else -1,
            seed=int(self.config.training.seed),
            lr_scheduler_type=str(self.config.training.lr_scheduler_type),
            adam_beta1=float(self.config.training.adam_beta1),
            adam_beta2=float(self.config.training.adam_beta2),
            adam_epsilon=float(self.config.training.adam_epsilon),
            eval_steps=int(self.config.training.eval_steps),
            logging_steps=int(self.config.training.logging_steps),
            save_steps=int(self.config.training.save_steps),
            save_total_limit=int(self.config.training.save_total_limit),
            fp16=bool(self.config.training.fp16),
            gradient_checkpointing=bool(self.config.training.gradient_checkpointing),
            dataloader_num_workers=int(self.config.training.dataloader_num_workers),
            remove_unused_columns=bool(self.config.training.remove_unused_columns),
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=str(self.config.training.metric_for_best_model),
            greater_is_better=bool(self.config.training.greater_is_better),
            eval_strategy=eval_strategy,
            save_strategy="steps",
            predict_with_generate=True,
            generation_max_length=int(self.config.model.max_length),
            report_to=list(self.config.training.report_to if self.config.training.report_to is not None else []),
            push_to_hub=bool(self.config.huggingface.push_to_hub),
            hub_model_id=str(self.config.huggingface.hub_model_id) if self.config.huggingface.hub_model_id else None,
            hub_private_repo=bool(self.config.huggingface.hub_private_repo),
            max_grad_norm=float(self.config.training.max_grad_norm),
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
        eval_dataset = dataset["validation"]
        if self.config.data.streaming and eval_dataset is not None:
            logger.info("Materializing validation set for training evaluation...")
            eval_dataset = self._materialize_iterable_eval(eval_dataset)

        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": dataset["train"],
            "eval_dataset": eval_dataset,
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
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and processor
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(self.config.training.output_dir)
        else:
            trainer.save_model()
            
        self.processor.save_pretrained(self.config.training.output_dir)
        
        logger.info("Model saved successfully")
    
    def evaluate(self, dataset: DatasetDict, trainer: Seq2SeqTrainer) -> Dict:
        """Evaluate the model on test set."""
        logger.info("Evaluating model on test set...")
        eval_dataset = dataset["test"]
        if isinstance(eval_dataset, IterableDataset):
            eval_dataset = self._materialize_iterable_eval(eval_dataset)

        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        
        # Ensure WER and CER are present
        if "eval_wer" in eval_results:
            eval_results["wer"] = eval_results["eval_wer"]
        if "eval_cer" in eval_results:
            eval_results["cer"] = eval_results["eval_cer"]

        logger.info(f"Test Results: {eval_results}")
        if "wer" in eval_results:
            logger.info(f"Final WER: {eval_results['wer']:.2%}")
        if "cer" in eval_results:
            logger.info(f"Final CER: {eval_results['cer']:.2%}")

        return eval_results

    def _materialize_iterable_eval(self, eval_dataset: IterableDataset) -> Dataset:
        """Materialize a small eval dataset from streaming data, skipping failed samples."""
        max_samples = self.config.data.eval_max_samples
        if max_samples is None:
            logger.info("Materializing full evaluation dataset...")
            it = iter(eval_dataset)
        else:
            max_samples = max(1, int(max_samples))
            logger.info(f"Materializing up to {max_samples} evaluation samples...")
            it = iter(eval_dataset.take(max_samples * 2)) # Take extra to account for failures

        samples = []
        failures = 0
        try:
            for sample in it:
                # Check for dummy features indicating load failure
                if torch.all(sample["input_features"] == 0):
                    failures += 1
                    continue
                
                samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break
        except Exception as e:
            logger.error(f"Error during eval materialization: {e}")

        if failures:
            logger.warning(f"Skipped {failures} failed samples during evaluation materialization")
        
        if not samples:
            logger.warning("Materialized evaluation dataset is empty! Evaluation will fail to produce metrics.")
            
        return Dataset.from_list(samples)
