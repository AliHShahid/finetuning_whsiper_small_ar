"""Hyperparameter tuning utilities."""

import optuna
import logging
from typing import Dict, Any
from .trainer import WhisperTrainer
from .config import Config
from datasets import DatasetDict

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Hyperparameter tuning with Optuna."""
    
    def __init__(self, config: Config, dataset: DatasetDict):
        self.config = config
        self.dataset = dataset
        self.best_params = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float(
            'learning_rate', 
            self.config.hyperparameter_tuning.search_space.learning_rate.low,
            self.config.hyperparameter_tuning.search_space.learning_rate.high,
            log=True
        )
        
        batch_size = trial.suggest_categorical(
            'per_device_train_batch_size',
            self.config.hyperparameter_tuning.search_space.per_device_train_batch_size.choices
        )
        
        warmup_steps = trial.suggest_int(
            'warmup_steps',
            self.config.hyperparameter_tuning.search_space.warmup_steps.low,
            self.config.hyperparameter_tuning.search_space.warmup_steps.high
        )
        
        # Update config with suggested parameters
        trial_config = self.config
        trial_config.training.learning_rate = learning_rate
        trial_config.training.per_device_train_batch_size = batch_size
        trial_config.training.warmup_steps = warmup_steps
        trial_config.training.output_dir = f"{self.config.training.output_dir}_trial_{trial.number}"
        
        # Train model with suggested parameters
        trainer_instance = WhisperTrainer(trial_config)
        trainer = trainer_instance.train(self.dataset)
        
        # Evaluate and return metric
        eval_results = trainer.evaluate(eval_dataset=self.dataset["validation"])
        
        return eval_results[f"eval_{self.config.training.metric_for_best_model}"]
    
    def tune(self) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        logger.info("Starting hyperparameter tuning...")
        
        study = optuna.create_study(
            direction="minimize" if not self.config.training.greater_is_better else "maximize"
        )
        
        study.optimize(
            self.objective, 
            n_trials=self.config.hyperparameter_tuning.n_trials
        )
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        
        return self.best_params
