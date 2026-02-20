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

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:
    kagglehub = None
    KaggleDatasetAdapter = None

logger = logging.getLogger(__name__)

class WhisperDataProcessor:
    """Data processor for Whisper fine-tuning."""
    
    def __init__(self, processor: WhisperProcessor, config):
        self.processor = processor
        self.config = config
        self.sampling_rate = config.data.sampling_rate
        self.max_duration = config.data.max_duration
        self.min_duration = config.data.min_duration
        self.data_source = getattr(config.data, "source", "local_csv")
        self.audio_column = getattr(config.data, "audio_column", "FilePath")
        self.text_column = getattr(config.data, "text_column", "Transcript")
        self.duration_column = getattr(config.data, "duration_column", "")
        self.readerlist_path = getattr(config.data, "readerlist_path", "")
        self.allowed_readers = getattr(config.data, "allowed_readers", None)
        self.dataset_root: Optional[Path] = None
        self.kaggle_dataset_slug = ""
        self._kaggle_path_index: Optional[Dict[str, str]] = None
        
    def _load_local_csv(self, csv_path: str) -> pd.DataFrame:
        """Load dataset from a local CSV file."""
        if str(csv_path).lower().endswith(".tsv"):
            df = pd.read_csv(csv_path, sep="\t")
        else:
            df = pd.read_csv(csv_path)
        logger.info(f"Loaded local CSV with {len(df)} rows")
        return df

    def _ensure_kaggle_dataset_root(self) -> None:
        """Download Kaggle dataset and set dataset_root for path resolution."""
        if kagglehub is None:
            raise ImportError(
                "kagglehub is not installed. Install it with: pip install 'kagglehub[pandas-datasets]'"
            )

        if self.dataset_root is not None:
            return

        kaggle_dataset = getattr(self.config.data, "kaggle_dataset", "")
        if not kaggle_dataset:
            raise ValueError("data.kaggle_dataset must be set when data.source is 'kaggle'")

        dataset_download_path = kagglehub.dataset_download(kaggle_dataset)
        self.dataset_root = Path(dataset_download_path)
        self.kaggle_dataset_slug = kaggle_dataset.split("/")[-1]
        logger.info(f"Kaggle dataset downloaded to: {self.dataset_root}")

    def _load_kaggle_dataset(self) -> pd.DataFrame:
        """Load dataset metadata from Kaggle using kagglehub API."""
        if kagglehub is None or KaggleDatasetAdapter is None:
            raise ImportError(
                "kagglehub is not installed. Install it with: pip install 'kagglehub[pandas-datasets]'"
            )

        kaggle_dataset = getattr(self.config.data, "kaggle_dataset", "")
        kaggle_file_path = getattr(self.config.data, "kaggle_file_path", "")

        if not kaggle_dataset:
            raise ValueError("data.kaggle_dataset must be set when data.source is 'kaggle'")
        if not kaggle_file_path:
            raise ValueError("data.kaggle_file_path must be set when loading Kaggle metadata")

        logger.info(
            f"Loading dataset from Kaggle: {kaggle_dataset} (file: {kaggle_file_path})"
        )
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            kaggle_dataset,
            kaggle_file_path,
        )

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Kaggle dataset loader did not return a pandas DataFrame")

        dataset_download_path = kagglehub.dataset_download(kaggle_dataset)
        self.dataset_root = Path(dataset_download_path)
        self.kaggle_dataset_slug = kaggle_dataset.split("/")[-1]
        logger.info(f"Kaggle dataset downloaded to: {self.dataset_root}")
        logger.info(f"Loaded Kaggle dataframe with {len(df)} rows")

        return df

    def _replace_dataset_token(self, file_path: str) -> str:
        """Replace ${DATASET_PATH} with a concrete dataset root if available."""
        if "${DATASET_PATH}" not in str(file_path):
            return str(file_path)

        if self.dataset_root is not None:
            return str(file_path).replace("${DATASET_PATH}", str(self.dataset_root))

        kaggle_input_root = Path("/kaggle/input")
        if self.kaggle_dataset_slug and kaggle_input_root.exists():
            return str(file_path).replace(
                "${DATASET_PATH}", str(kaggle_input_root / self.kaggle_dataset_slug)
            )

        return str(file_path).replace("${DATASET_PATH}", ".")

    def _resolve_audio_path(self, file_path: str) -> str:
        """Resolve relative audio paths using Kaggle dataset roots if available."""
        file_path = self._replace_dataset_token(file_path)
        normalized_path = str(file_path).replace("\\", "/").lstrip("./")
        path_obj = Path(normalized_path)
        if path_obj.exists() or path_obj.is_absolute() or self.dataset_root is None:
            return str(path_obj)

        candidate = self.dataset_root / path_obj
        if candidate.exists():
            return str(candidate)

        dataset_double = self.dataset_root / "Dataset" / path_obj
        if dataset_double.exists():
            return str(dataset_double)

        kaggle_input_root = Path("/kaggle/input")
        if self.kaggle_dataset_slug and kaggle_input_root.exists():
            kaggle_candidate = kaggle_input_root / self.kaggle_dataset_slug / path_obj
            if kaggle_candidate.exists():
                return str(kaggle_candidate)

            kaggle_double = kaggle_input_root / self.kaggle_dataset_slug / "Dataset" / path_obj
            if kaggle_double.exists():
                return str(kaggle_double)

        if path_obj.parts:
            first_part = path_obj.parts[0]
            for root in [self.dataset_root, kaggle_input_root / self.kaggle_dataset_slug]:
                if root.exists():
                    match = next(
                        (p for p in root.iterdir() if p.name.lower() == first_part.lower()),
                        None,
                    )
                    if match is not None:
                        case_candidate = match.joinpath(*path_obj.parts[1:])
                        if case_candidate.exists():
                            return str(case_candidate)

        if self._kaggle_path_index:
            lookup = normalized_path.lower().lstrip("/")
            candidates = [lookup]
            if lookup.startswith("dataset/"):
                candidates.append(lookup[len("dataset/"):])
            if lookup.startswith("dataset/dataset/"):
                candidates.append(lookup[len("dataset/dataset/"):])
            if not lookup.startswith("dataset/"):
                candidates.append(f"dataset/{lookup}")
            if not lookup.startswith("dataset/dataset/"):
                candidates.append(f"dataset/dataset/{lookup}")

            for candidate in candidates:
                if candidate in self._kaggle_path_index:
                    return self._kaggle_path_index[candidate]

        return str(path_obj)

    def _build_kaggle_path_index(self) -> None:
        """Index Kaggle dataset files for robust case-insensitive path resolution."""
        if self._kaggle_path_index is not None:
            return

        index: Dict[str, str] = {}
        roots = []
        if self.dataset_root and self.dataset_root.exists():
            roots.append(self.dataset_root)

        kaggle_input_root = Path("/kaggle/input")
        if self.kaggle_dataset_slug and kaggle_input_root.exists():
            slug_root = kaggle_input_root / self.kaggle_dataset_slug
            if slug_root.exists():
                roots.append(slug_root)

        for root in roots:
            for path in root.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(root).as_posix().lower()
                    index.setdefault(rel, str(path))

        self._kaggle_path_index = index

    def _load_readerlist(self) -> List[str]:
        """Load reader list from a TSV-like file (one reader per line)."""
        if not self.readerlist_path:
            return []

        path = Path(self.readerlist_path)
        if not path.exists():
            logger.warning(f"Reader list not found: {self.readerlist_path}")
            return []

        readers: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.lower().startswith("reader"):
                    continue
                if "[" in line:
                    line = line.split("[", 1)[0].strip()
                token = line.split()[0].strip()
                if token:
                    readers.append(token)
        return readers

    def _extract_reader_from_path(self, file_path: str) -> str:
        """Extract reader folder name from an audio path."""
        parts = Path(str(file_path).replace("\\", "/")).parts
        lowered = [part.lower() for part in parts]
        if "audio_data" in lowered:
            idx = lowered.index("audio_data")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return ""

    def load_and_validate_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load dataset metadata from configured source and validate it."""
        try:
            if self.data_source == "kaggle":
                self._ensure_kaggle_dataset_root()
                self._build_kaggle_path_index()

                source_csv = csv_path or self.config.data.metadata_path or self.config.data.csv_path
                if source_csv:
                    df = self._load_local_csv(source_csv)
                else:
                    df = self._load_kaggle_dataset()
            else:
                source_csv = csv_path or self.config.data.metadata_path or self.config.data.csv_path
                df = self._load_local_csv(source_csv)
            
            # Validate required columns
            required_cols = [self.audio_column, self.text_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Filter by reader list when provided
            allowed_readers = set()
            readers_from_file = self._load_readerlist()
            if readers_from_file:
                allowed_readers.update(reader.lower() for reader in readers_from_file)
            if self.allowed_readers:
                allowed_readers = set(reader.lower() for reader in self.allowed_readers)
                if readers_from_file:
                    allowed_readers = allowed_readers.intersection(
                        reader.lower() for reader in readers_from_file
                    )

            if allowed_readers:
                df = df.copy()
                df["_reader"] = df[self.audio_column].astype(str).apply(
                    self._extract_reader_from_path
                )
                before = len(df)
                df = df[df["_reader"].str.lower().isin(allowed_readers)].drop(columns=["_reader"])
                logger.info(f"Filtered by readers: {before} -> {len(df)} rows")

            # Resolve possible relative file paths (useful for Kaggle dataset files)
            df = df.copy()
            df[self.audio_column] = df[self.audio_column].astype(str).apply(self._resolve_audio_path)
            
            # Validate file paths
            valid_files = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating audio files"):
                file_path = row[self.audio_column]
                if Path(file_path).exists():
                    valid_files.append(idx)
                else:
                    logger.warning(f"File not found: {file_path}")
            
            df_valid = df.iloc[valid_files].reset_index(drop=True)
            logger.info(f"Found {len(df_valid)} valid audio files out of {len(df)}")
            
            return df_valid
            
        except Exception as e:
            logger.error(f"Error loading dataset metadata: {e}")
            raise
    
    def extract_audio_features(self, batch: Dict) -> Dict:
        """Extract audio features from file paths."""
        try:
            file_path = batch[self.audio_column]
            
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
            logger.error(f"Error processing audio file {batch.get(self.audio_column, 'unknown')}: {e}")
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
            
            transcription = str(batch.get(self.text_column, ""))
            
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
    
    def create_dataset(self, csv_path: Optional[str] = None) -> DatasetDict:
        """Create train/validation/test datasets."""
        logger.info(f"Creating dataset from source: {self.data_source}")
        
        # Load and validate data
        df = self.load_and_validate_data(csv_path)
        
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
