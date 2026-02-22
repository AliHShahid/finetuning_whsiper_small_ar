"""Data processing utilities for Whisper fine-tuning."""

import pandas as pd
import librosa
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
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
        self.streaming = bool(getattr(config.data, "streaming", False))
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

    def _count_metadata_rows(self, source_csv: str) -> int:
        """Count data rows (excluding header) without loading into memory."""
        path = Path(source_csv)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {source_csv}")

        with path.open("r", encoding="utf-8") as handle:
            total = sum(1 for _ in handle)

        return max(0, total - 1)

    def _resolve_metadata_path(self, source_csv: str) -> str:
        """Resolve metadata paths relative to the Kaggle dataset root."""
        if not source_csv or self.dataset_root is None:
            return source_csv

        path_obj = Path(str(source_csv))
        if path_obj.exists() or path_obj.is_absolute():
            return str(path_obj)

        candidate = self.dataset_root / path_obj
        if candidate.exists():
            return str(candidate)

        quran_public = self.dataset_root / "Quran_Ayat_public" / path_obj
        if quran_public.exists():
            return str(quran_public)

        dataset_double = self.dataset_root / "Dataset" / path_obj
        if dataset_double.exists():
            return str(dataset_double)

        return str(path_obj)

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
        normalized_path = str(file_path).replace("\\", "/")
        if normalized_path.startswith("./"):
            normalized_path = normalized_path[2:]
        if normalized_path.startswith("kaggle/input/"):
            normalized_path = "/" + normalized_path
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

    def _get_allowed_readers(self) -> set:
        """Build a normalized set of allowed readers."""
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
        return allowed_readers

    def _resolve_streaming_row(self, row: Dict) -> Dict:
        """Resolve audio paths for streaming datasets."""
        row[self.audio_column] = self._resolve_audio_path(row[self.audio_column])
        return row

    def _streaming_filter_row(self, row: Dict, allowed_readers: Optional[set] = None) -> bool:
        """Filter streaming rows by reader list and file existence."""
        if allowed_readers:
            reader = self._extract_reader_from_path(row.get(self.audio_column, ""))
            if reader.lower() not in allowed_readers:
                return False
        return Path(row.get(self.audio_column, "")).exists()

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
                    source_csv = self._resolve_metadata_path(source_csv)
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
            allowed_readers = self._get_allowed_readers()
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

            df_valid = df.loc[valid_files].reset_index(drop=True)
            logger.info(f"Found {len(df_valid)} valid audio files out of {len(df)}")
            
            return df_valid
            
        except Exception as e:
            logger.error(f"Error loading dataset metadata: {e}")
            raise
    
    def prepare_features(self, batch: Dict) -> Dict:
        """Load audio and prepare model-ready features in a single pass."""
        try:
            file_path = batch[self.audio_column]

            # Load audio
            waveform, sr = librosa.load(file_path, sr=self.sampling_rate)

            # Validate duration
            duration = len(waveform) / sr
            if duration < self.min_duration or duration > self.max_duration:
                logger.warning(
                    f"Audio duration {duration:.2f}s outside valid range for {file_path}"
                )
                target_length = int(self.sampling_rate * min(duration, self.max_duration))
                if len(waveform) > target_length:
                    waveform = waveform[:target_length]
                elif len(waveform) < int(self.sampling_rate * self.min_duration):
                    pad_length = int(self.sampling_rate * self.min_duration) - len(waveform)
                    waveform = np.pad(waveform, (0, pad_length), mode="constant")

            # Extract features directly without storing raw audio
            input_features = self.processor.feature_extractor(
                waveform,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            ).input_features[0]

            transcription = str(batch.get(self.text_column, ""))
            labels = self.processor.tokenizer(
                transcription,
                return_tensors="pt",
                padding="max_length",
                max_length=self.config.model.max_length,
                truncation=True,
            ).input_ids[0]

            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            batch["input_features"] = input_features
            batch["labels"] = labels
            return batch

        except Exception as e:
            logger.error(
                f"Error preparing features for {batch.get(self.audio_column, 'unknown')}: {e}"
            )
            # Return dummy features for failed samples
            batch["input_features"] = torch.zeros((80, 3000))
            batch["labels"] = torch.full((self.config.model.max_length,), -100)
            return batch
    
    def create_dataset(self, csv_path: Optional[str] = None) -> DatasetDict:
        """Create train/validation/test datasets."""
        logger.info(f"Creating dataset from source: {self.data_source}")

        if self.streaming:
            logger.info("Streaming mode enabled; using iterable datasets.")
            source_csv = csv_path or self.config.data.metadata_path or self.config.data.csv_path

            if self.data_source == "kaggle":
                self._ensure_kaggle_dataset_root()
                self._build_kaggle_path_index()
                if not source_csv:
                    source_csv = self.config.data.kaggle_file_path
                if not source_csv:
                    raise ValueError(
                        "Streaming requires data.metadata_path or data.kaggle_file_path for Kaggle sources"
                    )
                source_csv = self._resolve_metadata_path(source_csv)

            if not source_csv:
                raise ValueError("Streaming requires data.metadata_path or data.csv_path")

            delimiter = "\t" if str(source_csv).lower().endswith(".tsv") else ","
            total_rows = self._count_metadata_rows(source_csv)
            if total_rows <= 0:
                raise ValueError("No data rows found in metadata file for streaming")

            raw_dataset = load_dataset(
                "csv",
                data_files=source_csv,
                delimiter=delimiter,
                split="train",
                streaming=True,
            )

            raw_dataset = raw_dataset.map(self._resolve_streaming_row)
            allowed_readers = self._get_allowed_readers()
            raw_dataset = raw_dataset.filter(
                self._streaming_filter_row,
                fn_kwargs={"allowed_readers": allowed_readers},
            )

            buffer_size = min(10000, total_rows)
            if buffer_size > 1:
                raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=buffer_size)

            if total_rows < 3:
                train_count = max(1, total_rows)
                val_count = 0
                test_count = max(0, total_rows - train_count)
            else:
                train_count = max(1, int(total_rows * self.config.data.train_split))
                val_count = max(1, int(total_rows * self.config.data.val_split))
                test_count = max(1, total_rows - train_count - val_count)
                overflow = train_count + val_count + test_count - total_rows
                if overflow > 0:
                    train_count = max(1, train_count - overflow)

            train_split = raw_dataset.take(train_count)
            remainder = raw_dataset.skip(train_count)
            val_split = remainder.take(val_count) if val_count else remainder.take(0)
            test_split = remainder.skip(val_count)
            if test_count:
                test_split = test_split.take(test_count)

            dataset_dict = IterableDatasetDict(
                {
                    "train": train_split,
                    "validation": val_split,
                    "test": test_split,
                }
            )

            logger.info(
                f"Streaming dataset splits - Train: {train_count}, Val: {val_count}, Test: {test_count}"
            )

            logger.info("Preparing model features...")
            base_columns = []
            if dataset_dict["train"].column_names is not None:
                base_columns = list(dataset_dict["train"].column_names)
            elif raw_dataset.features is not None:
                base_columns = list(raw_dataset.features.keys())

            remove_columns = [
                col
                for col in base_columns
                if col not in ["input_features", "labels"]
            ]
            dataset_dict = IterableDatasetDict(
                {
                    "train": dataset_dict["train"].map(
                        self.prepare_features,
                        remove_columns=remove_columns,
                    ),
                    "validation": dataset_dict["validation"].map(
                        self.prepare_features,
                        remove_columns=remove_columns,
                    ),
                    "test": dataset_dict["test"].map(
                        self.prepare_features,
                        remove_columns=remove_columns,
                    ),
                }
            )

            return dataset_dict
        
        # Load and validate data
        df = self.load_and_validate_data(csv_path)

        # Deterministic speaker-based split: hold out Abdurrahmaan_As-Sudais_64kbps for val/test.
        holdout_reader = "Abdurrahmaan_As-Sudais_64kbps"
        df = df.copy()
        df["_reader"] = df[self.audio_column].astype(str).apply(self._extract_reader_from_path)
        holdout_mask = df["_reader"].str.lower() == holdout_reader.lower()

        holdout_df = df[holdout_mask].reset_index(drop=True)
        train_df = df[~holdout_mask].reset_index(drop=True)

        if holdout_df.empty:
            logger.warning(
                "No holdout samples found for Abdurrahmaan_As-Sudais_64kbps; falling back to random split."
            )
            dataset = Dataset.from_pandas(df.drop(columns=["_reader"]))
            train_test_split = dataset.train_test_split(
                test_size=1 - self.config.data.train_split,
                seed=42
            )

            remaining_split = self.config.data.val_split / (
                self.config.data.val_split + self.config.data.test_split
            )
            val_test_split = train_test_split["test"].train_test_split(
                test_size=1 - remaining_split,
                seed=42
            )

            dataset_dict = DatasetDict({
                "train": train_test_split["train"],
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            })
        else:
            split_index = len(holdout_df) // 2
            val_df = holdout_df.iloc[:split_index].drop(columns=["_reader"]).reset_index(drop=True)
            test_df = holdout_df.iloc[split_index:].drop(columns=["_reader"]).reset_index(drop=True)
            train_df = train_df.drop(columns=["_reader"]).reset_index(drop=True)

            dataset_dict = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(val_df),
                "test": Dataset.from_pandas(test_df),
            })
        
        logger.info(f"Dataset splits - Train: {len(dataset_dict['train'])}, "
                   f"Val: {len(dataset_dict['validation'])}, Test: {len(dataset_dict['test'])}")
        
        # Prepare model inputs in a single pass to reduce memory usage
        logger.info("Preparing model features...")
        dataset_dict = dataset_dict.map(
            self.prepare_features,
            desc="Preparing features",
            remove_columns=[
                col
                for col in dataset_dict["train"].column_names
                if col not in ["input_features", "labels"]
            ],
            num_proc=1,
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
