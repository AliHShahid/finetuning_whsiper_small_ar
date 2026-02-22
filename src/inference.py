"""Inference utilities for trained Whisper model."""

import torch
import librosa
import logging
from pathlib import Path
from typing import List, Optional, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)

class WhisperInference:
    """Inference class for trained Whisper model."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = WhisperProcessor.from_pretrained(
            model_path,
            language="ar",
            task="transcribe",
        )
        self.model.to(self.device)
        self.model.eval()

        self.model.generation_config.language = "ar"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        
        logger.info(f"Loaded model from {model_path} on {self.device}")
    
    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: str = "ar",
        return_timestamps: bool = False,
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = None,
    ) -> str:
        """Transcribe a single audio file."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Prepare input
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            generate_kwargs = {
                "language": "ar",
                "task": "transcribe",
                "return_timestamps": return_timestamps,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0,
            }
            if max_new_tokens is not None:
                generate_kwargs["max_new_tokens"] = int(max_new_tokens)
            elif max_length is not None:
                generate_kwargs["max_length"] = int(max_length)

            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    **generate_kwargs,
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return ""
    
    def transcribe_batch(
        self, 
        audio_paths: List[Union[str, Path]], 
        language: str = "en"
    ) -> List[str]:
        """Transcribe multiple audio files."""
        transcriptions = []
        
        for audio_path in audio_paths:
            transcription = self.transcribe_audio(audio_path, language)
            transcriptions.append(transcription)
        
        return transcriptions
    
    def transcribe_directory(
        self, 
        directory_path: Union[str, Path], 
        extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a'],
        language: str = "en"
    ) -> dict:
        """Transcribe all audio files in a directory."""
        directory = Path(directory_path)
        audio_files = []
        
        for ext in extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"**/*{ext}"))
        
        results = {}
        for audio_file in audio_files:
            transcription = self.transcribe_audio(audio_file, language)
            results[str(audio_file)] = transcription
        
        return results
