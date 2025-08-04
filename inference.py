#!/usr/bin/env python3
"""Inference script for trained Whisper model."""

import argparse
import logging
from pathlib import Path
import json

from src.inference import WhisperInference
from src.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Whisper model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--audio-path", type=str,
                       help="Path to single audio file")
    parser.add_argument("--audio-dir", type=str,
                       help="Path to directory containing audio files")
    parser.add_argument("--output-file", type=str, default="results/transcriptions.json",
                       help="Output file for transcriptions")
    parser.add_argument("--language", type=str, default="en",
                       help="Language code for transcription")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"],
                       default="auto", help="Device to use for inference")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize inference
    device = None if args.device == "auto" else args.device
    inference = WhisperInference(args.model_path, device)
    
    results = {}
    
    try:
        if args.audio_path:
            # Single file transcription
            logger.info(f"Transcribing single file: {args.audio_path}")
            transcription = inference.transcribe_audio(args.audio_path, args.language)
            results[args.audio_path] = transcription
            print(f"Transcription: {transcription}")
            
        elif args.audio_dir:
            # Directory transcription
            logger.info(f"Transcribing directory: {args.audio_dir}")
            results = inference.transcribe_directory(args.audio_dir, language=args.language)
            
            for file_path, transcription in results.items():
                print(f"{file_path}: {transcription}")
        
        else:
            logger.error("Please provide either --audio-path or --audio-dir")
            return
        
        # Save results
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
