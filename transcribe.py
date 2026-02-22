import torch
import librosa
import os
import re
import evaluate
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# =====================================
# CONFIG
# =====================================
# If your finetuned model is local, use the local path instead of model_id
model_id = "alihassanshahid/arabic-whisper" # Or local path like "./models/whisper-finetuned"
base_model = "openai/whisper-small"
audio_path = "001001 (1).mp3"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================================
# HELPERS
# =====================================
def normalize_arabic(text):
    if not text: return ""
    # Remove diacritics
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    # Normalize letters
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("ـ", "") # Keshida
    return re.sub(r"\s+", " ", text).strip()

# =====================================
# LOAD PROCESSOR
# =====================================
print("Loading processor...")
try:
    processor = WhisperProcessor.from_pretrained(base_model, language="ar", task="transcribe")
except Exception as e:
    print(f"Error loading processor: {e}")
    # Fallback manual load
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    tokenizer = WhisperTokenizer.from_pretrained(base_model, language="ar", task="transcribe")
    processor = WhisperProcessor(feature_extractor, tokenizer)

# =====================================
# LOAD MODEL
# =====================================
print(f"Loading model: {model_id}...")
try:
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
except Exception as e:
    print(f"Error loading model from {model_id}. Ensure you are logged into HuggingFace or the model is public.")
    print(f"Fallback to base model for testing script functionality...")
    model = WhisperForConditionalGeneration.from_pretrained(base_model).to(device)

# =====================================
# LOAD AUDIO
# =====================================
if not os.path.exists(audio_path):
    print(f"ERROR: Audio file {audio_path} not found!")
    # Create dummy audio for script verification if needed
    import numpy as np
    audio = np.random.uniform(-1, 1, 16000)
    sr = 16000
else:
    print(f"Loading audio: {audio_path}...")
    audio, sr = librosa.load(audio_path, sr=16000)

# =====================================
# PREPROCESS
# =====================================
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device)

# =====================================
# GENERATE (ROBUST SETTINGS)
# =====================================
print("Transcribing with robust decoding (Anti-Loop)...")

# Using Beam Search and Repetition Penalty to kill loops
generation_kwargs = {
    "num_beams": 5,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "max_new_tokens": 256,
    "condition_on_previous_text": False, # Important to stop segment-level loops
}

with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        **generation_kwargs
    )

# =====================================
# DECODE & METRICS
# =====================================
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
normalized_pred = normalize_arabic(transcription)

print("\n" + "="*40)
print("RAW TRANSCRIPTION:")
print(f"'{transcription}'")
print("\nNORMALIZED:")
print(f"'{normalized_pred}'")
print("="*40 + "\n")

# If you have a reference text, you can calculate WER/CER here
# reference = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
# wer = evaluate.load("wer")
# score = wer.compute(predictions=[normalized_pred], references=[normalize_arabic(reference)])
# print(f"WER: {score:.2%}")
