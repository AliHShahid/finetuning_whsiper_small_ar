import datasets
from datasets import load_dataset, Audio

dataset = load_dataset("tarteel-ai/everyayah", streaming=True, trust_remote_code=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
train = dataset["train"]

for i, x in enumerate(train):
    audio = x["audio"]
    print("Type:", type(audio))
    print("Dir:", dir(audio))
    if isinstance(audio, dict):
        print("Keys:", audio.keys())
    break
