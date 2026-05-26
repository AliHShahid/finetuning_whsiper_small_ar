# Whisper Fine-tuning (Arabic)

Fine-tune OpenAI Whisper for Arabic speech-to-text with:

- Hugging Face datasets (streaming) — default: `tarteel-ai/everyayah`
- Kaggle datasets via `kagglehub`
- Local CSV/TSV metadata + audio files

Includes LoRA fine-tuning (PEFT), WER/CER evaluation, and optional Hugging Face Hub upload.

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Train

```bash
python train.py --config config/config.yaml
```

By default, training uses the Hugging Face dataset configured in `config/config.yaml`.

## Configuration

The main configuration file is `config/config.yaml`.

Common settings:

- `model.name`: base checkpoint (default: `openai/whisper-small`)
- `model.lora.enabled`: LoRA on/off
- `data.source`: one of `huggingface`, `kaggle`, `local_csv`
- `data.audio_column`, `data.text_column`, `data.duration_column`: column names used by the dataset
- `data.streaming`: `true` uses iterable datasets (recommended for large datasets)
- `training.output_dir`: where the model is saved
- `training.fp16`: set to `false` if you are training on CPU

## Data sources

### Option A — Hugging Face dataset (default)

`config/config.yaml` ships with:

```yaml
data:
    source: "huggingface"
    huggingface_dataset: "tarteel-ai/everyayah"
    streaming: true
    audio_column: "audio"
    text_column: "text"
```

Run training:

```bash
python train.py --config config/config.yaml
```

### Option B — Kaggle dataset via `kagglehub`

This repo supports downloading Kaggle datasets using `kagglehub` and then resolving local audio paths.

1) Update `config/config.yaml`:

```yaml
data:
    source: "kaggle"
    kaggle_dataset: "bigguyubuntu/quran-ayat-speech-to-text"
    # Provide either:
    # - a metadata file in your repo (recommended), OR
    # - a file inside the Kaggle dataset via kagglehub
    metadata_path: "transcripts.tsv"

    # Adjust columns to match your metadata file
    audio_column: "PATH"
    text_column: "TRANSCRIPT"
    duration_column: "DURATION"
```

2) Train:

```bash
python train.py --config config/config.yaml
```

Notes:

- `transcripts.tsv` in this repo uses `${DATASET_PATH}` in the `PATH` column. The data loader will replace it with the downloaded dataset location when possible.
- If you use a Kaggle-provided metadata file instead, set `data.kaggle_file_path` in the config.

### Option C — Local CSV/TSV

If you already have audio files locally and a metadata file, set:

```yaml
data:
    source: "local_csv"
    csv_path: "/path/to/metadata.tsv"
    audio_column: "path"
    text_column: "text"
    duration_column: "duration" 
```

Run:

```bash
python train.py --config config/config.yaml
```

You can also override paths from the CLI:

```bash
python train.py --config config/config.yaml --csv-path /path/to/metadata.tsv --output-dir ./models/whisper-finetuned
```

## Outputs

After training you should see:

- `models/whisper-finetuned/` (or `training.output_dir`) — model + processor
- `models/whisper-finetuned/training_results.json` — config + evaluation results
- `logs/training.log` — training logs

## Hugging Face Hub (optional)

The trainer can push checkpoints to the Hugging Face Hub via Transformers `push_to_hub`.

1) Log in once:

```bash
huggingface-cli login
```

2) Enable in `config/config.yaml`:

```yaml
huggingface:
    push_to_hub: true
    hub_model_id: "your-username/your-model"
    hub_private_repo: true
```

3) Train as usual.

## Reported results (example)

On a Kaggle run (Arabic Quran audio), the following metrics were reported:

- WER: 23.39%
- CER: 6.68%

Links:

- Model: https://huggingface.co/alihassanshahid/whisper_everyayah
- Kaggle notebook: https://www.kaggle.com/code/alihassanshahid/finetuning-whisper-on-arabic-audio

## Project layout

```text
.
├─ config/
│  └─ config.yaml
├─ src/
│  ├─ config.py
│  ├─ data_processor.py
│  ├─ trainer.py
│  └─ utils.py
└─ train.py
```
