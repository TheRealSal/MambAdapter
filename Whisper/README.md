### Whisper Experiments

This directory contains training recipes and hyperparameter YAMLs to fine-tune OpenAI Whisper for multilingual ASR using SpeechBrain. It supports full fine-tuning and adapter-based methods (LoRA, Houlsby, MambAdapter) on Common Voice-style CSV datasets.

### Directory layout
- **hparams/**: Experiment YAMLs grouped by dataset/locale family (e.g., `AB/`, `CKB/`, `EO/`, `RW/`). Each subfolder has configurations for:
  - **FT**: Full fine-tuning (e.g., `AB_FT.yaml`).
  - **encoder/**, **decoder/**, **enc_dec/**: Adapter placements and variants (e.g., `*_LoRA.yaml`, `*_MambAdapter.yaml`).
- **train_ft.py**: Entry point for full fine-tuning.
- **train_ft_adapters.py**: Entry point for adapter-based training; dynamically inserts adapters per YAML.
- **results/**: Default output root for checkpoints, logs, and WER files organized by model variant and experiment.
- **plots/**: Pre-generated plots used in the paper/analysis.

### Data expectations
Datasets are expected in a folder with CSV splits: `train.csv`, `dev.csv`, `test.csv`. CSVs should reference audio paths with a `data_root` replacement and include at least the following columns used by the recipes:
- **mp3**: path to audio file
- **wrd**: transcript text
- **locale**: locale code (e.g., `en`, `ab`, `ckb`)

Point the recipe to your dataset root using the `--data_folder` CLI override or set `data_folder` directly in the YAML. Max durations for each split can be limited with `max_durations` in YAML.

### ⚡ Dataset [download](https://zenodo.org/record/8065754)

Our training recipes are adapted from the CL-MASR benchmark is extracted from [Common Voice 13](https://commonvoice.mozilla.org/en/datasets) (see [reference paper](https://arxiv.org/abs/1912.06670)). Each of the 20 languages in the dataset includes approximately 10 hours of training material, with an additional 1 hour designated for validation and another 1 hour for testing purposes.

Download the dataset from [here](https://zenodo.org/record/8065754) and extract it to a data folder of your choice (`CL-MASR` by default).

### Key YAML fields
Below are commonly used fields you may want to modify:
- **experiment_name**: Label for the run; used in output paths and logging.
- **locale_name**, **base_locales**, **new_locales**: Continual learning setup. Base locales are evaluated before training; new locales are added incrementally for training and evaluation.
- **adapter_type**: One of `FT`, `LoRA`, `Houlsby`, `MambAdapter` (primarily for logging and tagging).
- **whisper_variant**: Whisper checkpoint to load (e.g., `whisper-small`, `whisper-large-v2`).
- **freeze**, **freeze_encoder**, **freeze_decoder**: Component freezing controls.
- **train_batch_size**, **valid_batch_size**, workers, epochs, lr, schedulers: Training hyperparameters.
- **sorting**, **avoid_if_longer_than**: Data loading strategy and max utterance length filtering.
- **precision**, **gradient_checkpointing**: Memory/performance tuning.
- **output_folder**, **save_folder**: Where artifacts are written. Often templated from other fields.
- For adapters (adapter recipes only):
  - **projection_size** (rank), and for MambAdapter: **d_state**, **expand**, **kernel_size**, **alpha_init**.
  - **adapter_config**: Defines adapter class, target layers (e.g., `model.encoder.layers.*.mlp`), and kwargs.

### Running locally
Install dependencies from the project root as documented in the top-level README or the `requirements.txt` in this repo (SpeechBrain, torchaudio, hyperpyyaml, etc.). Then run one of the entrypoints:

Full fine-tuning example (AB locale):
```bash
cd Whisper
python train_ft.py hparams/AB/AB_FT.yaml --data_folder /path/to/data --scratch_folder /path/for/tmp --seed 12345
```

Adapter training example (decoder MambAdapter on AB):
```bash
cd Whisper
python train_ft_adapters.py hparams/AB/decoder/AB_MambAdapter.yaml \
  --data_folder /path/to/data \
  --scratch_folder /path/for/tmp \
  --seed 12345 \
  --projection_size 32 \
  --location decoder
```

Notes:
- Use `--projection_size` (rank) and `--location` for adapter runs; these are logged and often part of output path templates.
- `--scratch_folder` is required by some SLURM workflows but can be any writable path locally.
- The recipes automatically set the forced decoder locale per phase and handle tokenizer updates for new locales.

### Example: multi-seed adapter runs (inspired by internal scripts)
To reproduce multi-seed runs, you can loop in bash and draw random seeds in Python:
```bash
cd Whisper
for i in {1..5}; do
  seed=$(python - <<'PY'
import torch
print(torch.randint(0, 2**32-1, (1,)).item())
PY
)
  echo "Training with seed $seed"
  python train_ft_adapters.py hparams/AB/decoder/AB_MambAdapter.yaml \
    --data_folder /path/to/CL-MASR \
    --scratch_folder /tmp/whisper \
    --seed $seed \
    --projection_size 32 \
    --location decoder
done
```

### Outputs and checkpoints
By default, artifacts are stored under `results/<whisper_variant>/<experiment_name>/...` with subfolders for locale, location, rank, and seed depending on the YAML template. Inside each locale folder you will find:
- Checkpoints under `save/`
- Logs from the `WandBLogger` (to console/file; set `entity`/`project` in YAML to use remote WandB)
- `wer_test_before.txt`, `wer_test_after_<locale>.txt`: Evaluation reports

### Tips and common overrides
- Override data and scratch paths from CLI: `--data_folder`, `--scratch_folder`.
- Change seeds per run: `--seed`.
- Adjust rank/placement for adapters: `--projection_size`, `--location`.
- For large-batch or memory-constrained runs, tune `precision`, `gradient_checkpointing`, and batch sizes in YAML.

### Examples
- CKB MambAdapter (encoder MLP targets by default in YAML):
```bash
python train_ft_adapters.py hparams/CKB/encoder/CKB_MambAdapter.yaml \
  --data_folder /path/to/data \
  --scratch_folder /tmp/whisper \
  --seed 42 \
  --projection_size 24 \
  --location encoder
```

### Troubleshooting
- If you see out-of-memory errors, reduce `train_batch_size`, increase sorting strictness (`ascending`), enable `gradient_checkpointing`, or shorten `avoid_if_longer_than`.
- If WandB is not desired, leave `entity` and `project` empty or switch to a different logger.
- Ensure your CSVs include valid `mp3`, `wrd`, and `locale` columns, and that paths resolve after `data_root` replacement.


