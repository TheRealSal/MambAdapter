# MambAdapter: Shared Selective State Space Adapters for Speech and Audio

This repository provides the official implementation of **MambAdapter**, a parameter-efficient transfer learning (PETL) approach that integrates **Mamba state-space modules** into **shared bottleneck adapters** for speech and audio models.  
It extends the adapter training framework originally developed by **[Cappellazzo et al. (2023)](https://github.com/umbertocappellazzo/PETL_AST)**, adapting their multi-dataset, multi-fold training code for our new Mamba-based architecture.

---

## ✳️ Overview

MambAdapter combines:
- Shared bottleneck projections across adapter layers.
- Lightweight Mamba blocks for long-range temporal modeling.
- Learnable per-layer scaling factors for adaptive modulation.

This design allows efficient fine-tuning of the Audio Spectrogram Transformer across multiple datasets with minimal parameter overhead.

---

## ⚙️ Installation

```bash
cd MambAdapter
pip install -r requirements.txt
```

**Dependencies:**
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- torchaudio ≥ 2.0  
- PyYAML, numpy, tqdm  
- mamba_ssm
- wandb *(optional for experiment tracking)*

---

## 🚀 Usage

The training entry point is **`main.py`**.  
It supports both **Adapter** and **LoRA** fine-tuning.

### Example 1: Adapters on ESC-50
```bash
python main.py \
  --dataset_name ESC-50 \
  --data_path /path/to/ESC-50 \
  --adapter_block conformer \
  --adapter_type Pfeiffer \
  --kernel_size 31 \
  --epochs 10
```

```bash
python main.py \
  --dataset_name ESC-50 \
  --data_path /path/to/ESC-50 \
  --adapter_block mambadapter \
  --adapter_type Pfeiffer \
  --d_conv 4 \
  --expand 2 \
  --d_state 16 \
  --epochs 10 \
  --use_wandb
```

### Example 2: LoRA on GSC
```bash
python main.py \
  --dataset_name GSC \
  --data_path /path/to/GSC \
  --method LoRA \
  --epochs 10 \
```

### Boolean Flags

| Flag | Description |
|------|--------------|
| `--use_wandb true|false` | Enable/disable Weights & Biases logging. |
| `--save_best_ckpt` | Save the best model per fold under `--output_path`. |

---

## 📦 Datasets

Supported datasets:
- **ESC-50** (5-fold)
- **UrbanSound8K** (10-fold, no validation set)
- **Google Speech Commands (GSC)**
- **Fluent Speech Commands (FSC)**

Each dataset retrieve their configurations from `hparams/train.yaml`.

---

## 🧠 Mamba Hyperparameters

Applicable only for **MambAdapter**:

| Argument | Description | Default |
|-----------|-------------|----------|
| `--d_state` | Mamba hidden state dimension | 16 |
| `--d_conv`  | Convolution kernel size | 4 |
| `--expand`  | Channel expansion factor | 2 |

### Example:
```bash
python main.py --dataset_name ESC-50 \
  --data_path /data/ESC-50 \
  --method adapter \
  --d_state 32 --d_conv 8 --expand 3
```