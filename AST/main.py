from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

from src.AST import AST
from src.AST_LoRA import AST_LoRA, AST_LoRA_ablation
from src.AST_adapters import AST_adapter, AST_adapter_ablation
from dataset.fluentspeech import FluentSpeech
from dataset.esc_50 import ESC_50
from dataset.urban_sound_8k import Urban_Sound_8k
from dataset.google_speech_commands_v2 import Google_Speech_Commands_v2
from dataset.iemocap import IEMOCAP
from utils.engine import eval_one_epoch, train_one_epoch


# Config & Utilities

@dataclass
class MambaConfig:
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 5e-3
    weight_decay: float = 1e-1
    batch_size: int = 32

    method: str = "adapter" # {"adapter", "LoRA"}
    reduction_rate_adapter: int = 64
    adapter_type: str = "Pfeiffer"
    adapter_block: str = ""
    kernel_size: int = 31

    # Data / runtime
    dataset_name: str = "ESC-50" # {"FSC","ESC-50","urbansound8k","GSC","IEMOCAP"}
    data_path: str = ""
    cache_dir: str = ""
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 10

    model_ckpt_AST: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    model_ckpt_wav: str = "facebook/wav2vec2-base-960h"
    max_len_audio: int = 128000

    use_wandb: bool = True
    project_name: str = "ASTAdapter"
    entity: str = ""
    save_best_ckpt: bool = False
    output_path: str = "./checkpoints"

    seq_or_par: str = "parallel"
    apply_residual: bool = False
    is_AST: bool = True
    mamba: MambaConfig = field(default_factory=MambaConfig)

    exp_name: str = ""

    def to_dict_for_wandb(self) -> Dict:
        d = asdict(self)
        d["mamba_parameters"] = d.pop("mamba")
        return d


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: Optional[int] = None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_parameters(model: torch.nn.Module) -> Tuple[int, int, int]:
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    head_params = 0
    if hasattr(model, "classification_head"):
        head = getattr(model, "classification_head")
        if hasattr(head, "weight") and head.weight is not None:
            head_params += head.weight.numel()
        if hasattr(head, "bias") and head.bias is not None:
            head_params += head.bias.numel()

    n_trainable_ex_head = n_trainable - head_params
    return n_total, n_trainable_ex_head, head_params


# Data & Model Factories
def load_train_yaml(path: str = "hparams/train.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dataset_splits(config: TrainConfig, train_params: Dict) -> Dict:
    """Return split metadata, folds lists, and training schedule for the chosen dataset."""
    name = config.dataset_name
    meta = {}

    if name == "FSC":
        meta.update(dict(
            max_len_AST=train_params["max_len_AST_FSC"],
            num_classes=train_params["num_classes_FSC"],
            batch_size=train_params["batch_size_FSC"],
            epochs=train_params["epochs_FSC_AST"] if config.is_AST else train_params["epochs_FSC_WAV"],
            fold_number=1,
            has_val=True,
        ))
    elif name == "ESC-50":
        meta.update(dict(
            max_len_AST=train_params["max_len_AST_ESC"],
            num_classes=train_params["num_classes_ESC"],
            batch_size=train_params["batch_size_ESC"],
            epochs=train_params["epochs_ESC_AST"],
            fold_number=5,
            has_val=True,
            folds_train=[[1,2,3],[2,3,4],[3,4,5],[4,5,1],[5,1,2]],
            folds_valid=[[4],[5],[1],[2],[3]],
            folds_test=[[5],[1],[2],[3],[4]],
        ))
    elif name == "urbansound8k":
        meta.update(dict(
            max_len_AST=train_params["max_len_AST_US8K"],
            num_classes=train_params["num_classes_US8K"],
            batch_size=train_params["batch_size_US8K"],
            epochs=train_params["epochs_US8K"],
            fold_number=10,
            has_val=False,
            folds_train=[
                [1,2,3,4,5,6,7,8,9],
                [2,3,4,5,6,7,8,9,10],
                [3,4,5,6,7,8,9,10,1],
                [4,5,6,7,8,9,10,1,2],
                [5,6,7,8,9,10,1,2,3],
                [6,7,8,9,10,1,2,3,4],
                [7,8,9,10,1,2,3,4,5],
                [8,9,10,1,2,3,4,5,6],
                [9,10,1,2,3,4,5,6,7],
                [10,1,2,3,4,5,6,7,8],
            ],
            folds_test=[[10],[1],[2],[3],[4],[5],[6],[7],[8],[9]],
        ))
    elif name == "GSC":
        meta.update(dict(
            max_len_AST=train_params["max_len_AST_GSC"],
            num_classes=train_params["num_classes_GSC"],
            batch_size=train_params["batch_size_GSC"],
            epochs=train_params["epochs_GSC_AST"] if config.is_AST else train_params["epochs_GSC_WAV"],
            fold_number=1,
            has_val=True,
        ))
    elif name == "IEMOCAP":
        meta.update(dict(
            max_len_AST=train_params["max_len_AST_IEMO"],
            num_classes=train_params["num_classes_IEMO"],
            batch_size=train_params["batch_size_IEMO"],
            epochs=train_params["epochs_IEMO"],
            fold_number=10,
            has_val=True,
            sessions_train=[[1,2,3,4],[1,2,3,4],[2,3,4,5],[2,3,4,5],[3,4,5,1],[3,4,5,1],[4,5,1,2],[4,5,1,2],[5,1,2,3],[5,1,2,3]],
            session_val=[[5],[5],[1],[1],[2],[2],[3],[3],[4],[4]],
            speaker_id_val=['F','M','F','M','F','M','F','M','F','M'],
            speaker_id_test=['M','F','M','F','M','F','M','F','M','F'],
        ))
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    meta["final_output"] = train_params["final_output"]
    return meta


def build_dataloaders(
    config: TrainConfig,
    meta: Dict,
    fold_idx: int,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    name = config.dataset_name
    max_len_AST = meta["max_len_AST"]
    batch_size = meta["batch_size"]

    if name == "FSC":
        train_data = FluentSpeech(config.data_path, max_len_AST, train=True, apply_SpecAug=False,
                                  few_shot=False, samples_per_class=64)
        val_data   = FluentSpeech(config.data_path, max_len_AST, train="valid")
        test_data  = FluentSpeech(config.data_path, max_len_AST, train=False)

    elif name == "ESC-50":
        train_data = ESC_50(
            config.data_path, max_len_AST, 'train',
            train_fold_nums=meta["folds_train"][fold_idx],
            valid_fold_nums=meta["folds_valid"][fold_idx],
            test_fold_nums=meta["folds_test"][fold_idx],
            apply_SpecAug=True, few_shot=False, samples_per_class=64
        )
        val_data = ESC_50(
            config.data_path, max_len_AST, 'valid',
            train_fold_nums=meta["folds_train"][fold_idx],
            valid_fold_nums=meta["folds_valid"][fold_idx],
            test_fold_nums=meta["folds_test"][fold_idx]
        )
        test_data = ESC_50(
            config.data_path, max_len_AST, 'test',
            train_fold_nums=meta["folds_train"][fold_idx],
            valid_fold_nums=meta["folds_valid"][fold_idx],
            test_fold_nums=meta["folds_test"][fold_idx]
        )

    elif name == "urbansound8k":
        train_data = Urban_Sound_8k(
            config.data_path, max_len_AST, 'train',
            train_fold_nums=meta["folds_train"][fold_idx],
            test_fold_nums=meta["folds_test"][fold_idx],
            apply_SpecAug=True, few_shot=False, samples_per_class=64,
        )
        val_data = None
        test_data = Urban_Sound_8k(
            config.data_path, max_len_AST, 'test',
            train_fold_nums=meta["folds_train"][fold_idx],
            test_fold_nums=meta["folds_test"][fold_idx],
        )

    elif name == "GSC":
        train_data = Google_Speech_Commands_v2(
            config.data_path, max_len_AST, 'train', apply_SpecAug=False, few_shot=False, samples_per_class=64
        )
        val_data = Google_Speech_Commands_v2(config.data_path, max_len_AST, 'valid')
        test_data = Google_Speech_Commands_v2(config.data_path, max_len_AST, 'test')

    else:
        train_data = IEMOCAP(
            config.data_path, config.max_len_audio, max_len_AST,
            sessions=meta["sessions_train"][fold_idx],
            speaker_id='both', is_AST=config.is_AST,
            apply_SpecAug=False, few_shot=False, samples_per_class=64
        )
        val_data = IEMOCAP(
            config.data_path, config.max_len_audio, max_len_AST,
            sessions=meta["session_val"][fold_idx],
            speaker_id=meta["speaker_id_val"][fold_idx],
            is_AST=config.is_AST
        )
        test_data = IEMOCAP(
            config.data_path, config.max_len_audio, max_len_AST,
            sessions=meta["session_val"][fold_idx],
            speaker_id=meta["speaker_id_test"][fold_idx],
            is_AST=config.is_AST
        )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = None if val_data is None else DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader, test_loader


def build_model(config: TrainConfig, meta: Dict, device: torch.device) -> Tuple[torch.nn.Module, float]:
    max_len_AST = meta["max_len_AST"]
    num_classes = meta["num_classes"]
    final_output = meta["final_output"]

    if config.method.lower() == "adapter":
        model = AST_adapter(
            max_length=max_len_AST,
            num_classes=num_classes,
            final_output=final_output,
            reduction_rate=config.reduction_rate_adapter,
            adapter_type=config.adapter_type,
            seq_or_par=config.seq_or_par,
            apply_residual=config.apply_residual,
            adapter_block=config.adapter_block,
            kernel_size=config.kernel_size,
            model_ckpt=config.model_ckpt_AST,
            mamba_config=asdict(config.mamba),
            cache_dir=config.cache_dir,
        ).to(device)
        lr = config.learning_rate

    elif config.method.lower() == "lora":
        model = AST_LoRA(
            max_length=max_len_AST,
            num_classes=num_classes,
            final_output=final_output,
            rank=6,
            alpha=16,
            model_ckpt=config.model_ckpt_AST,
        ).to(device)
        lr = config.learning_rate

    else:
        raise ValueError(f"Unsupported method: {config.method}")

    return model, lr


def maybe_init_wandb(config: TrainConfig, sweep_like_config: Dict) -> None:
    if not config.use_wandb:
        return
    if not _WANDB_AVAILABLE:
        logging.warning("wandb is not installed; proceeding without W&B logging.")
        return
    # name/tag choices: keep adapter_block for continuity
    wandb.init(
        project=config.project_name,
        name=config.adapter_block,
        entity=config.entity or None,
        tags=[config.dataset_name, config.seq_or_par, config.adapter_block, config.adapter_type],
        config=sweep_like_config,
    )


def train(config: TrainConfig) -> None:
    setup_logging()
    start_time = dt.datetime.now()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    # Load training constants from YAML
    train_params = load_train_yaml("hparams/train.yaml")
    meta = dataset_splits(config, train_params)

    # Prepare output dir
    out_dir = Path(config.output_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B (optional). Preserve your original sweep-config layout:
    sweep_like = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "mamba_parameters": asdict(config.mamba),
        "reduction_rate_adapter": config.reduction_rate_adapter,
        "data_path": config.data_path,
        "adapter_block": config.adapter_block,
        "seed": config.seed,
        "adapter_type": config.adapter_type,
        "kernel_size": config.kernel_size,
        "dataset_name": config.dataset_name,
        "method": config.method,
        "cache_dir": config.cache_dir,
        "use_wandb": config.use_wandb,
        "seq_or_par": config.seq_or_par,
    }
    maybe_init_wandb(config, sweep_like)

    fold_number = meta["fold_number"]
    epochs = meta["epochs"]
    accuracy_folds: List[float] = []

    logging.info(f"Dataset: {config.dataset_name} | Folds: {fold_number} | Epochs: {epochs}")
    logging.info(f"Device: {device} | Batch size: {meta['batch_size']} | LR: {config.learning_rate}")

    # Train across folds
    best_params_across_folds = None
    n_total = n_trainable_ex_head = head_params = 0

    for fold in range(fold_number):
        logging.info(f"=== Fold {fold+1}/{fold_number} ===")

        train_loader, val_loader, test_loader = build_dataloaders(config, meta, fold)
        model, lr = build_model(config, meta, device)

        # Parameter reporting (once)
        if fold == 0:
            n_total, n_trainable_ex_head, head_params = count_parameters(model)
            logging.info(f"Model params (total): {n_total:,}")
            logging.info(f"Trainable params (excluding head): {n_trainable_ex_head:,}")
            logging.info(f"Classification head params: {head_params:,}")

        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = -1.0
        best_state_dict = None

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion)
            if val_loader is not None:
                val_loss, val_acc = eval_one_epoch(model, val_loader, device, criterion)
            else:
                # urbansound8k has no val; evaluate on test each epoch as before
                val_loss, val_acc = eval_one_epoch(model, test_loader, device, criterion)

            current_lr = optimizer.param_groups[0]["lr"]

            logging.info(
                f"[Fold {fold+1}] Epoch {epoch+1:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% lr={current_lr:.6f}"
            )

            if config.use_wandb and _WANDB_AVAILABLE:
                wandb.log({
                    "train_loss": train_loss, "valid_loss": val_loss,
                    "train_accuracy": train_acc, "val_accuracy": val_acc,
                    "lr": current_lr, "fold": fold,
                    "epoch": epoch
                })

            if val_acc > best_acc:
                best_acc = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                if config.save_best_ckpt:
                    ckpt_path = out_dir / f"bestmodel_{config.dataset_name}_fold{fold}.pt"
                    torch.save(best_state_dict, ckpt_path)
                    logging.info(f"Saved checkpoint: {ckpt_path}")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        test_loss, test_acc = eval_one_epoch(model, test_loader, device, criterion)
        accuracy_folds.append(test_acc)

        logging.info(f"[Fold {fold+1}] TEST: loss={test_loss:.4f} acc={test_acc*100:.2f}%")
        logging.info(f"Folds so far: { [round(a*100,2) for a in accuracy_folds] }")
        logging.info(f"Avg acc: {np.mean(accuracy_folds)*100:.2f}% | Std: {np.std(accuracy_folds)*100:.2f}%")

        if config.use_wandb and _WANDB_AVAILABLE:
            wandb.log({
                "test_loss": test_loss, "test_accuracy": test_acc,
                "avg_accuracy_over_folds": float(np.mean(accuracy_folds)),
                "std_accuracy_over_folds": float(np.std(accuracy_folds)),
            })

        if best_params_across_folds is None and best_state_dict is not None:
            best_params_across_folds = best_state_dict

    percent_trained = (n_trainable_ex_head / n_total * 100) if n_total > 0 else 0.0
    logging.info(f"Params summary: total={n_total:,}, trainable_ex_head={n_trainable_ex_head:,}, head={head_params:,}")
    logging.info(f"% params trained (excl. head): {percent_trained:.2f}%")

    if config.use_wandb and _WANDB_AVAILABLE:
        wandb.log({
            "#params_total": n_total,
            "#params_trainable_ex_head": n_trainable_ex_head,
            "#params_head": head_params,
            "%params_trained_ex_head": percent_trained,
        })

    elapsed = dt.datetime.now() - start_time
    logging.info(f"Training time: {str(elapsed).split('.')[0]}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)

    # training hyperparams
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-3, dest="learning_rate")
    p.add_argument("--weight-decay", type=float, default=1e-1, dest="weight_decay")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")

    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)

    p.add_argument("--reduction_rate_adapter", type=int, default=64)
    p.add_argument("--adapter_type", type=str, default="Pfeiffer")
    p.add_argument("--adapter_block", type=str, default="")
    p.add_argument("--kernel_size", type=int, default=31)
    p.add_argument("--method", type=str, default="adapter", choices=["adapter", "LoRA"])

    p.add_argument("--dataset_name", type=str, default="ESC-50",
                   choices=["FSC", "ESC-50", "urbansound8k", "GSC", "IEMOCAP"])
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=10)

    # logging/persistence
    p.add_argument("--use_wandb", type=bool, default=True)
    p.add_argument("--project_name", type=str, default="ASTAdapter")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--save_best_ckpt", action="store_true")
    p.add_argument("--output_path", type=str, default="./checkpoints")

    args = p.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        method=args.method,
        reduction_rate_adapter=args.reduction_rate_adapter,
        adapter_type=args.adapter_type,
        adapter_block=args.adapter_block,  # preserved
        kernel_size=args.kernel_size,
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        cache_dir=args.cache_dir,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        entity=args.entity,
        save_best_ckpt=bool(args.save_best_ckpt),
        output_path=args.output_path,
        mamba=MambaConfig(d_state=args.d_state, d_conv=args.d_conv, expand=args.expand),
    )
    return config


if __name__ == "__main__":
    config = parse_args()
    train(config)