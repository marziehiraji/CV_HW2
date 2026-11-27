"""Approach 1: full fine-tuning of MobileNetV3-Small."""
from __future__ import annotations

import json
import pathlib

import torch
from torch.utils.data import DataLoader

from . import config
from .data import PetDataset, train_transform, eval_transform, load_split
from .model import build_mobilenet, set_trainable_full, count_trainable_parameters
from .seed import set_seed
from .trainer import train_model


def main() -> None:
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_items = load_split("train")
    val_items = load_split("val")
    train_ds = PetDataset(config.DATA_DIR, train_items, transform=train_transform())
    val_ds = PetDataset(config.DATA_DIR, val_items, transform=eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=False, drop_last=False,
    )

    model = build_mobilenet(config.NUM_CLASSES).to(device)
    set_trainable_full(model)
    n_params = count_trainable_parameters(model)
    print(f"trainable params (full): {n_params:,}")

    ckpt = config.CHECKPOINT_DIR / "full_finetune.pt"
    log = config.RESULTS_DIR / "full_finetune_log.json"
    summary = train_model(
        model, train_loader, val_loader,
        epochs=config.EPOCHS_FULL, lr=config.LR_FULL,
        weight_decay=config.WEIGHT_DECAY, device=device,
        ckpt_path=ckpt, log_path=log,
    )
    summary["trainable_params"] = n_params
    summary["approach"] = "full_finetune"
    summary["epochs_requested"] = config.EPOCHS_FULL
    summary["lr"] = config.LR_FULL
    summary["weight_decay"] = config.WEIGHT_DECAY
    summary["batch_size"] = config.BATCH_SIZE
    with log.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
