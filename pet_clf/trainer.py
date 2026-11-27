"""Generic training loop with checkpointing on validation accuracy.

The loop:
  * trains for at most ``epochs`` epochs;
  * after every epoch, evaluates on the validation set;
  * keeps the checkpoint with the highest validation accuracy
    (a stable, deterministic metric) on disk;
  * stops early if validation accuracy has not improved for
    ``PATIENCE`` consecutive epochs.

The test set is NEVER touched in this module.
"""
from __future__ import annotations

import json
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config


def epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    ckpt_path: pathlib.Path,
    log_path: pathlib.Path,
    patience: int = config.PATIENCE,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    history: list[dict] = []
    best_acc = -1.0
    best_epoch = -1
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = epoch_train(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = epoch_eval(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )
        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({epoch_time:.1f}s)"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch} (no val_acc gain for {patience} epochs)")
                break

    total_time = time.time() - start_time
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "total_time_sec": total_time,
        "history": history,
    }
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"best epoch {best_epoch} val_acc={best_acc:.4f}  saved {ckpt_path.name}")
    return summary
