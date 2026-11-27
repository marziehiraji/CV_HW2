"""Evaluate the selected checkpoints for all three approaches on the test set.

This script:
  * loads the checkpoint kept by approach 1 and approach 2 (best val_acc);
  * evaluates each on the test set once, computing accuracy, macro F1,
    per-class precision/recall/F1, and a confusion matrix;
  * reads the test predictions already produced by approach 3;
  * saves a single ``eval_results.json`` used by the report generator.

The test set is loaded ONLY here and NEVER during training.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from . import config
from .data import PetDataset, eval_transform, load_split
from .model import build_mobilenet
from .seed import set_seed


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        preds.append(out.argmax(dim=1).cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": [
            {"class": cls, "precision": float(p), "recall": float(r), "f1": float(f1)}
            for cls, p, r, f1 in zip(classes, per_class_p, per_class_r, per_class_f1)
        ],
        "confusion_matrix": cm.tolist(),
    }


def eval_approach(name: str, classes: list[str], device) -> dict:
    items = load_split("test")
    ds = PetDataset(config.DATA_DIR, items, transform=eval_transform())
    loader = DataLoader(
        ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=False, drop_last=False,
    )

    if name in ("full_finetune", "head_train"):
        ckpt_path = config.CHECKPOINT_DIR / f"{name}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = build_mobilenet(config.NUM_CLASSES).to(device)
        model.load_state_dict(ckpt["model_state"])
        t0 = time.time()
        preds, ys = predict(model, loader, device)
        eval_time = time.time() - t0
    else:
        # Approach 3: predictions already saved in features_log.json
        with (config.RESULTS_DIR / "features_log.json").open() as f:
            feat_log = json.load(f)
        chosen = feat_log["selected_classifier"]
        preds = np.array(feat_log["classifiers"][chosen]["test_predictions"])
        ys = np.array([it[1] for it in items])
        eval_time = 0.0  # already accounted for in features pipeline

    res = metrics_dict(ys, preds, classes)
    res["eval_time_sec"] = eval_time
    res["predictions"] = preds.tolist()
    res["true_labels"] = ys.tolist()
    return res


def main() -> None:
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from .data import class_names
    classes = class_names()

    results = {}
    for name in ("full_finetune", "head_train", "feature_extraction"):
        print(f"--- evaluating {name} ---")
        results[name] = eval_approach(name, classes, device)
        print(f"  acc={results[name]['accuracy']:.4f}  macro_f1={results[name]['macro_f1']:.4f}")

    out = config.RESULTS_DIR / "eval_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": classes,
                "results": results,
                "seed": config.SEED,
            },
            f, indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
