"""Generate the figures and tables used in the report.

Outputs (under results/figures/):
  * confusion_full.png, confusion_head.png, confusion_features.png
  * examples_full.png, examples_head.png  (correct + incorrect images)
  * results_table.csv and results_table.tex

Also writes ``summary_metrics.json`` used by the report PDF generator.
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from PIL import Image

from . import config


def load_classes() -> list[str]:
    with (config.RESULTS_DIR / "eval_results.json").open() as f:
        return json.load(f)["classes"]


def load_eval() -> dict:
    with (config.RESULTS_DIR / "eval_results.json").open() as f:
        return json.load(f)


def load_logs() -> dict:
    out = {}
    for name in ("full_finetune", "head_train", "features"):
        p = config.RESULTS_DIR / f"{name.replace('features', 'features')}_log.json"
        if name == "features":
            p = config.RESULTS_DIR / "features_log.json"
        else:
            p = config.RESULTS_DIR / f"{name}_log.json"
        if p.exists():
            with p.open() as f:
                out[name] = json.load(f)
    return out


def plot_confusion(cm: list[list[int]], classes: list[str], title: str, out: pathlib.Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_examples(name: str, classes: list[str], preds: list[int], trues: list[int], items: list[tuple[str, int]], out: pathlib.Path) -> None:
    """Pick 3 correct + 3 incorrect test images and tile them."""
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    correct_idx = np.where(preds == trues)[0]
    wrong_idx = np.where(preds != trues)[0]
    n_correct = min(3, len(correct_idx))
    n_wrong = min(3, len(wrong_idx))
    rng = np.random.default_rng(config.SEED)
    correct_pick = rng.choice(correct_idx, n_correct, replace=False)
    wrong_pick = rng.choice(wrong_idx, n_wrong, replace=False)
    picks = list(correct_pick) + list(wrong_pick)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, idx in zip(axes.flat, picks):
        file, label = items[idx]
        img_path = config.DATA_DIR / "images" / f"{file}.jpg"
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        true_cls = classes[trues[idx]]
        pred_cls = classes[preds[idx]]
        ok = trues[idx] == preds[idx]
        color = "green" if ok else "red"
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls}", color=color, fontsize=9)
    fig.suptitle(f"{name}: top row correct, bottom row incorrect", fontsize=12)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def results_table(eval_data: dict, logs: dict, classes: list[str]) -> tuple[list[list[str]], dict]:
    """Build the comparison table.

    Columns: Approach, trainable params, total train time (s),
    train acc (best epoch), val acc (best epoch), test acc, macro F1.
    """
    rows = []
    summary = {}
    for name in ("full_finetune", "head_train", "feature_extraction"):
        r = eval_data["results"][name]
        log = logs.get("full_finetune" if name == "full_finetune" else
                       "head_train" if name == "head_train" else "features")
        if name == "feature_extraction":
            n_params = 0  # all frozen
            # total time = feature extraction + classifier training time
            feat_time = log["feature_extraction_time_sec"]
            cls_time = sum(
                v["train_time_sec"]
                for k, v in log["classifiers"].items()
                if k == log["selected_classifier"]
            )
            total_time = feat_time + cls_time
            train_acc = None
            val_acc = log["classifiers"][log["selected_classifier"]]["val_acc"]
        else:
            n_params = log["trainable_params"]
            total_time = log["total_time_sec"]
            # training accuracy of the best epoch
            best_epoch = log["best_epoch"]
            hist = [h for h in log["history"] if h["epoch"] == best_epoch][0]
            train_acc = hist["train_acc"]
            val_acc = hist["val_acc"]
        rows.append([
            name,
            f"{n_params:,}",
            f"{total_time:.1f}",
            "-" if train_acc is None else f"{train_acc:.4f}",
            f"{val_acc:.4f}",
            f"{r['accuracy']:.4f}",
            f"{r['macro_f1']:.4f}",
        ])
        summary[name] = {
            "trainable_params": n_params,
            "total_time_sec": total_time,
            "train_acc_best_epoch": train_acc,
            "val_acc": val_acc,
            "test_acc": r["accuracy"],
            "test_macro_f1": r["macro_f1"],
        }
    return rows, summary


def main() -> None:
    classes = load_classes()
    eval_data = load_eval()
    logs = load_logs()

    # Confusion matrices
    for name, label in [
        ("full_finetune", "Full fine-tuning"),
        ("head_train", "Classifier-head training"),
        ("feature_extraction", "Feature extraction + LogReg"),
    ]:
        cm = eval_data["results"][name]["confusion_matrix"]
        plot_confusion(
            cm, classes,
            f"Confusion matrix — {label}",
            config.FIGURES_DIR / f"confusion_{name}.png",
        )
        print(f"wrote confusion_{name}.png")

    # Prediction examples for the two NN approaches (need access to test item lists)
    from .data import load_split
    test_items = load_split("test")
    for name in ("full_finetune", "head_train"):
        preds = eval_data["results"][name]["predictions"]
        trues = eval_data["results"][name]["true_labels"]
        plot_examples(
            name, classes, preds, trues, test_items,
            config.FIGURES_DIR / f"examples_{name}.png",
        )
        print(f"wrote examples_{name}.png")

    # Comparison table
    rows, summary = results_table(eval_data, logs, classes)
    headers = ["Approach", "Trainable params", "Total time (s)",
               "Best-epoch train acc", "Val acc", "Test acc", "Macro F1"]
    csv_path = config.RESULTS_DIR / "results_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"wrote {csv_path}")

    summary_path = config.RESULTS_DIR / "summary_metrics.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "classes": classes,
                "approaches": summary,
                "train_history": {
                    "full_finetune": logs.get("full_finetune", {}).get("history", []),
                    "head_train": logs.get("head_train", {}).get("history", []),
                },
            },
            f, indent=2,
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
