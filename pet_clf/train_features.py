"""Approach 3: feature extraction + scikit-learn classifier.

We use the frozen MobileNetV3-Small feature extractor to produce 576-d
pooled feature vectors for every image in train/val/test (no
augmentation, eval transforms only).  We then fit two classifiers:

  * LogisticRegression (max_iter=2000, C=1.0, balanced) — selected by
    default and reported as the canonical classifier;
  * LinearSVC + calibration (optional comparison).

We pick the best classifier on the validation set (macro F1) and only
evaluate it once on the test set.
"""
from __future__ import annotations

import json
import time
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from . import config
from .data import PetDataset, eval_transform, load_split
from .model import build_feature_extractor
from .seed import set_seed


@torch.no_grad()
def extract_features(extractor, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    extractor.eval()
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = extractor(imgs)
        feats.append(out.cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def build_classifiers():
    """Return a dict of {name: fitted-on-X-pipeline}. The pipelines
    include StandardScaler so feature scaling is fit on training data
    only."""
    return {
        "logreg": LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced",
            solver="lbfgs", n_jobs=2, random_state=config.SEED,
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(
                C=1.0, class_weight="balanced",
                max_iter=5000, random_state=config.SEED,
            ),
            cv=3,
        ),
    }


def main() -> None:
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    extractor, feat_dim = build_feature_extractor()
    extractor = extractor.to(device)
    print(f"feature dim: {feat_dim}")

    loaders = {}
    for name in ("train", "val", "test"):
        items = load_split(name)
        ds = PetDataset(config.DATA_DIR, items, transform=eval_transform())
        loaders[name] = DataLoader(
            ds, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=False, drop_last=False,
        )

    t0 = time.time()
    X_train, y_train = extract_features(extractor, loaders["train"], device)
    X_val, y_val = extract_features(extractor, loaders["val"], device)
    X_test, y_test = extract_features(extractor, loaders["test"], device)
    feature_time = time.time() - t0
    print(f"feature extraction time: {feature_time:.1f}s  shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    # Scaling fit on training features only.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # StandardScaler + classifier records feature-extraction + scaling time
    # in 'feature_time' and per-classifier training time separately.
    candidates = build_classifiers()
    results = {}
    chosen_name = None
    chosen_val_f1 = -1.0
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train_s, y_train)
        train_time = time.time() - t0
        val_pred = clf.predict(X_val_s)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        results[name] = {
            "train_time_sec": train_time,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        print(f"  {name}: val_acc={val_acc:.4f} val_f1={val_f1:.4f} train_time={train_time:.1f}s")
        if val_f1 > chosen_val_f1:
            chosen_val_f1 = val_f1
            chosen_name = name

    # Evaluate chosen classifier on the test set ONCE.
    chosen = candidates[chosen_name]
    test_pred = chosen.predict(X_test_s)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    results[chosen_name]["test_acc"] = test_acc
    results[chosen_name]["test_macro_f1"] = test_f1
    results[chosen_name]["test_predictions"] = test_pred.tolist()

    summary = {
        "approach": "feature_extraction",
        "feature_dim": feat_dim,
        "feature_extraction_time_sec": feature_time,
        "feature_scaling": "StandardScaler fit on train only",
        "selected_classifier": chosen_name,
        "selected_by": "val macro-F1 (no test tuning)",
        "classifiers": results,
    }
    out = config.RESULTS_DIR / "features_log.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Save test labels alongside predictions to keep eval reproducible.
    np.savez(
        config.RESULTS_DIR / "features_arrays.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"selected: {chosen_name}  test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
