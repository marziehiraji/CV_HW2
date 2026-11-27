"""Quick sanity test: load a few batches, forward + one backward step.

Run with the same PYTHONPATH/PET_DATA_DIR as the real training modules.
"""
from __future__ import annotations
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from pet_clf import config
from pet_clf.data import PetDataset, train_transform, eval_transform, load_split
from pet_clf.model import (
    build_mobilenet, set_trainable_full, set_trainable_head_only,
    build_feature_extractor, count_trainable_parameters,
)
from pet_clf.seed import set_seed


def main():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_items = load_split("train")
    val_items = load_split("val")
    train_ds = PetDataset(config.DATA_DIR, train_items, transform=train_transform())
    val_ds = PetDataset(config.DATA_DIR, val_items, transform=eval_transform())

    # Tiny subset for sanity
    sub_train = Subset(train_ds, list(range(32)))
    sub_val = Subset(val_ds, list(range(32)))
    tl = DataLoader(sub_train, batch_size=16, shuffle=True, num_workers=0)
    vl = DataLoader(sub_val, batch_size=16, shuffle=False, num_workers=0)

    for mode, fn in [("full", set_trainable_full), ("head", set_trainable_head_only)]:
        model = build_mobilenet(config.NUM_CLASSES).to(device)
        fn(model)
        n_tr = count_trainable_parameters(model)
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        crit = nn.CrossEntropyLoss()
        model.train()
        t0 = time.time()
        imgs, labels = next(iter(tl))
        imgs = imgs.to(device); labels = labels.to(device)
        out = model(imgs)
        loss = crit(out, labels)
        loss.backward(); optim.step()
        print(f"  [{mode}] trainable={n_tr:,} out={tuple(out.shape)} loss={loss.item():.4f} step={time.time()-t0:.2f}s")

    # Feature extractor
    ext, dim = build_feature_extractor()
    ext = ext.to(device)
    ext.eval()
    imgs, labels = next(iter(vl))
    imgs = imgs.to(device)
    with torch.no_grad():
        feats = ext(imgs)
    print(f"  [feat] dim={dim} feats={tuple(feats.shape)}")

    print("sanity OK")


if __name__ == "__main__":
    main()
