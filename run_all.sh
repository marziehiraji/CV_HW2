#!/usr/bin/env bash
# End-to-end run: download dataset, split, train all three approaches, evaluate, make figures.
# Run from /home/z/my-project/repo with PET_DATA_DIR set (defaults to a sensible path).
# The download step is skipped automatically if the dataset is already present.
set -euo pipefail
export PET_DATA_DIR="${PET_DATA_DIR:-/home/z/my-project/data/oxford-iiit-pet}"
export PYTHONPATH=.

echo "=== 0. download dataset (auto-skipped if already present) ==="
python -m pet_clf.download_data

echo "=== 1. splits ==="
python -m pet_clf.make_splits

echo "=== 2. train: full fine-tuning ==="
python -m pet_clf.train_full

echo "=== 3. train: classifier head ==="
python -m pet_clf.train_head

echo "=== 4. feature extraction + sklearn ==="
python -m pet_clf.train_features

echo "=== 5. evaluate on test set ==="
python -m pet_clf.evaluate

echo "=== 6. generate figures and tables ==="
python -m pet_clf.make_figures

echo "DONE"
