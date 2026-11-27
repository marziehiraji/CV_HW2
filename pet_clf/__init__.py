"""Image classification pipeline for Oxford-IIIT Pet dataset.

Comparing three transfer-learning strategies with MobileNetV3-Small:
  1. Full fine-tuning
  2. Classifier-head training (frozen conv feature extractor)
  3. Feature extraction + sklearn classifier

Run sub-commands (from repo root):
    python -m pet_clf.download_data   # auto-download the dataset
    python -m pet_clf.make_splits     # build stratified train/val splits
    python -m pet_clf.train_full      # approach 1
    python -m pet_clf.train_head      # approach 2
    python -m pet_clf.train_features  # approach 3
    python -m pet_clf.evaluate        # evaluate selected checkpoints on test set
    python -m pet_clf.make_figures    # generate figures and tables for the report
"""
