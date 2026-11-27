# Oxford-IIIT Pet — Transfer-Learning Comparison (Homework 2)

Image classification on the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
dataset comparing three transfer-learning strategies with a pre-trained
MobileNetV3-Small:

1. **Full fine-tuning** — train the entire network end-to-end.
2. **Classifier-head training** — freeze the convolutional feature
   extractor, train only the fully connected classifier head.
3. **Feature extraction + scikit-learn** — freeze the network, extract
   the 576-d pooled features, fit `LogisticRegression` (with
   `LinearSVC` as an additional comparison), evaluate on the held-out
   test set.

This repository is the supporting code for the homework report PDF. The
report describes the dataset, methods, evaluation protocol, results,
analysis, and limitations; this README documents the environment and
the exact commands needed to reproduce every reported number.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.12.16 |
| PyTorch | 2.15.0+cpu |
| TorchVision | 0.29.2+cpu |
| scikit-learn | 1.5.2 |
| NumPy | 2.1.3 |
| Pillow | 11.3.0 |
| pandas | 2.2.3 |
| matplotlib | 3.9.2 |

CPU-only (`torch.cuda.is_available() == False`).

Install everything with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn numpy pillow pandas matplotlib
```

## Dataset

Oxford-IIIT Pet, 37 cat/dog breeds, ~7,349 images total. We use the
official `trainval` (3,680 images) and `test` (3,669 images) partitions
shipped with the dataset's `annotations/trainval.txt` and
`annotations/test.txt`. The `trainval` partition is further split into
`train` (3,128 images) and `val` (552 images) by **stratified random
sampling on breed** (15% validation, seed 42) — implemented in
`pet_clf/make_splits.py`. The test partition is held out and used
**only once** per approach for the final evaluation.

- **Source**: https://www.robots.ox.ac.uk/~vgg/data/pets/
- **License**: The Oxford-IIIT Pet dataset is released for
  non-commercial research and educational use; see the dataset page
  for the full terms. Redistribution is not permitted; obtain the
  dataset through `torchvision.datasets.OxfordIIITPet(download=True)`
  or directly from the source above.

Per-breed counts in each split (min / max / mean across the 37 breeds):

| split | total | min | max | mean |
| --- | --- | --- | --- | --- |
| train | 3,128 | 79 | 85 | 84.5 |
| val   |   552 | 14 | 15 | 14.9 |
| test  | 3,669 | 88 | 100 | 99.2 |

The splits are balanced by design; no class-weighting or
class-balancing is required for the training loop. The full
per-class breakdown is in `splits/class_counts.json`.

## Reproducibility

All experiments use `seed = 42` (see `pet_clf/config.py`), seeded into
Python's `random`, NumPy, and PyTorch. `cudnn.deterministic` is not
enabled because the host is CPU-only; the `CUBLAS_WORKSPACE_CONFIG`
environment variable is not set. With CPU kernels, small numerical
differences can still occur between hardware/libmkl versions — see the
"Limitations" section of the report.

To download the dataset (≈ 800 MB for images, ≈ 14 MB for annotations)
automatically — this is the recommended path and is also the first step
of `run_all.sh`:

```bash
python -m pet_clf.download_data            # uses the default PET_DATA_DIR
# or with a custom location:
PET_DATA_DIR=/path/to/oxford-iiit-pet python -m pet_clf.download_data
```

The `download_data.py` module wraps
`torchvision.datasets.OxfordIIITPet(download=True)` and is idempotent:
if the images and annotation split files are already present at the
configured location, it does nothing.

## Running the experiments

From this repository root:

```bash
export PET_DATA_DIR=/path/to/oxford-iiit-pet     # contains images/ and annotations/
export PYTHONPATH=.

python -m pet_clf.download_data     # auto-download dataset (skipped if already present)
python -m pet_clf.make_splits       # build stratified train/val + test JSON splits
python -m pet_clf.train_full        # Approach 1: full fine-tuning
python -m pet_clf.train_head        # Approach 2: classifier-head training
python -m pet_clf.train_features    # Approach 3: feature extraction + sklearn
python -m pet_clf.evaluate          # evaluate all three on the held-out test set
python -m pet_clf.make_figures      # generate confusion matrices, examples, results_table.csv
```

Or, in one command:

```bash
PET_DATA_DIR=/path/to/oxford-iiit-pet PYTHONPATH=. bash run_all.sh
```

Or use the Python orchestrator (recommended — it gives you per-stage
control and a single traceback on failure):

```bash
python run.py                       # full pipeline (download -> splits -> train -> eval -> figures)
python run.py --skip-train          # reuse existing checkpoints, re-evaluate + re-plot
python run.py --only make_figures   # regenerate just the figures/tables
python run.py --start 5             # start at the evaluate stage
python run.py --help                # list all options
```

`run.py` imports each `pet_clf.*` module and calls its `main()` in order,
so any failure surfaces as a single Python traceback. By default it points
`PET_DATA_DIR` at `./data/oxford-iiit-pet` (next to this README); override it
with the environment variable or by editing `pet_clf/config.py`.

Recorded hyperparameters (also stored in `results/*_log.json`):

| Approach | Optimizer | LR | Weight decay | Batch | Epochs | LR schedule |
| --- | --- | --- | --- | --- | --- | --- |
| Full fine-tuning | AdamW | 1e-4 | 1e-4 | 64 | 8 (early-stop patience 4) | fixed |
| Classifier-head training | AdamW | 1e-3 | 1e-4 | 64 | 15 (early-stop patience 4) | fixed |
| Feature extraction + sklearn | `LogisticRegression` (lbfgs, C=1.0, balanced, max_iter=2000) | n/a | n/a | — | — | n/a |

Selection rule for the neural approaches: keep the checkpoint with the
highest **validation accuracy** (a deterministic integer-count metric)
and stop early if it has not improved for 4 consecutive epochs. The
selected checkpoint is evaluated **once** on the test set in
`pet_clf/evaluate.py`.

For Approach 3 the classifier is selected by **validation macro-F1**
(no test set involvement); the LinearSVC is included only as a
comparison and is reported alongside LogReg in
`results/features_log.json`.

## Repository layout

```
pet_clf/
    __init__.py        # package marker + module docstring
    config.py          # seed, paths, hyperparameters
    seed.py            # RNG seeding helper
    data.py            # dataset wrapper + train/eval transforms
    download_data.py   # auto-download Oxford-IIIT Pet via torchvision (idempotent)
    make_splits.py     # stratified train/val + test JSON splits + per-breed counts
    model.py           # MobileNetV3-Small construction + trainable-parameter switches
    trainer.py         # training loop with checkpoint on val accuracy
    train_full.py      # Approach 1 entry point
    train_head.py      # Approach 2 entry point
    train_features.py  # Approach 3 entry point (feature extraction + sklearn)
    evaluate.py        # final test-set evaluation for all three approaches
    make_figures.py    # confusion matrices, prediction examples, results table
splits/
    train.json  val.json  test.json   # reproducible partitions (filename, label)
    class_counts.json                  # per-breed counts per split
results/
    full_finetune_log.json  head_train_log.json  features_log.json
    eval_results.json  summary_metrics.json  results_table.csv
    figures/
        confusion_full.png  confusion_head.png  confusion_features.png
        examples_full.png   examples_head.png
checkpoints/
    full_finetune.pt  head_train.pt
run_all.sh           # runs the whole pipeline (bash)
run.py               # runs the whole pipeline (Python orchestrator, recommended)
README.md            # this file
```

## Test-set evaluation protocol

The held-out test partition (3,669 images, 88-100 per breed) is used
**exactly once** per approach, after all model selection is complete.
We never use it for early stopping, feature selection, threshold
selection, or hyperparameter tuning. Reported metrics:

- Test accuracy;
- Macro-averaged F1-score (unweighted across the 37 breeds);
- A 37 × 37 confusion matrix with breed names on both axes.

Results are stored in `results/eval_results.json` and summarized in
`results/results_table.csv`.
