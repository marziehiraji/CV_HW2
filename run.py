"""Single-entry Python orchestrator for the Oxford-IIIT Pet classification pipeline.

Run every stage of the pipeline with one command:

    python run.py                # full pipeline (download -> splits -> train -> eval -> figures)
    python run.py --skip-train  # skip the three training stages (reuse checkpoints)
    python run.py --only figures # only regenerate the figures/tables from existing results
    python run.py --help         # show all options

The orchestrator does NOT shell out to `python -m pet_clf.*`; instead it imports
the modules and calls their `main()` so failures raise a single Python traceback.
Set PET_DATA_DIR to point at the dataset root (the folder that contains
`images/` and `annotations/`). If the dataset is missing, stage 0 downloads it
automatically via torchvision.

Stages:
    0. download_data      - auto-download Oxford-IIIT Pet (idempotent)
    1. make_splits        - build stratified train/val + test JSON splits
    2. train_full         - Approach 1: full fine-tuning
    3. train_head         - Approach 2: classifier-head training
    4. train_features     - Approach 3: feature extraction + scikit-learn
    5. evaluate           - evaluate all three on the held-out test set
    6. make_figures       - confusion matrices, prediction examples, results table
"""
from __future__ import annotations  # enable postponed-evaluation type hints

import argparse  # CLI argument parsing
import os  # environment variables and path manipulation
import sys  # interpreter access (argv, exit)
import time  # timing each stage

# Make the repo root importable regardless of where the user runs this script.
# This file lives at the repo root, so its directory is the repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))  # absolute repo path
if REPO_ROOT not in sys.path:  # avoid duplicate entries
    sys.path.insert(0, REPO_ROOT)  # allow `import pet_clf`
os.environ.setdefault("PYTHONPATH", REPO_ROOT)  # mirror the shell convention

# Allow the user to point PET_DATA_DIR at a custom dataset location.
# Default mirrors run_all.sh: a `data/` folder next to this script.
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "oxford-iiit-pet")
os.environ.setdefault("PET_DATA_DIR", DEFAULT_DATA_DIR)


# Ordered list of (stage-name, module-name) pairs. Each module must expose a
# top-level `main()` callable that takes no required arguments.
STAGES = [
    ("download_data", "pet_clf.download_data"),    # 0: fetch dataset via torchvision
    ("make_splits", "pet_clf.make_splits"),        # 1: stratified splits + class counts
    ("train_full", "pet_clf.train_full"),          # 2: Approach 1
    ("train_head", "pet_clf.train_head"),          # 3: Approach 2
    ("train_features", "pet_clf.train_features"),  # 4: Approach 3
    ("evaluate", "pet_clf.evaluate"),              # 5: test-set evaluation
    ("make_figures", "pet_clf.make_figures"),      # 6: figures + results table
]


def run_stage(module_name: str) -> None:
    """Import a pet_clf module and invoke its main() function.

    Raises SystemExit / ImportError / RuntimeError upward so the orchestrator
    stops at the first failing stage with a clear traceback.
    """
    print(f"\n>>> import {module_name}")  # progress log
    module = __import__(module_name, fromlist=["main"])  # load the module
    main_fn = getattr(module, "main", None)  # find the main callable
    if main_fn is None or not callable(main_fn):  # defensive check
        raise RuntimeError(f"{module_name} has no callable main()")  # clear error
    print(f">>> running {module_name}.main()")  # progress log
    main_fn()  # execute the stage


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the orchestrator."""
    parser = argparse.ArgumentParser(  # build the parser
        description="Run the full Oxford-IIIT Pet classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # preserve newlines
    )
    # Option to skip the three training stages (useful to re-evaluate existing
    # checkpoints without re-training).
    parser.add_argument(
        "--skip-train", action="store_true",
        help="skip stages 2/3/4 (train_full, train_head, train_features)",
    )
    # Option to run only a single named stage.
    parser.add_argument(
        "--only", choices=[name for name, _ in STAGES],
        help="run only this stage and exit",
    )
    # Option to start from a specific stage index (0-based).
    parser.add_argument(
        "--start", type=int, default=0, choices=range(len(STAGES)),
        help="start from this stage index (0-based, default 0)",
    )
    # Option to stop after a specific stage index (inclusive).
    parser.add_argument(
        "--stop", type=int, default=len(STAGES) - 1, choices=range(len(STAGES)),
        help="stop after this stage index (inclusive, default last)",
    )
    return parser.parse_args()  # return the parsed namespace


def main() -> int:
    """Entry point: parse args, run selected stages, report total time."""
    args = parse_args()  # CLI options
    print("=" * 70)
    print("Oxford-IIIT Pet — full pipeline")
    print(f"  PET_DATA_DIR = {os.environ.get('PET_DATA_DIR')}")
    print(f"  REPO_ROOT    = {REPO_ROOT}")
    print("=" * 70)

    # Build the list of stages to run based on CLI options.
    if args.only is not None:  # --only takes precedence
        indices = [i for i, (name, _) in enumerate(STAGES) if name == args.only]
    else:
        indices = list(range(args.start, args.stop + 1))  # inclusive stop
        if args.skip_train:  # remove the three training stages
            train_names = {"train_full", "train_head", "train_features"}
            indices = [i for i in indices
                       if STAGES[i][0] not in train_names]

    if not indices:  # nothing to do
        print("No stages selected. Exiting.")
        return 0  # success exit code

    total_start = time.time()  # wall clock for the whole run
    for i in indices:  # iterate selected stage indices
        name, module_name = STAGES[i]  # unpack the stage tuple
        print(f"\n{'=' * 70}\nSTAGE {i}: {name}\n{'=' * 70}")
        stage_start = time.time()  # wall clock for this stage
        try:
            run_stage(module_name)  # run the stage
        except Exception as exc:  # any failure stops the pipeline
            print(f"\n✗ STAGE {i} ({name}) FAILED: {exc}", file=sys.stderr)
            return 1  # error exit code
        elapsed = time.time() - stage_start  # seconds
        print(f"✓ STAGE {i} ({name}) done in {elapsed:.1f}s")

    total = time.time() - total_start  # total elapsed seconds
    print(f"\n{'=' * 70}\nALL STAGES DONE in {total:.1f}s\n{'=' * 70}")
    return 0  # success exit code


if __name__ == "__main__":
    # When executed directly, run main() and propagate its exit code.
    sys.exit(main())
