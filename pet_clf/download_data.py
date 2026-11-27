"""Auto-download the Oxford-IIIT Pet dataset.

This module downloads and extracts the Oxford-IIIT Pet dataset via torchvision.
If the files already exist at the configured location, nothing is downloaded.

Usage:
    python -m pet_clf.download_data            # default path
    PET_DATA_DIR=/path python -m pet_clf.download_data   # custom path

Afterwards run:
    python -m pet_clf.make_splits
to produce the stratified train/val splits.
"""
from __future__ import annotations  # enable postponed-evaluation type hints
import os  # to read environment variables
import pathlib  # cross-platform path handling
import sys  # interpreter access for logging output

# Make the repo root importable so we can import config regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pet_clf import config  # central configuration (dataset paths, etc.)


def main() -> None:
    """Download/extract the dataset, or confirm it is already present."""
    # Parent directory under which torchvision builds the oxford-iiit-pet folder.
    root = config.DATA_ROOT_PARENT
    print(f"Dataset parent directory: {root}")
    print("Checking / downloading Oxford-IIIT Pet dataset ...")
    # Import torchvision lazily so that an uninstalled dependency produces a
    # clearer error and keeps the initial import cheap.
    from torchvision.datasets import OxfordIIITPet  # torchvision dataset class

    # Check whether the dataset has already been downloaded and extracted.
    images_dir = config.DATA_DIR / "images"  # extracted images folder
    ann_dir = config.DATA_DIR / "annotations"  # extracted annotations folder
    trainval_txt = ann_dir / "trainval.txt"  # official trainval split file
    test_txt = ann_dir / "test.txt"  # official test split file

    # If both images and annotation split files are present, skip re-downloading.
    if images_dir.exists() and trainval_txt.exists() and test_txt.exists():
        # Count how many images are already on disk.
        n_images = len(list(images_dir.glob("*.jpg")))
        # Report that the dataset is already prepared.
        print(f"Dataset already prepared: {n_images} images in {images_dir}")
        return  # early exit, nothing to do

    # Otherwise build the dataset with download=True so torchvision fetches it.
    # On first call, images (~800 MB) and annotations (~14 MB) are downloaded
    # from robots.ox.ac.uk and then extracted under the same path.
    ds = OxfordIIITPet(
        root=str(root),  # dataset root path
        split="trainval",  # use the official trainval partition
        target_types=["category"],  # we only want the 37-class breed label
        download=True,  # download and extract if missing
    )
    # Confirm the download/extraction completed successfully.
    print(f"Download/extraction complete. trainval samples: {len(ds)}")
    # Report the number of classes (breeds).
    print(f"Number of classes (breeds): {len(ds.classes)}")
    # Print the first 5 class names as a sanity check.
    print(f"Sample classes: {ds.classes[:5]} ...")


if __name__ == "__main__":
    # When this file is executed directly, call the main function.
    main()
