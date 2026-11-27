"""Dataset and transform construction for Oxford-IIIT Pet.

We use torchvision's ``OxfordIIITPet`` class, which exposes the official
``trainval`` and ``test`` partitions defined in the dataset's
``annotations/trainval.txt`` and ``annotations/test.txt`` files.  We
keep the test partition exactly as released and further split the
trainval partition into ``train`` / ``val`` by stratified sampling on
breed (see ``make_splits.py``).
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights

from . import config


# ------------------------------------------------------------------
# Preprocessing consistent with the pre-trained MobileNetV3-Small weights
# ------------------------------------------------------------------
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
# These transforms match the IMAGENET1K_V1 normalization: 224x224 resize
# + center crop for evaluation, and the same mean/std normalization.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


# ------------------------------------------------------------------
# Split-aware wrapper
# ------------------------------------------------------------------
class PetDataset(Dataset):
    """A subset of Oxford-IIIT Pet defined by a list of (filename, label) tuples.

    Using a fixed list of filenames guarantees that the train/val split is
    reproducible across runs regardless of the order in which the
    underlying ``OxfordIIITPet`` loader iterates its files.
    """

    def __init__(
        self,
        root: pathlib.Path,
        sample_list: list[tuple[str, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = pathlib.Path(root)
        self.images_folder = self.root / "images"
        self.transform = transform
        self.samples = sample_list  # list of (relative_filename_without_ext, label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_name, label = self.samples[idx]
        # Pet images are JPGs named exactly by their listing in annotations
        img_path = self.images_folder / f"{rel_name}.jpg"
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ------------------------------------------------------------------
# Helpers for loading splits from JSON
# ------------------------------------------------------------------
def load_split(split_name: str) -> list[tuple[str, int]]:
    """Load a split file written by ``make_splits``.

    The split JSON has the shape::

        {"split": "train", "items": [{"file": "Abyssinian_1", "label": 0}, ...]}
    """
    path = config.SPLITS_DIR / f"{split_name}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["file"], item["label"]) for item in data["items"]]


def class_names() -> list[str]:
    """Return the 37 breed names in label-index order."""
    # We instantiate the dataset purely to access the canonical class list.
    ds = OxfordIIITPet(
        root=config.DATA_ROOT_PARENT, split="trainval", target_types=["category"]
    )
    return list(ds.classes)


def class_to_species() -> pd.DataFrame:
    """Build a small table mapping each of the 37 breeds to cat / dog.

    Pet filenames begin with the breed name; the official dataset groups
    25 dog breeds and 12 cat breeds. We identify species by inspecting the
    per-image binary label exposed by torchvision, which is derived from
    ``annotations/list.txt``.
    """
    ds = OxfordIIITPet(
        root=config.DATA_ROOT_PARENT,
        split="trainval",
        target_types=["category", "binary-category"],
    )
    rows = {}
    for _, (cat_label, bin_label) in ds:
        if cat_label not in rows:
            rows[cat_label] = bin_label
    # binary-category: 1 = dog, 0 = cat (per torchvision convention)
    rows = {label: "dog" if v == 1 else "cat" for label, v in rows.items()}
    return rows
