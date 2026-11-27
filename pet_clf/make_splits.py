"""Build stratified train/val splits from the official trainval partition.

The official Oxford-IIIT Pet ``trainval`` partition is stratified by
breed by the dataset authors.  We further split it into ``train`` /
``val`` with stratification by breed so that no breed is held out
entirely from either subset.  The official ``test`` partition is left
untouched and written to disk as ``test.json`` so that evaluation is
later performed on exactly the same images every run.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import OxfordIIITPet

from . import config
from .seed import set_seed


def _list_dataset(split: str) -> list[tuple[str, int]]:
    """Return list of (filestem, label) for the given official split.

    The torchvision dataset sorts samples alphabetically, but the actual
    filename for an index ``i`` is not directly exposed; we therefore use
    the underlying ``_image_data`` (or its equivalent) through the public
    ``_labels`` attribute combined with the image paths.
    """
    ds = OxfordIIITPet(
        root=config.DATA_ROOT_PARENT, split=split, target_types=["category"]
    )
    items: list[tuple[str, int]] = []
    # OxfordIIITPet stores ``self._images`` and ``self._labels`` lists.
    for img_path, label in zip(ds._images, ds._labels):
        # img_path looks like ``.../images/Abyssinian_1.jpg`` -> store stem only.
        items.append((pathlib.Path(img_path).stem, int(label)))
    return items


def main() -> None:
    set_seed(config.SEED)

    trainval = _list_dataset("trainval")
    test = _list_dataset("test")
    print(f"trainval size: {len(trainval)}  test size: {len(test)}")

    files = np.array([x[0] for x in trainval])
    labels = np.array([x[1] for x in trainval])

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.VAL_FRACTION, random_state=config.SEED
    )
    train_idx, val_idx = next(sss.split(files, labels))

    splits = {
        "train": files[train_idx].tolist(),
        "val": files[val_idx].tolist(),
    }
    train_labels = {f: int(l) for f, l in zip(files[train_idx], labels[train_idx])}
    val_labels = {f: int(l) for f, l in zip(files[val_idx], labels[val_idx])}
    test_labels = {f: int(l) for f, l in test}

    # Write each split as a JSON file with explicit labels.
    for name, file_list, label_map in [
        ("train", splits["train"], train_labels),
        ("val", splits["val"], val_labels),
        ("test", [x[0] for x in test], test_labels),
    ]:
        items = [{"file": f, "label": label_map[f]} for f in file_list]
        out = {"split": name, "seed": config.SEED, "items": items}
        with (config.SPLITS_DIR / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"wrote {name}: {len(items)} items")

    # Per-breed counts per split, for the report.
    counts = {}
    for name, label_map in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        per_class = np.zeros(config.NUM_CLASSES, dtype=int)
        for lab in label_map.values():
            per_class[lab] += 1
        counts[name] = per_class.tolist()

    with (config.SPLITS_DIR / "class_counts.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": OxfordIIITPet(
                    root=config.DATA_ROOT_PARENT, target_types=["category"]
                ).classes,
                "counts": counts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("wrote class_counts.json")


if __name__ == "__main__":
    main()
