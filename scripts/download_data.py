"""Download a PlantVillage image subset into an ImageFolder tree.

Default source is the public Hugging Face dataset ``geraldmc/plantvillage-tiny``
(CC0-1.0): 50 images per class across the 38 PlantVillage classes. The original
collection is the PlantVillage repository (Hughes & Salathé; Mohanty, Hughes &
Salathé). See DATA_LICENSE.md.

The script writes::

    data/plantvillage/train/<class>/*.jpg
    data/plantvillage/test/<class>/*.jpg

``scripts/train.py`` can also train on any other dataset laid out the same way,
including a manually prepared PlantDoc export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="geraldmc/plantvillage-tiny",
        help="Hugging Face dataset id. Default is the CC0 PlantVillage tiny subset.",
    )
    parser.add_argument("--revision", default="v0.1.0")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/plantvillage"),
        help="Output root. Split folders are created underneath.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class, counted across both splits. Useful for a dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, revision=args.revision, split="train")
    out_root: Path = args.out
    counts: dict[tuple[str, str], int] = {}
    saved = 0

    for index, row in enumerate(dataset):
        label = str(row["class_label"])
        split = str(row.get("split") or "train")
        if split not in {"train", "test"}:
            split = "train"
        key = (split, label)
        used = counts.get(key, 0)
        if args.max_per_class is not None and used >= args.max_per_class:
            continue
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Row {index} did not contain a PIL image.")
        dest_dir = out_root / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{index:05d}.jpg"
        image.convert("RGB").save(dest, format="JPEG", quality=90)
        counts[key] = used + 1
        saved += 1
        if saved % 200 == 0:
            print(f"saved {saved} images...", flush=True)

    print(f"Wrote {saved} images under {out_root.resolve()}")
    splits = sorted({split for split, _ in counts})
    labels = sorted({label for _, label in counts})
    print(f"splits={splits} classes={len(labels)}")


if __name__ == "__main__":
    main()
