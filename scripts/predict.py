"""Run the committed crop-disease checkpoint against one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model import DiseaseClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a JPEG, PNG, or WebP leaf image.")
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        choices=range(1, 21),
        metavar="N",
        help="Number of ranked classes to print (1-20; default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    try:
        with Image.open(args.image) as image:
            ranked = DiseaseClassifier().predict(image, topk=args.topk)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not classify {args.image}: {exc}") from exc

    top_label, top_confidence = ranked[0]
    response = {
        "class": top_label,
        "confidence": top_confidence,
        "topk": [
            {"class": label, "confidence": confidence}
            for label, confidence in ranked
        ],
    }
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()