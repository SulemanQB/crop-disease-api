"""Checkpoint loading and CPU inference for the crop-disease classifier."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "model.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

SUPPORTED_ARCHS = ("resnet18", "efficientnet_b0")


def eval_transform(
    input_size: int = INPUT_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> transforms.Compose:
    """Match the validation preprocessing used in scripts/train.py."""
    return transforms.Compose(
        [
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Construct an untrained backbone with a new classification head.

    Weights come from the checkpoint. ImageNet initialization happens only in training.
    """
    if arch == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture '{arch}'. Expected one of {SUPPORTED_ARCHS}.")


def resolve_model_path() -> Path:
    raw = os.environ.get("MODEL_PATH")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_MODEL_PATH


def _load_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Checkpoint at {path} is missing a 'state_dict' entry.")
    return payload


def _load_labels(model_path: Path, payload: dict) -> list[str]:
    labels_path = model_path.parent / "labels.json"
    embedded = payload.get("labels")
    if labels_path.is_file():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise RuntimeError(f"{labels_path} must be a JSON list of class name strings.")
        if embedded is not None and list(embedded) != labels:
            raise RuntimeError(
                f"Labels in {labels_path} do not match the labels stored in {model_path}."
            )
        return labels
    if isinstance(embedded, list) and embedded:
        return [str(item) for item in embedded]
    raise RuntimeError(f"No labels found next to {model_path} (expected labels.json).")


class DiseaseClassifier:
    """ResNet18 or EfficientNet-B0 classifier loaded on CPU."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = Path(model_path) if model_path is not None else resolve_model_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at {path}. "
                "Set MODEL_PATH or train one with scripts/train.py."
            )
        payload = _load_payload(path)
        labels = _load_labels(path, payload)
        arch = str(payload.get("arch", "resnet18"))
        self.model_path = path
        self.arch = arch
        self.labels = labels
        self.input_size = int(payload.get("input_size", INPUT_SIZE))
        mean = payload.get("mean", IMAGENET_MEAN)
        std = payload.get("std", IMAGENET_STD)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.transform = eval_transform(self.input_size, self.mean, self.std)
        self.model = build_model(arch, len(labels))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self._lock = threading.Lock()

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0)

    @torch.inference_mode()
    def predict(self, image: Image.Image, topk: int = 3) -> list[tuple[str, float]]:
        """Return up to ``topk`` (label, probability) pairs, highest first."""
        k = max(1, min(int(topk), self.num_classes))
        batch = self.preprocess(image)
        with self._lock:
            logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=k)
        return [
            (self.labels[int(idx)], float(val))
            for val, idx in zip(values.tolist(), indices.tolist(), strict=True)
        ]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    labels: list[str],
    arch: str,
) -> None:
    """Write model.pt and a sibling labels.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "labels": list(labels),
        "input_size": INPUT_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    torch.save(payload, path)
    labels_path = path.parent / "labels.json"
    labels_path.write_text(json.dumps(list(labels), indent=2) + "\n", encoding="utf-8")
