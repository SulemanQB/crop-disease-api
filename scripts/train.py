"""Fine-tune a pretrained classifier on an ImageFolder plant-disease dataset.

The backbone (ResNet18 or EfficientNet-B0) stays frozen. Only the new
classification head is trained, which keeps a CPU run short. The held-out
``test`` split is used for the accuracy number written to ``artifacts/metrics.json``.

Example::

    python scripts/download_data.py
    python scripts/train.py --data-dir data/plantvillage
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model import IMAGENET_MEAN, IMAGENET_STD, DiseaseClassifier, eval_transform, save_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be at least 1")
        return parsed

    def positive_float(value: str) -> float:
        parsed = float(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be greater than 0")
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/plantvillage"))
    parser.add_argument("--arch", choices=("resnet18", "efficientnet_b0"), default="resnet18")
    parser.add_argument("--epochs", type=positive_int, default=6)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument(
        "--fixture-image",
        type=Path,
        default=Path("tests/fixtures/sample.jpg"),
        help="Copy of a correctly classified test image for the API smoke test.",
    )
    parser.add_argument(
        "--dataset-name",
        default="geraldmc/plantvillage-tiny",
        help="Label stored in metrics. Does not download data.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_pretrained(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(arch)


def make_loaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(
            f"Expected ImageFolder splits at {train_dir} and {test_dir}. "
            "Run scripts/download_data.py first."
        )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = eval_transform()
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    if train_ds.classes != test_ds.classes:
        raise SystemExit("Train and test class folders do not match.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_ds.classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    # Frozen BatchNorm layers should stay in eval so running stats are not updated
    # from a tiny batch. Dropout is unused on ResNet18; EfficientNet dropout stays
    # active because it lives on the trainable head.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    total_loss = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        batch = targets.size(0)
        total_loss += float(loss.item()) * batch
        seen += batch
    return total_loss / max(seen, 1)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, str | None, str | None]:
    """Return top-1 accuracy and the path/label of the first correct test image."""
    model.eval()
    dataset: datasets.ImageFolder = loader.dataset  # type: ignore[assignment]
    correct = 0
    total = 0
    seen = 0
    fixture_path: str | None = None
    fixture_label: str | None = None
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += int(targets.size(0))
        if fixture_path is None:
            matches = (preds == targets).nonzero(as_tuple=False)
            if len(matches):
                offset = int(matches[0].item())
                index = seen + offset
                fixture_path = dataset.samples[index][0]
                fixture_label = dataset.classes[int(targets[offset].item())]
        seen += int(targets.size(0))
    accuracy = correct / total if total else 0.0
    return accuracy, fixture_path, fixture_label


def measure_latency(model_path: Path, image_path: Path, runs: int = 20) -> dict[str, float]:
    classifier = DiseaseClassifier(model_path)
    image = Image.open(image_path).convert("RGB")
    for _ in range(3):
        classifier.predict(image, topk=3)
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        classifier.predict(image, topk=3)
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    mid = len(samples) // 2
    p95_index = min(len(samples) - 1, max(0, int(round(0.95 * (len(samples) - 1)))))
    return {
        "latency_ms_median": round(samples[mid], 2),
        "latency_ms_p95": round(samples[p95_index], 2),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, labels = make_loaders(args.data_dir, args.batch_size)
    model = build_pretrained(args.arch, len(labels))
    model.to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    print(
        f"device={device.type} arch={args.arch} classes={len(labels)} "
        f"train={len(train_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        accuracy, _, _ = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch}/{args.epochs} loss={loss:.4f} test_top1={accuracy:.4f}",
            flush=True,
        )

    accuracy, fixture_src, fixture_label = evaluate(model, test_loader, device)
    model.to("cpu")
    save_checkpoint(args.output, model, labels, args.arch)

    if fixture_src and fixture_label:
        args.fixture_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_src, args.fixture_image)
        args.fixture_image.with_name("expected_label.txt").write_text(fixture_label + "\n", encoding="utf-8")
        latency = measure_latency(args.output, args.fixture_image)
    else:
        latency = {"latency_ms_median": None, "latency_ms_p95": None}

    metrics = {
        "arch": args.arch,
        "epochs": args.epochs,
        "frozen_backbone": True,
        "dataset": args.dataset_name,
        "num_classes": len(labels),
        "train_images": len(train_loader.dataset),
        "test_images": len(test_loader.dataset),
        "top1_accuracy": round(accuracy, 4),
        "device": device.type,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        **latency,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
