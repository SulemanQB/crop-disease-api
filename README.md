# Crop Disease Classifier API

An end-to-end computer-vision project that serves a ResNet18 plant-disease classifier through a small FastAPI service. Upload a leaf image and receive the predicted class, confidence, and a ranked shortlist of alternatives.

The repository contains a CPU inference API, a reusable PyTorch classifier, a transfer-learning script, dataset preparation tooling, a local CLI, tests, and a Docker runtime.

It covers model packaging, image preprocessing, inference API design, validation, experiment metadata, and the limits of a controlled-image dataset.

> This is a first-pass classifier for PlantVillage-style images. It is not a field diagnosis or a replacement for an agronomist.

## Problem and Motivation

Identifying a common leaf disease from a photograph can be slow and inconsistent. This project packages a small image-classification experiment as a service, including the path from data preparation to prediction.

The model was trained on studio-style PlantVillage images: generally one leaf, a plain background, and even lighting. The reported performance applies to that distribution, not to general field photographs.

## Key Features

- `POST /predict` accepts JPEG, PNG, and WebP uploads up to 8 MB.
- Returns the top class, softmax confidence, and configurable top-k results.
- Loads a committed CPU checkpoint during application startup.
- `GET /health` reports the loaded architecture and class count.
- Swagger/OpenAPI documentation is available at `/docs`.
- Local CLI inference uses the same classifier as the API.
- Training supports ResNet18 and EfficientNet-B0 with a generic ImageFolder layout.
- Tests cover health, prediction shape, ranking, invalid uploads, query validation, and OpenAPI availability.

## Architecture and Workflow

```mermaid
flowchart LR
    A[PlantVillage subset] --> B[Download to ImageFolder]
    B --> C[Resize, crop, normalize]
    C --> D[Transfer learning]
    D --> E[Evaluate on held-out split]
    E --> F[model.pt + labels.json + metrics.json]
    F --> G[FastAPI lifespan loads checkpoint]
    G --> H[POST /predict or local CLI]
    H --> I[Ranked class probabilities]
```

At inference time, the image is converted to RGB, resized and center-cropped to 224x224, normalized with ImageNet statistics, passed through the checkpoint, and converted to probabilities. The API layer owns upload validation; `app/model.py` owns model loading and inference.

## Results

The committed checkpoint reports the following values in [`artifacts/metrics.json`](artifacts/metrics.json). They describe this small experiment, not a general benchmark.

| Metric | Result |
| --- | --- |
| Dataset | `geraldmc/plantvillage-tiny`, revision `v0.1.0` |
| Architecture | ResNet18 with frozen ImageNet backbone and trainable head |
| Classes | 38 |
| Train / held-out images | 1,524 / 376 |
| Epochs | 6 |
| Held-out top-1 accuracy | 79.26% (298 / 376) |
| Median CPU inference | 9.44 ms |
| CPU inference p95 | 11.45 ms |

Latency measures preprocessing, forward pass, and softmax in-process. It excludes HTTP overhead and is hardware-dependent. The held-out split is small, so the accuracy is noisy and should not be generalized to field photographs or the full PlantVillage dataset.

## Technologies

| Area | Tools |
| --- | --- |
| API | Python, FastAPI, Uvicorn, Pydantic |
| ML | PyTorch, Torchvision, ResNet18, optional EfficientNet-B0 |
| Images | Pillow |
| Data | Hugging Face Datasets, ImageFolder |
| Quality | Pytest, Docker Compose |

## Project Structure

```text
app/
  main.py              FastAPI app, lifespan, routes, upload validation
  model.py             Checkpoint loading, preprocessing, inference
  schemas.py            Typed API response models
scripts/
  predict.py           Reproducible local inference demo
  download_data.py     Dataset-to-ImageFolder preparation
  train.py             Training, evaluation, latency, artifact writing
artifacts/
  model.pt             Committed CPU checkpoint for the demo
  labels.json          Class labels paired with the checkpoint
  metrics.json         Training and evaluation metadata
tests/
  test_api.py          API smoke and validation tests
  fixtures/            Small held-out image and expected label
DATA_LICENSE.md        Dataset provenance and licensing notes
Dockerfile              Minimal CPU runtime image
docker-compose.yml      Local container entry point
```

## Setup

Python 3.11 or newer is recommended. The committed runtime uses CPU PyTorch wheels; a supported PyTorch build is required for the selected Python version and platform.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The repository includes the checkpoint and labels, so training is not required for the API demo.

## Demonstration

### Local CLI

This is the fastest way to demonstrate inference without starting a server:

```bash
python -m scripts.predict tests/fixtures/sample.jpg
```

Example output shape:

```json
{
  "class": "Apple___Apple_scab",
  "confidence": 0.5121915340423584,
  "topk": [
    {"class": "Apple___Apple_scab", "confidence": 0.5121915340423584},
    {"class": "Apple___Black_rot", "confidence": 0.12800629436969757},
    {"class": "Squash___Powdery_mildew", "confidence": 0.04736236110329628}
  ]
}
```

Use `--topk 5` for a longer ranked list. The exact probabilities depend on the committed checkpoint and preprocessing metadata.

### API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs>, or use curl:

```bash
curl -s -F "file=@tests/fixtures/sample.jpg" http://localhost:8000/predict
curl -s -F "file=@tests/fixtures/sample.jpg" "http://localhost:8000/predict?topk=5"
curl -s http://localhost:8000/health
```

### Docker

```bash
docker compose up --build
```

The service listens on port 8000. The image includes the committed checkpoint and sets `MODEL_PATH=/app/artifacts/model.pt`.

## API Contract

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Confirms the process started with a loaded checkpoint |
| `POST` | `/predict` | Multipart field `file`; optional `topk` from 1 to 20 |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/openapi.json` | Generated API schema |

`/predict` returns `400` for empty or unreadable images, `413` for uploads over 8 MB, and `422` for invalid query parameters. The response uses the JSON key `class` for compatibility with the public API while the internal Pydantic field is named `class_name`.

## Configuration and Reproducibility

| Setting | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `artifacts/model.pt` | Checkpoint to load; `labels.json` is read beside it |
| `--data-dir` | `data/plantvillage` | Training ImageFolder root |
| `--arch` | `resnet18` | `resnet18` or `efficientnet_b0` |
| `--epochs` | `6` | Training epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `0.001` | AdamW learning rate |
| `--seed` | `42` | Python and PyTorch random seed |

Copy `.env.example` to `.env` for the model-path example. The application does not load `.env` automatically; export the variable in the shell or pass it through Docker Compose.

## Training and Evaluation

Training is intentionally separate from inference and is not needed for the demo:

```bash
python -m pip install -r requirements-train.txt
python scripts/download_data.py
python scripts/train.py
```

The downloader writes gitignored data under `data/plantvillage/`. The trainer expects `train/<class>` and `test/<class>` folders, freezes the pretrained backbone, trains a new classification head with AdamW and cross-entropy loss, evaluates held-out top-1 accuracy, measures CPU latency, and writes the three artifacts under `artifacts/`. Use `--arch efficientnet_b0` to train the supported alternative architecture.

The dataset provenance, upstream citations, and license notes are documented in [`DATA_LICENSE.md`](DATA_LICENSE.md). The code is MIT licensed; see [`LICENSE`](LICENSE).

## Technical Highlights

- Separated HTTP concerns from checkpoint loading and tensor inference.
- Stored preprocessing statistics and labels with the checkpoint to keep inference aligned with training.
- Added startup loading so a misconfigured checkpoint fails before serving predictions.
- Added bounded uploads and bounded `topk` input at the API boundary.
- Made the training script produce machine-readable metrics and a correctly classified fixture when available.
- Kept the runtime CPU-oriented and Dockerized.

## Limitations and Future Improvements

The model sees a small, controlled dataset and can be overconfident outside that distribution. It has no field-image validation, calibration, disease severity estimate, segmentation, authentication, rate limiting, or monitoring. The committed checkpoint is also intentionally a demo-scale experiment.

Potential follow-up work includes field-data evaluation, stronger train/validation methodology, confidence calibration, image-quality checks, structured logging and metrics, and a versioned model registry.

## What the code shows

1. Start with the architecture diagram and explain the data-to-inference path.
2. Open `app/model.py` and show how checkpoint metadata keeps labels and preprocessing together.
3. Run `python -m scripts.predict tests/fixtures/sample.jpg` and inspect the ranked JSON output.
4. Start Uvicorn, open `/docs`, and send the same image through `POST /predict`.
5. Explain why the API validates upload size and query bounds, and why model loading happens during lifespan startup.
6. Show `artifacts/metrics.json` and discuss why the result is useful for demonstration but not a field benchmark.
7. Discuss field-data validation, calibration, and operational controls as the next engineering steps.

## Tests and Troubleshooting

Run the test suite with:

```bash
pytest
```

If PyTorch cannot import, use a clean virtual environment and install the pinned CPU wheels from `requirements.txt`; the wheel must match your Python version and operating system. If startup reports a missing checkpoint, verify `MODEL_PATH` and ensure a sibling `labels.json` exists. If training reports missing folders, run the downloader or provide an ImageFolder tree with both `train` and `test` splits.

## Project Information

This repository contains the implementation, experiment metadata, dataset notes, and tests needed to inspect and run the PyTorch image-classification workflow without private infrastructure.
