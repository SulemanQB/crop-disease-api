# crop-disease-api

Upload a leaf photo and the service returns a plant-disease class (or healthy), a confidence score, and a short ranked list. Inference is a ResNet18 checkpoint running on CPU through a small FastAPI app.

## Problem

Telling common leaf diseases apart from a photo is slow to do by hand and easy to get wrong early in the season. This service is a first-pass classifier for that task: one image in, a class label and a probability out.

It was trained on studio-style PlantVillage leaves (one leaf, plain background, even light). It is a demo of that distribution, not a field diagnosis and not a replacement for a crop advisor.

## Results

Figures below are from the checkpoint in `artifacts/model.pt`. The machine-readable copy is `artifacts/metrics.json`.

Training froze the ImageNet backbone and fit only the classification head on [geraldmc/plantvillage-tiny](https://huggingface.co/datasets/geraldmc/plantvillage-tiny) (CC0, revision `v0.1.0`): 1,524 train images, 376 held-out images, 38 classes. Optimizer was AdamW (learning rate 0.001, batch size 32, seed 42) for 6 epochs on CPU. Chance accuracy on 38 classes is about 2.6%.

| Metric | Result |
| --- | --- |
| Architecture | ResNet18, ImageNet-1K initialization, frozen backbone |
| Held-out top-1 accuracy | 79.26% (298 / 376) |
| Median inference latency | 9.44 ms |
| 95th percentile inference latency | 11.45 ms |

Latency is one call to the inference function on CPU: resize, center crop, forward pass, and softmax. It does not include HTTP. A separate in-process `POST /predict` of `tests/fixtures/sample.jpg` on the same machine was 12.88 ms median and 14.4 ms at the 95th percentile (20 calls after 3 warmup calls).

Held-out top-1 by epoch (the saved checkpoint is epoch 6):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| Top-1 | 55.05% | 69.68% | 73.67% | 77.66% | 80.05% | 79.26% |

The tiny subset has about 10 test images per class, so this accuracy is noisy. It is not a result for full PlantVillage or for phone photos taken in a field. `scripts/train.py` can keep training on a larger ImageFolder dataset; pass `--arch efficientnet_b0` to fine-tune EfficientNet-B0 the same way.

## How to run

Python 3.11 or newer. The commands below use the CPU PyTorch wheels pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Predict:

```bash
curl -s -F "file=@tests/fixtures/sample.jpg" http://localhost:8000/predict
```

Optional query `topk` (default 3, max 20):

```bash
curl -s -F "file=@tests/fixtures/sample.jpg" "http://localhost:8000/predict?topk=5"
```

Response shape:

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

Set `MODEL_PATH` to use a different checkpoint file. Class names are read from `labels.json` in the same directory. The default path is `artifacts/model.pt`. See `.env.example`.

### Docker

```bash
docker compose up --build
```

The image listens on port 8000 and sets `MODEL_PATH=/app/artifacts/model.pt`.

### Tests

```bash
pytest
```

The smoke test posts `tests/fixtures/sample.jpg` and checks the response shape. That file is one held-out leaf the checkpoint labels correctly (`tests/fixtures/expected_label.txt`).

### Train again

```bash
pip install -r requirements-train.txt
python scripts/download_data.py
python scripts/train.py
```

`download_data.py` writes `data/plantvillage/train` and `data/plantvillage/test` (gitignored). `train.py` overwrites `artifacts/model.pt`, `artifacts/labels.json`, and `artifacts/metrics.json`. Any ImageFolder tree with those two split directories works, including a PlantDoc export you arrange yourself. PlantDoc is not downloaded or redistributed here; follow that dataset's own terms.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process is up and the checkpoint loaded |
| `POST` | `/predict` | Multipart field `file` (JPEG, PNG, or WebP, up to 8 MB) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/openapi.json` | OpenAPI schema |

`/health` returns `status`, `model_loaded`, `arch`, and `num_classes`.

## Layout

```
app/main.py          routes
app/model.py         checkpoint load and CPU inference
app/schemas.py       response models
scripts/download_data.py
scripts/train.py
artifacts/model.pt   committed checkpoint
artifacts/labels.json
artifacts/metrics.json
tests/test_api.py
tests/fixtures/sample.jpg
```

## Data and license

Code is MIT. See `LICENSE`.

Training images are PlantVillage, used here via the CC0 tiny subset. Citation, subset details, and the PlantDoc note are in `DATA_LICENSE.md`.
