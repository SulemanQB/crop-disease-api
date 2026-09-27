"""Smoke tests for health and predict. They load the committed CPU checkpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample.jpg"
EXPECTED = ROOT / "tests" / "fixtures" / "expected_label.txt"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_classes"] >= 2
    assert body["arch"] in {"resnet18", "efficientnet_b0"}


def test_predict_smoke(client: TestClient) -> None:
    assert FIXTURE.is_file(), "Missing tests/fixtures/sample.jpg. Train or add a fixture image."
    with FIXTURE.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"file": ("sample.jpg", handle, "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["class"], str) and body["class"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["topk"], list) and body["topk"]
    assert body["topk"][0]["class"] == body["class"]
    confidences = [item["confidence"] for item in body["topk"]]
    assert confidences == sorted(confidences, reverse=True)
    assert abs(body["confidence"] - confidences[0]) < 1e-6
    if EXPECTED.is_file():
        expected = EXPECTED.read_text(encoding="utf-8").strip()
        assert body["class"] == expected


def test_predict_topk_query(client: TestClient) -> None:
    with FIXTURE.open("rb") as handle:
        response = client.post(
            "/predict?topk=5",
            files={"file": ("sample.jpg", handle, "image/jpeg")},
        )
    assert response.status_code == 200
    assert len(response.json()["topk"]) == 5


def test_predict_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_openapi_docs(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    assert "/predict" in paths
    assert "/health" in paths
