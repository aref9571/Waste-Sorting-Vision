from http import HTTPStatus
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wastevision.config import Settings
from wastevision.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = ROOT / "tests" / "fixtures" / "sample.png"
RESPONSE_KEYS = {"category", "label", "confidence", "unsure", "top3"}


@pytest.fixture
def client():
    with TestClient(create_app(Settings(models_dir=ROOT / "models"))) as test_client:
        yield test_client


def test_sample_image_is_a_small_png():
    with Image.open(SAMPLE_IMAGE) as image:
        assert image.format == "PNG"
        assert image.size == (64, 48)
    assert SAMPLE_IMAGE.stat().st_size < 20_000


def test_sample_image_is_classified_with_the_keys_the_smoke_test_expects(client: TestClient):
    files = {"image": ("sample.png", SAMPLE_IMAGE.read_bytes(), "image/png")}
    response = client.post("/api/v1/classify", files=files)
    assert response.status_code == HTTPStatus.OK
    assert set(response.json()) == RESPONSE_KEYS
    assert len(response.json()["top3"]) == 3


def test_readiness_and_page_used_by_the_container_checks(client: TestClient):
    assert client.get("/health/ready").status_code == HTTPStatus.OK
    assert "<form" in client.get("/").text


def test_dockerfile_matches_the_service_and_has_no_torch():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 8000" in dockerfile
    assert "/health/ready" in dockerfile
    assert "USER wastevision" in dockerfile
    assert "torch" not in dockerfile


def test_dockerignore_keeps_the_model_and_drops_weights():
    entries = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert "models" not in entries
    assert "*.pt" in entries
