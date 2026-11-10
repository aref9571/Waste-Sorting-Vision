from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wastevision.config import Settings
from wastevision.inference.predictor import Predictor, load_metadata
from wastevision.labels import CATEGORY_BY_CLASS, CLASS_NAMES
from wastevision.web.app import PredictorLoader, create_app

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "waste-mobilenetv3.onnx"
CLASSIFY_URL = "/api/v1/classify"
CONTENT_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

ClientFactory = Callable[..., TestClient]


def encoded_image(image_format: str = "PNG", size: tuple[int, int] = (320, 240)) -> bytes:
    pixels = np.random.default_rng(7).integers(0, 256, (size[1], size[0], 3), np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format=image_format)
    return buffer.getvalue()


def upload(client: TestClient, data: bytes, content_type: str = "image/png", field: str = "image"):
    return client.post(CLASSIFY_URL, files={field: ("photo", data, content_type)})


@pytest.fixture(scope="module")
def committed_predictor() -> Predictor:
    return Predictor.from_directory(MODELS_DIR)


@pytest.fixture
def make_client(
    monkeypatch: pytest.MonkeyPatch, committed_predictor: Predictor
) -> Iterator[ClientFactory]:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    open_clients: list[TestClient] = []

    def make(load_predictor: PredictorLoader | None = None, **settings_values) -> TestClient:
        settings_values.setdefault("models_dir", MODELS_DIR)
        app = create_app(
            Settings(**settings_values),
            load_predictor=load_predictor or (lambda _: committed_predictor),
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        open_clients.append(client)
        return client

    yield make
    for client in open_clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    return make_client()


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
def test_classify_returns_a_prediction(client: TestClient, image_format: str):
    response = upload(client, encoded_image(image_format), CONTENT_TYPES[image_format])

    body = response.json()
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert set(body) == {"category", "label", "confidence", "unsure", "top3"}
    assert body["label"] in CLASS_NAMES
    assert body["category"] == CATEGORY_BY_CLASS[body["label"]]
    assert isinstance(body["unsure"], bool)
    assert 0.0 <= body["confidence"] <= 1.0


def test_classify_lists_the_three_most_likely_classes(client: TestClient):
    body = upload(client, encoded_image()).json()

    top3 = body["top3"]
    probabilities = [entry["probability"] for entry in top3]
    assert len(top3) == 3
    assert all(set(entry) == {"label", "probability"} for entry in top3)
    assert len({entry["label"] for entry in top3}) == 3
    assert probabilities == sorted(probabilities, reverse=True)
    assert top3[0]["label"] == body["label"]
    assert top3[0]["probability"] == body["confidence"]


def test_classify_matches_the_predictor(client: TestClient, committed_predictor: Predictor):
    data = encoded_image(size=(500, 375))

    body = upload(client, data).json()
    expected = committed_predictor.predict(Image.open(BytesIO(data)))

    assert body["label"] == expected.label
    assert body["confidence"] == pytest.approx(expected.confidence)
    assert body["unsure"] == expected.unsure
    assert [entry["label"] for entry in body["top3"]] == [label for label, _ in expected.top]


@pytest.mark.parametrize(("threshold", "unsure"), [(1.0, True), (0.0, False)])
def test_classify_reports_unsure_below_the_threshold(
    make_client: ClientFactory, threshold: float, unsure: bool
):
    predictor = Predictor(MODEL_PATH, load_metadata(MODELS_DIR / "metadata.json"), threshold)
    client = make_client(load_predictor=lambda _: predictor)

    body = upload(client, encoded_image()).json()

    assert body["unsure"] is unsure


def test_health_live(client: TestClient):
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_health_ready_when_the_model_is_loaded(client: TestClient):
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_default_loader_reads_the_committed_model(make_client: ClientFactory):
    client = make_client(load_predictor=Predictor.from_directory)

    assert client.get("/health/ready").status_code == 200
    assert upload(client, encoded_image()).status_code == 200


def test_not_ready_without_a_model(make_client: ClientFactory, tmp_path: Path):
    client = make_client(load_predictor=Predictor.from_directory, models_dir=tmp_path)

    live = client.get("/health/live")
    ready = client.get("/health/ready")
    classify = upload(client, encoded_image())

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.headers["content-type"] == "application/problem+json"
    assert ready.json()["detail"] == "The model is not loaded."
    assert classify.status_code == 503


def test_a_broken_model_file_is_logged_and_not_ready(
    make_client: ClientFactory, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    (tmp_path / "metadata.json").write_text(
        (MODELS_DIR / "metadata.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "waste-mobilenetv3.onnx").write_bytes(b"not a model")

    client = make_client(load_predictor=Predictor.from_directory, models_dir=tmp_path)

    assert client.get("/health/ready").status_code == 503
    assert "could not be loaded" in caplog.text


def test_create_app_applies_the_pixel_limit_to_pillow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)

    create_app(Settings(max_image_pixels=1234))

    assert Image.MAX_IMAGE_PIXELS == 1234


def test_create_app_uses_default_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    monkeypatch.delenv("WASTEVISION_RATE_LIMIT_PER_MINUTE", raising=False)

    app = create_app()

    assert app.state.settings == Settings()
    assert app.state.rate_limiter is None


def test_openapi_documents_the_classify_endpoint(client: TestClient):
    paths = client.get("/openapi.json").json()["paths"]

    assert "post" in paths[CLASSIFY_URL]
    assert {"/health/live", "/health/ready"} <= set(paths)
