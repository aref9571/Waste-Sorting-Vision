import struct
import zlib
from collections.abc import Callable, Iterator
from http import HTTPStatus
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


def png_header_only(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload)
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 16))
        + chunk(b"IEND", b"")
    )


def assert_problem(response, status: int) -> dict:
    body = response.json()
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert set(body) == {"type", "title", "status", "detail", "instance"}
    assert body["type"] == "about:blank"
    assert body["title"] == HTTPStatus(status).phrase
    assert body["status"] == status
    return body


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


class BrokenPredictor:
    def predict(self, image: Image.Image):
        raise RuntimeError("internal failure with SECRET details")


def test_upload_over_the_size_limit_is_rejected(make_client: ClientFactory):
    client = make_client(max_upload_bytes=1000)

    response = upload(client, encoded_image())

    body = assert_problem(response, 413)
    assert body["detail"] == "The image must not be larger than 1000 bytes."
    assert body["instance"] == CLASSIFY_URL


def test_upload_over_the_default_size_limit_is_rejected(client: TestClient):
    response = upload(client, b"\x00" * (5 * 1024 * 1024 + 1))

    assert_problem(response, 413)


def test_upload_at_the_size_limit_is_accepted(make_client: ClientFactory):
    data = encoded_image(size=(32, 32))
    client = make_client(max_upload_bytes=len(data))

    assert upload(client, data).status_code == 200


@pytest.mark.parametrize("content_type", ["image/gif", "text/plain", "application/octet-stream"])
def test_upload_with_a_wrong_content_type_is_rejected(client: TestClient, content_type: str):
    response = upload(client, encoded_image("GIF"), content_type)

    body = assert_problem(response, 415)
    assert body["detail"] == "Upload a JPEG, PNG or WebP image."


def test_upload_whose_content_does_not_match_its_type_is_rejected(client: TestClient):
    response = upload(client, encoded_image("PNG"), "image/jpeg")

    assert_problem(response, 415)


@pytest.mark.parametrize(
    "data",
    [b"SECRET text that is not an image", b"\x89PNG\r\n\x1a\nSECRET" + b"\x00" * 40],
)
def test_corrupt_upload_is_rejected_without_echoing_it(client: TestClient, data: bytes):
    response = upload(client, data)

    body = assert_problem(response, 400)
    assert body["detail"] == "The file could not be read as an image."
    assert "SECRET" not in response.text


def test_truncated_upload_is_rejected(client: TestClient):
    response = upload(client, encoded_image("JPEG")[:500], "image/jpeg")

    assert_problem(response, 400)


def test_empty_upload_is_rejected(client: TestClient):
    response = upload(client, b"")

    body = assert_problem(response, 400)
    assert body["detail"] == "The uploaded file is empty."


def test_upload_with_huge_dimensions_is_rejected(client: TestClient):
    response = upload(client, png_header_only(20_000, 20_000))

    assert_problem(response, 400)


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_upload_over_the_configured_pixel_limit_is_rejected(make_client: ClientFactory):
    client = make_client(max_image_pixels=10_000)

    response = upload(client, encoded_image(size=(101, 100)))

    body = assert_problem(response, 400)
    assert body["detail"] == "The image must not have more than 10000 pixels."


def test_missing_image_field_is_a_problem_without_the_content(client: TestClient):
    response = upload(client, b"SECRET payload", field="file")

    body = assert_problem(response, 422)
    assert "body.image" in body["detail"]
    assert "SECRET" not in response.text


def test_unknown_path_is_a_problem(client: TestClient):
    body = assert_problem(client.get("/nothing-here"), 404)

    assert body["instance"] == "/nothing-here"


def test_wrong_method_is_a_problem(client: TestClient):
    assert_problem(client.get(CLASSIFY_URL), 405)


def test_unexpected_error_is_a_problem_without_details(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
):
    client = make_client(load_predictor=lambda _: BrokenPredictor())

    response = upload(client, encoded_image())

    body = assert_problem(response, 500)
    assert body["detail"] == "An internal error occurred."
    assert "SECRET" not in response.text
    assert "Unexpected error" in caplog.text


def test_rate_limit_rejects_requests_over_the_limit(make_client: ClientFactory):
    client = make_client(rate_limit_per_minute=2)

    statuses = [upload(client, encoded_image()).status_code for _ in range(2)]
    response = upload(client, encoded_image())

    body = assert_problem(response, 429)
    assert statuses == [200, 200]
    assert body["detail"] == "Too many requests. Try again later."
    assert 1 <= int(response.headers["retry-after"]) <= 60


def test_rate_limit_counts_rejected_uploads(make_client: ClientFactory):
    client = make_client(rate_limit_per_minute=1)

    first = upload(client, b"not an image")
    second = upload(client, encoded_image())

    assert first.status_code == 400
    assert second.status_code == 429


def test_rate_limit_does_not_apply_to_health_checks(make_client: ClientFactory):
    client = make_client(rate_limit_per_minute=1)
    upload(client, encoded_image())

    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_no_rate_limit_by_default(client: TestClient):
    statuses = {upload(client, encoded_image(size=(64, 64))).status_code for _ in range(5)}

    assert statuses == {200}
