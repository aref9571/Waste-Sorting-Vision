from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wastevision.config import Settings
from wastevision.inference.predictor import Prediction, Predictor, load_metadata
from wastevision.labels import CATEGORY_BY_CLASS, CLASS_NAMES, SORTING_HINT_BY_CLASS
from wastevision.web.app import create_app

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PAGE_URL = "/"

ClientFactory = Callable[..., TestClient]


class StubPredictor:
    def __init__(self, prediction: Prediction) -> None:
        self.prediction = prediction

    def predict(self, image: Image.Image) -> Prediction:
        return self.prediction


class TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


def tags_of(html: str, name: str) -> list[dict[str, str | None]]:
    collector = TagCollector()
    collector.feed(html)
    return [attributes for tag, attributes in collector.tags if tag == name]


def encoded_image(image_format: str = "PNG") -> bytes:
    pixels = np.random.default_rng(3).integers(0, 256, (240, 320, 3), np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format=image_format)
    return buffer.getvalue()


def prediction_for(label: str, confidence: float, unsure: bool) -> Prediction:
    rest = (1.0 - confidence) / 2
    others = [name for name in CLASS_NAMES if name != label][:2]
    return Prediction(
        top=((label, confidence), (others[0], rest), (others[1], rest)),
        category=CATEGORY_BY_CLASS[label],
        unsure=unsure,
    )


def post_photo(client: TestClient, data: bytes | None = None, content_type: str = "image/png"):
    files = {"image": ("photo.png", data if data is not None else encoded_image(), content_type)}
    return client.post(PAGE_URL, files=files)


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[ClientFactory]:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    open_clients: list[TestClient] = []

    def make(prediction: Prediction | None = None, **settings_values) -> TestClient:
        stub = StubPredictor(prediction or prediction_for("glass", 0.9, False))
        app = create_app(Settings(**settings_values), load_predictor=lambda _: stub)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        open_clients.append(client)
        return client

    yield make
    for client in open_clients:
        client.__exit__(None, None, None)


def test_index_renders_the_upload_form(make_client: ClientFactory):
    response = make_client().get(PAGE_URL)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    (form,) = tags_of(response.text, "form")
    assert form["method"] == "post"
    assert form["enctype"] == "multipart/form-data"
    assert form["action"] == "/"
    assert "Classify" in response.text


def test_the_file_field_accepts_images_and_has_a_label(make_client: ClientFactory):
    html = make_client().get(PAGE_URL).text

    (field,) = tags_of(html, "input")
    assert field["type"] == "file"
    assert field["name"] == "image"
    assert field["accept"] == "image/*"
    assert 'for="image"' in html
    assert field["id"] == "image"


def test_the_page_declares_its_language_and_shows_the_notes(make_client: ClientFactory):
    html = make_client().get(PAGE_URL).text

    assert '<html lang="en">' in html
    assert "Recycling rules differ between cities" in html
    assert "up to 5 MB" in html


def test_the_page_shows_a_configured_size_limit(make_client: ClientFactory):
    html = make_client(max_upload_bytes=2 * 1024 * 1024).get(PAGE_URL).text

    assert "up to 2 MB" in html


def test_the_preview_image_has_alt_text_and_starts_hidden(make_client: ClientFactory):
    html = make_client().get(PAGE_URL).text

    (preview,) = tags_of(html, "img")
    assert preview["alt"]
    assert "hidden" in preview


def test_the_page_has_no_result_or_error_before_a_post(make_client: ClientFactory):
    html = make_client().get(PAGE_URL).text

    assert 'role="alert"' not in html
    assert "Top three guesses" not in html


def test_the_page_has_no_inline_script(make_client: ClientFactory):
    html = make_client().get(PAGE_URL).text

    scripts = tags_of(html, "script")
    assert [script["src"] for script in scripts] == ["http://testserver/static/preview.js"]
    assert "onchange" not in html


def test_static_files_are_served(make_client: ClientFactory):
    client = make_client()

    style = client.get("/static/style.css")
    script = client.get("/static/preview.js")

    assert style.status_code == 200
    assert style.headers["content-type"].startswith("text/css")
    assert script.status_code == 200
    assert "createObjectURL" in script.text


def test_posting_a_photo_shows_the_predicted_category(make_client: ClientFactory):
    client = make_client(prediction_for("glass", 0.9, False))

    response = post_photo(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Glass" in response.text
    assert "Confidence: 90%" in response.text
    assert SORTING_HINT_BY_CLASS["glass"] in response.text
    assert "I am not sure" not in response.text


@pytest.mark.parametrize("label", CLASS_NAMES)
def test_every_class_shows_its_category_and_hint(make_client: ClientFactory, label: str):
    response = post_photo(make_client(prediction_for(label, 0.8, False)))

    assert CATEGORY_BY_CLASS[label] in response.text
    assert SORTING_HINT_BY_CLASS[label] in response.text


def test_the_result_lists_three_guesses_with_meters(make_client: ClientFactory):
    html = post_photo(make_client(prediction_for("metal", 0.6, False))).text

    assert "Top three guesses" in html
    assert html.count("<li>") == 3
    values = [meter["value"] for meter in tags_of(html, "meter")]
    assert values == ["0.6", "0.6", "0.2", "0.2"]
    assert "60%" in html
    assert "20%" in html


def test_the_confidence_meter_is_labelled(make_client: ClientFactory):
    html = post_photo(make_client(prediction_for("paper", 0.75, False))).text

    meters = tags_of(html, "meter")
    assert meters[0]["id"] == "confidence"
    assert 'for="confidence"' in html
    assert all(meter.get("aria-label") for meter in meters[1:])


def test_an_unsure_result_shows_the_unsure_message_and_no_hint(make_client: ClientFactory):
    client = make_client(prediction_for("plastic", 0.3, True))

    html = post_photo(client).text

    assert "I am not sure about this one." in html
    assert "Best guess:" in html
    assert "Plastic" in html
    assert "30%" in html
    assert SORTING_HINT_BY_CLASS["plastic"] not in html
    assert 'id="confidence"' not in html


def test_the_form_is_shown_again_after_a_result(make_client: ClientFactory):
    html = post_photo(make_client()).text

    assert len(tags_of(html, "form")) == 1


@pytest.mark.parametrize(
    ("image_format", "content_type"), [("JPEG", "image/jpeg"), ("WEBP", "image/webp")]
)
def test_other_image_types_are_accepted(
    make_client: ClientFactory, image_format: str, content_type: str
):
    response = post_photo(make_client(), encoded_image(image_format), content_type)

    assert response.status_code == 200
    assert "Glass" in response.text


def test_the_page_works_with_the_committed_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    predictor = Predictor(
        MODELS_DIR / "waste-mobilenetv3.onnx", load_metadata(MODELS_DIR / "metadata.json")
    )
    data = encoded_image()
    expected = predictor.predict(Image.open(BytesIO(data)))
    app = create_app(Settings(models_dir=MODELS_DIR), load_predictor=lambda _: predictor)

    with TestClient(app) as client:
        response = post_photo(client, data)

    assert response.status_code == 200
    assert expected.category in response.text


def test_posting_without_a_file_gives_a_form_error(make_client: ClientFactory):
    response = make_client().post(PAGE_URL, data={"other": "value"})

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("text/html")
    assert 'role="alert"' in response.text
    assert "Choose an image" in response.text


def test_posting_an_empty_file_name_gives_a_form_error(make_client: ClientFactory):
    response = make_client().post(PAGE_URL, files={"image": ("", b"", "application/octet-stream")})

    assert response.status_code == 400
    assert "Choose an image" in response.text


@pytest.mark.parametrize(
    ("data", "content_type", "status"),
    [
        (b"plain text", "text/plain", 415),
        (b"not an image", "image/png", 400),
        (b"", "image/png", 400),
        (encoded_image("PNG"), "image/jpeg", 415),
    ],
)
def test_rejected_uploads_render_an_error_page(
    make_client: ClientFactory, data: bytes, content_type: str, status: int
):
    response = post_photo(make_client(), data, content_type)

    assert response.status_code == status
    assert response.headers["content-type"].startswith("text/html")
    assert 'role="alert"' in response.text
    assert "could not be classified" in response.text
    assert "Top three guesses" not in response.text


def test_a_too_large_upload_renders_an_error_page(make_client: ClientFactory):
    response = post_photo(make_client(max_upload_bytes=1000), encoded_image())

    assert response.status_code == 413
    assert "must not be larger than 1000 bytes" in response.text


def test_an_error_page_never_echoes_the_upload(make_client: ClientFactory):
    secret = b"SECRET-CONTENT-12345"

    response = post_photo(make_client(), secret, "image/png")

    assert "SECRET-CONTENT-12345" not in response.text
    assert "photo.png" not in response.text


def test_the_error_page_keeps_the_form(make_client: ClientFactory):
    response = post_photo(make_client(), b"x", "text/plain")

    assert len(tags_of(response.text, "form")) == 1


def test_the_page_returns_503_when_the_model_is_not_loaded(make_client: ClientFactory):
    app = create_app(Settings(models_dir=Path("missing")))
    with TestClient(app) as client:
        response = post_photo(client)

    assert response.status_code == 503
    assert "The model is not loaded." in response.text


def test_the_page_is_rate_limited_with_retry_after(make_client: ClientFactory):
    client = make_client(rate_limit_per_minute=1)

    first = post_photo(client)
    second = post_photo(client)

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) > 0
    assert "Too many requests" in second.text
    assert second.headers["content-type"].startswith("text/html")


def test_viewing_the_page_is_not_rate_limited(make_client: ClientFactory):
    client = make_client(rate_limit_per_minute=1)

    statuses = [client.get(PAGE_URL).status_code for _ in range(3)]

    assert statuses == [200, 200, 200]


def test_the_api_and_the_page_still_work_side_by_side(make_client: ClientFactory):
    client = make_client()

    api = client.post("/api/v1/classify", files={"image": ("p", encoded_image(), "image/png")})

    assert api.status_code == 200
    assert api.json()["category"] == "Glass"
    assert client.get("/health/live").status_code == 200
