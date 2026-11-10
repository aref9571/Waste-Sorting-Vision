from pathlib import Path

import pytest
from pydantic import ValidationError

from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE, Settings


def test_image_sizes_resize_larger_than_crop():
    assert IMAGE_SIZE == 224
    assert RESIZE_SIZE > IMAGE_SIZE


def test_normalisation_values_are_imagenet_statistics():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def test_settings_default_to_no_worker_processes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WASTEVISION_NUM_WORKERS", raising=False)

    assert Settings().num_workers == 0


def test_settings_read_num_workers_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WASTEVISION_NUM_WORKERS", "3")

    assert Settings().num_workers == 3


def test_settings_reject_negative_num_workers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WASTEVISION_NUM_WORKERS", "-1")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_default_service_limits(monkeypatch: pytest.MonkeyPatch):
    for name in ("MODELS_DIR", "MAX_UPLOAD_BYTES", "MAX_IMAGE_PIXELS", "RATE_LIMIT_PER_MINUTE"):
        monkeypatch.delenv(f"WASTEVISION_{name}", raising=False)

    settings = Settings()

    assert settings.models_dir == Path("models")
    assert settings.max_upload_bytes == 5 * 1024 * 1024
    assert settings.max_image_pixels == 40_000_000
    assert settings.rate_limit_per_minute == 0


def test_settings_read_service_limits_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WASTEVISION_MODELS_DIR", "/srv/models")
    monkeypatch.setenv("WASTEVISION_MAX_UPLOAD_BYTES", "1000")
    monkeypatch.setenv("WASTEVISION_MAX_IMAGE_PIXELS", "2000")
    monkeypatch.setenv("WASTEVISION_RATE_LIMIT_PER_MINUTE", "30")

    settings = Settings()

    assert settings.models_dir == Path("/srv/models")
    assert settings.max_upload_bytes == 1000
    assert settings.max_image_pixels == 2000
    assert settings.rate_limit_per_minute == 30


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_UPLOAD_BYTES", "0"),
        ("MAX_IMAGE_PIXELS", "0"),
        ("RATE_LIMIT_PER_MINUTE", "-1"),
    ],
)
def test_settings_reject_invalid_service_limits(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
):
    monkeypatch.setenv(f"WASTEVISION_{name}", value)

    with pytest.raises(ValidationError):
        Settings()
