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
