import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")

import onnx
from torch import nn

from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE
from wastevision.data.download import DATASET_SHA256, sha256_of_file
from wastevision.labels import CATEGORY_BY_CLASS, CLASS_NAMES
from wastevision.training import export
from wastevision.training.export import (
    PARITY_TOLERANCE,
    ParityError,
    ParityResult,
    check_parity,
    export_model,
    export_onnx,
    labels_document,
    onnx_logits,
)
from wastevision.training.loop import EpochMetrics, save_checkpoint
from wastevision.training.model import build_model


def untrained_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return build_model(pretrained=False).eval()


def random_images(count: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(count, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[nn.Module, Path]:
    model = untrained_model()
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx(model, path)
    return model, path


@pytest.mark.parametrize("batch_size", [1, 3])
def test_exported_model_matches_pytorch_for_any_batch_size(
    exported_model: tuple[nn.Module, Path], batch_size: int
):
    model, path = exported_model
    images = random_images(batch_size)

    logits = onnx_logits(path, images.numpy())

    with torch.no_grad():
        expected = model(images).numpy()
    assert logits.shape == (batch_size, len(CLASS_NAMES))
    assert np.abs(logits - expected).max() < PARITY_TOLERANCE


def test_export_removes_debug_metadata(exported_model: tuple[nn.Module, Path]):
    _, path = exported_model

    exported = onnx.load(path)

    assert exported.doc_string == ""
    assert all(node.doc_string == "" for node in exported.graph.node)
    assert all(len(node.metadata_props) == 0 for node in exported.graph.node)
    assert b"stack_trace" not in path.read_bytes()


def test_export_creates_the_parent_folder(tmp_path: Path):
    path = tmp_path / "models" / "model.onnx"

    export_onnx(untrained_model(), path)

    assert path.is_file()


def test_check_parity_passes_for_the_exported_model(exported_model: tuple[nn.Module, Path]):
    model, path = exported_model

    parity = check_parity(model, path, random_images(4))

    assert parity.images == 4
    assert parity.top1_matches == 4
    assert parity.max_abs_difference < PARITY_TOLERANCE
    assert parity.passed


def test_check_parity_fails_for_a_different_model(exported_model: tuple[nn.Module, Path]):
    _, path = exported_model

    parity = check_parity(untrained_model(seed=1), path, random_images(4))

    assert parity.max_abs_difference > PARITY_TOLERANCE
    assert not parity.passed


def test_check_parity_rejects_no_images(exported_model: tuple[nn.Module, Path]):
    model, path = exported_model

    with pytest.raises(ValueError, match="without any images"):
        check_parity(model, path, torch.zeros(0, 3, IMAGE_SIZE, IMAGE_SIZE))


@pytest.mark.parametrize(
    ("difference", "matches", "passed"),
    [(1e-6, 5, True), (PARITY_TOLERANCE, 5, False), (1e-6, 4, False)],
)
def test_parity_result_needs_small_differences_and_equal_top1(
    difference: float, matches: int, passed: bool
):
    assert ParityResult(5, difference, matches).passed is passed


def test_labels_document_keeps_the_class_order():
    document = labels_document()

    assert [entry["index"] for entry in document] == list(range(len(CLASS_NAMES)))
    assert [entry["class"] for entry in document] == list(CLASS_NAMES)
    assert [entry["category"] for entry in document] == [
        CATEGORY_BY_CLASS[name] for name in CLASS_NAMES
    ]


def save_untrained_checkpoint(path: Path) -> None:
    save_checkpoint(untrained_model(), path, 3, EpochMetrics(1.0, 0.5, 0.25))


def test_export_model_writes_the_model_labels_and_metadata(
    tiny_dataset: tuple[Path, Path], tmp_path: Path
):
    images_dir, split_path = tiny_dataset
    checkpoint = tmp_path / "best.pt"
    save_untrained_checkpoint(checkpoint)
    models_dir = tmp_path / "models"

    result = export_model(checkpoint, images_dir, split_path, models_dir, date(2026, 11, 7))

    assert result.model_path == models_dir / "waste-mobilenetv3.onnx"
    assert result.parity.passed
    assert result.parity.images == len(CLASS_NAMES)
    labels = json.loads(result.labels_path.read_text(encoding="utf-8"))
    assert labels == labels_document()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata == result.metadata
    assert metadata["model_file"] == "waste-mobilenetv3.onnx"
    assert metadata["model_sha256"] == sha256_of_file(result.model_path)
    assert metadata["opset"] >= 17
    assert metadata["input"]["resize"] == RESIZE_SIZE
    assert metadata["input"]["crop"] == IMAGE_SIZE
    assert metadata["input"]["mean"] == list(IMAGENET_MEAN)
    assert metadata["input"]["std"] == list(IMAGENET_STD)
    assert metadata["class_names"] == list(CLASS_NAMES)
    assert metadata["dataset_sha256"] == DATASET_SHA256
    assert metadata["split_sha256"] == sha256_of_file(split_path)
    assert metadata["checkpoint_sha256"] == sha256_of_file(checkpoint)
    assert metadata["checkpoint_epoch"] == 3
    assert metadata["metrics"]["validation"] == {"accuracy": 0.5, "macro_f1": 0.25}
    assert metadata["metrics"]["test"]["images"] == len(CLASS_NAMES)
    assert 0.0 <= metadata["metrics"]["test"]["macro_f1"] <= 1.0
    assert metadata["parity"]["top1_matches"] == len(CLASS_NAMES)
    assert metadata["exported_on"] == "2026-11-07"


def test_export_model_rejects_a_missing_checkpoint(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset

    with pytest.raises(FileNotFoundError, match="does not exist"):
        export_model(tmp_path / "missing.pt", images_dir, split_path, tmp_path / "models")

    assert not (tmp_path / "models").exists()


def test_export_model_removes_the_model_when_parity_fails(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    images_dir, split_path = tiny_dataset
    checkpoint = tmp_path / "best.pt"
    save_untrained_checkpoint(checkpoint)
    models_dir = tmp_path / "models"
    monkeypatch.setattr(
        export, "onnx_logits", lambda path, images: np.zeros((len(images), len(CLASS_NAMES)))
    )

    with pytest.raises(ParityError, match="differs from PyTorch"):
        export_model(checkpoint, images_dir, split_path, models_dir)

    assert list(models_dir.iterdir()) == []
