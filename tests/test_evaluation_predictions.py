from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from wastevision.evaluation.predictions import (
    Predictions,
    load_trained_model,
    predict_probabilities,
    predict_split,
)
from wastevision.labels import CLASS_NAMES
from wastevision.training.loop import EpochMetrics, save_checkpoint
from wastevision.training.model import build_model


def save_untrained_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    save_checkpoint(build_model(pretrained=False), path, 1, EpochMetrics(1.0, 0.5, 0.5))


def test_predictions_properties():
    predictions = Predictions(
        labels=np.array([0, 1, 2]),
        probabilities=np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]),
    )

    assert predictions.predicted.tolist() == [0, 0, 2]
    assert predictions.confidences.tolist() == pytest.approx([0.7, 0.6, 0.8])
    assert predictions.correct.tolist() == [True, False, True]


def test_predict_probabilities_keeps_the_order_and_sums_to_one():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    images = torch.randn(5, 3, 2, 2)
    targets = torch.tensor([2, 0, 1, 1, 0])
    batches = [(images[:3], targets[:3]), (images[3:], targets[3:])]

    predictions = predict_probabilities(model, batches)

    assert predictions.labels.tolist() == [2, 0, 1, 1, 0]
    assert predictions.probabilities.shape == (5, 3)
    assert predictions.probabilities.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-6)
    with torch.no_grad():
        expected = torch.softmax(model(images), dim=1).numpy()
    assert predictions.probabilities == pytest.approx(expected, abs=1e-6)


def test_predict_probabilities_puts_the_model_in_eval_mode():
    model = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(12, 3))
    model.train()

    predict_probabilities(model, [(torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))])

    assert not model.training


def test_predict_probabilities_rejects_an_empty_loader():
    with pytest.raises(ValueError, match="empty"):
        predict_probabilities(nn.Linear(2, 2), [])


def test_load_trained_model_restores_the_weights(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    save_untrained_checkpoint(checkpoint)
    torch.manual_seed(0)
    expected = build_model(pretrained=False).state_dict()

    model = load_trained_model(checkpoint)

    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_load_trained_model_rejects_a_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_trained_model(tmp_path / "missing.pt")


def test_predict_split_uses_only_the_requested_split(
    tiny_dataset: tuple[Path, Path], tmp_path: Path
):
    images_dir, split_path = tiny_dataset
    checkpoint = tmp_path / "best.pt"
    save_untrained_checkpoint(checkpoint)

    predictions = predict_split(checkpoint, images_dir, split_path, "test", batch_size=4)

    assert predictions.labels.tolist() == list(range(len(CLASS_NAMES)))
    assert predictions.probabilities.shape == (len(CLASS_NAMES), len(CLASS_NAMES))


def test_predict_split_is_deterministic(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset
    checkpoint = tmp_path / "best.pt"
    save_untrained_checkpoint(checkpoint)

    first = predict_split(checkpoint, images_dir, split_path, "validation")
    second = predict_split(checkpoint, images_dir, split_path, "validation")

    assert np.array_equal(first.probabilities, second.probabilities)
