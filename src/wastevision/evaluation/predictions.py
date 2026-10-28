from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wastevision.config import Settings
from wastevision.data.image_dataset import make_data_loader, split_dataset
from wastevision.training.loop import Batches, load_checkpoint
from wastevision.training.model import build_model

BATCH_SIZE = 32


@dataclass(frozen=True)
class Predictions:
    labels: np.ndarray
    probabilities: np.ndarray

    @property
    def predicted(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)

    @property
    def confidences(self) -> np.ndarray:
        return self.probabilities.max(axis=1)

    @property
    def correct(self) -> np.ndarray:
        return self.predicted == self.labels


def predict_probabilities(model: nn.Module, batches: Batches) -> Predictions:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images, targets in batches:
            probabilities.append(torch.softmax(model(images), dim=1).numpy())
            labels.append(targets.numpy())
    if not labels:
        raise ValueError("Cannot predict on an empty data loader")
    return Predictions(np.concatenate(labels), np.concatenate(probabilities))


def load_trained_model(checkpoint_path: Path) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
    model = build_model(pretrained=False)
    load_checkpoint(model, checkpoint_path)
    return model


def predict_split(
    checkpoint_path: Path,
    images_dir: Path,
    split_path: Path,
    split: str = "test",
    batch_size: int = BATCH_SIZE,
    settings: Settings | None = None,
) -> Predictions:
    model = load_trained_model(checkpoint_path)
    dataset = split_dataset(split_path, images_dir, split)
    loader = make_data_loader(dataset, batch_size, shuffle=False, settings=settings)
    return predict_probabilities(model, loader)
