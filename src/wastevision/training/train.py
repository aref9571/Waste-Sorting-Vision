from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wastevision.config import Settings
from wastevision.data.image_dataset import WasteImageDataset, make_data_loader, split_dataset
from wastevision.labels import CLASS_NAMES
from wastevision.training.loop import (
    EpochRecord,
    FitResult,
    fit,
    inverse_frequency_weights,
    load_checkpoint,
    set_seed,
)
from wastevision.training.model import (
    build_model,
    freeze_backbone,
    parameter_groups,
    trainable_parameters,
)

DEFAULT_HEAD_RUN_DIR = Path("runs/head")
DEFAULT_FULL_RUN_DIR = Path("runs/full")
HEAD_EPOCHS = 8
HEAD_LEARNING_RATE = 1e-3
FULL_EPOCHS = 25
BACKBONE_LEARNING_RATE = 1e-4
PATIENCE = 4
BATCH_SIZE = 32
SEED = 42

EpochReport = Callable[[EpochRecord], None]
Loader = DataLoader[tuple[torch.Tensor, int]]


def make_loaders(
    images_dir: Path, split_path: Path, batch_size: int, seed: int, settings: Settings | None
) -> tuple[Loader, Loader]:
    train_loader = make_data_loader(
        split_dataset(split_path, images_dir, "train"),
        batch_size,
        shuffle=True,
        settings=settings,
        seed=seed,
    )
    validation_loader = make_data_loader(
        split_dataset(split_path, images_dir, "validation"),
        batch_size,
        shuffle=False,
        settings=settings,
        seed=seed,
    )
    return train_loader, validation_loader


def loss_weights(dataset: WasteImageDataset, use_class_weights: bool) -> torch.Tensor | None:
    if not use_class_weights:
        return None
    labels = [CLASS_NAMES.index(record.class_name) for record in dataset.records]
    return inverse_frequency_weights(labels, len(CLASS_NAMES))


def train_head(
    images_dir: Path,
    split_path: Path,
    run_dir: Path = DEFAULT_HEAD_RUN_DIR,
    epochs: int = HEAD_EPOCHS,
    learning_rate: float = HEAD_LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    use_class_weights: bool = False,
    pretrained: bool = True,
    settings: Settings | None = None,
    report: EpochReport | None = None,
) -> FitResult:
    set_seed(seed)
    model = build_model(pretrained=pretrained)
    freeze_backbone(model)
    train_loader, validation_loader = make_loaders(
        images_dir, split_path, batch_size, seed, settings
    )
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=learning_rate)
    return fit(
        model,
        train_loader,
        validation_loader,
        run_dir,
        optimizer,
        epochs,
        class_weights=loss_weights(train_loader.dataset, use_class_weights),
        report=report,
    )


def train_full(
    images_dir: Path,
    split_path: Path,
    head_checkpoint: Path,
    run_dir: Path = DEFAULT_FULL_RUN_DIR,
    epochs: int = FULL_EPOCHS,
    head_learning_rate: float = HEAD_LEARNING_RATE,
    backbone_learning_rate: float = BACKBONE_LEARNING_RATE,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    use_class_weights: bool = False,
    settings: Settings | None = None,
    report: EpochReport | None = None,
) -> FitResult:
    if not head_checkpoint.is_file():
        raise FileNotFoundError(f"Head checkpoint {head_checkpoint} does not exist")
    set_seed(seed)
    model = build_model(pretrained=False)
    load_checkpoint(model, head_checkpoint)
    train_loader, validation_loader = make_loaders(
        images_dir, split_path, batch_size, seed, settings
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(model, head_learning_rate, backbone_learning_rate)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return fit(
        model,
        train_loader,
        validation_loader,
        run_dir,
        optimizer,
        epochs,
        class_weights=loss_weights(train_loader.dataset, use_class_weights),
        scheduler=scheduler,
        patience=patience,
        report=report,
    )
