import csv
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from wastevision.labels import CLASS_NAMES

Batches = Iterable[tuple[torch.Tensor, torch.Tensor]]

CHECKPOINT_FILE_NAME = "best.pt"
METRICS_FILE_NAME = "metrics.csv"
METRICS_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "seconds",
)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train: EpochMetrics
    validation: EpochMetrics
    seconds: float


@dataclass(frozen=True)
class FitResult:
    history: list[EpochRecord]
    best_epoch: int
    best_validation: EpochMetrics
    checkpoint_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def inverse_frequency_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError(f"Every one of the {num_classes} classes needs at least one label")
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def set_train_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and module.affine and not module.weight.requires_grad:
            module.eval()


def compute_metrics(total_loss: float, labels: list[int], predictions: list[int]) -> EpochMetrics:
    if not labels:
        raise ValueError("Cannot compute metrics for an empty data loader")
    return EpochMetrics(
        loss=total_loss / len(labels),
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(
            f1_score(
                labels,
                predictions,
                labels=list(range(len(CLASS_NAMES))),
                average="macro",
                zero_division=0,
            )
        ),
    )


def train_epoch(
    model: nn.Module,
    batches: Batches,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> EpochMetrics:
    set_train_mode(model)
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for images, targets in batches:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        labels.extend(targets.tolist())
        predictions.extend(logits.argmax(dim=1).tolist())
    return compute_metrics(total_loss, labels, predictions)


def evaluate_epoch(model: nn.Module, batches: Batches, criterion: nn.Module) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, targets in batches:
            logits = model(images)
            total_loss += criterion(logits, targets).item() * len(targets)
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).tolist())
    return compute_metrics(total_loss, labels, predictions)


def save_checkpoint(model: nn.Module, path: Path, epoch: int, validation: EpochMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "class_names": list(CLASS_NAMES),
        "validation": asdict(validation),
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint {path} was trained for other classes")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def write_metrics_row(path: Path, record: EpochRecord) -> None:
    is_new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        if is_new_file:
            writer.writerow(METRICS_COLUMNS)
        writer.writerow(
            [
                record.epoch,
                f"{record.train.loss:.6f}",
                f"{record.train.accuracy:.6f}",
                f"{record.train.macro_f1:.6f}",
                f"{record.validation.loss:.6f}",
                f"{record.validation.accuracy:.6f}",
                f"{record.validation.macro_f1:.6f}",
                f"{record.seconds:.1f}",
            ]
        )


def fit(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    run_dir: Path,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    class_weights: torch.Tensor | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    patience: int | None = None,
    report: Callable[[EpochRecord], None] | None = None,
) -> FitResult:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience is not None and patience < 1:
        raise ValueError("patience must be at least 1")
    metrics_path = run_dir / METRICS_FILE_NAME
    if metrics_path.exists():
        raise FileExistsError(f"Run folder {run_dir} already has {METRICS_FILE_NAME}")
    run_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    checkpoint_path = run_dir / CHECKPOINT_FILE_NAME
    history: list[EpochRecord] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train = train_epoch(model, train_loader, criterion, optimizer)
        validation = evaluate_epoch(model, validation_loader, criterion)
        record = EpochRecord(epoch, train, validation, time.perf_counter() - start)
        history.append(record)
        write_metrics_row(metrics_path, record)
        if report is not None:
            report(record)
        if validation.macro_f1 > best_macro_f1:
            best_macro_f1 = validation.macro_f1
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_path, epoch, validation)
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if patience is not None and epochs_without_improvement >= patience:
            break

    best = max(history, key=lambda record: record.validation.macro_f1)
    return FitResult(history, best.epoch, best.validation, checkpoint_path)
