import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wastevision.labels import CLASS_NAMES
from wastevision.training.loop import (
    CHECKPOINT_FILE_NAME,
    METRICS_COLUMNS,
    METRICS_FILE_NAME,
    EpochMetrics,
    compute_metrics,
    evaluate_epoch,
    fit,
    inverse_frequency_weights,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    set_train_mode,
    train_epoch,
)
from wastevision.training.model import build_model, freeze_backbone, trainable_parameters

IMAGE_SIZE = 32


def random_batch(seed: int, size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(size, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    labels = torch.arange(size) % len(CLASS_NAMES)
    return images, labels


def random_loader(seed: int, size: int = 12) -> DataLoader:
    images, labels = random_batch(seed, size)
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def adamw(model: nn.Module, learning_rate: float = 1e-3) -> torch.optim.AdamW:
    return torch.optim.AdamW(trainable_parameters(model), lr=learning_rate)


def batch_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(batch[0]), batch[1]).item()


def test_one_training_step_reduces_loss_on_the_same_batch():
    set_seed(0)
    model = build_model(pretrained=False)
    batch = random_batch(seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_before = batch_loss(model, batch)

    train_epoch(model, [batch], nn.CrossEntropyLoss(), optimizer)

    assert batch_loss(model, batch) < loss_before


def test_training_with_frozen_backbone_changes_only_the_classifier():
    set_seed(0)
    model = build_model(pretrained=False)
    freeze_backbone(model)
    features_before = {key: value.clone() for key, value in model.features.state_dict().items()}
    classifier_before = [parameter.clone() for parameter in model.classifier.parameters()]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    train_epoch(model, [random_batch(seed=1)], nn.CrossEntropyLoss(), optimizer)

    for key, value in model.features.state_dict().items():
        assert torch.equal(value, features_before[key]), key
    assert any(
        not torch.equal(before, after)
        for before, after in zip(classifier_before, model.classifier.parameters(), strict=True)
    )


def test_set_train_mode_keeps_frozen_batch_norm_in_evaluation_mode():
    model = build_model(pretrained=False)
    freeze_backbone(model)

    set_train_mode(model)

    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    assert batch_norms
    assert not any(module.training for module in batch_norms)
    assert model.training
    assert model.classifier.training


def test_set_train_mode_trains_batch_norm_that_is_not_frozen():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.BatchNorm2d(4, affine=False))
    model.eval()

    set_train_mode(model)

    assert all(module.training for module in model.modules())


def test_evaluate_epoch_does_not_change_the_model():
    set_seed(0)
    model = build_model(pretrained=False)
    state_before = {key: value.clone() for key, value in model.state_dict().items()}

    metrics = evaluate_epoch(model, random_loader(seed=2), nn.CrossEntropyLoss())

    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key]), key
    assert metrics.loss > 0
    assert 0 <= metrics.accuracy <= 1


def test_compute_metrics_by_hand():
    metrics = compute_metrics(total_loss=2.0, labels=[0, 0, 1, 1], predictions=[0, 1, 1, 1])

    assert metrics.loss == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.75)
    expected_f1_class_0 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
    expected_f1_class_1 = 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0)
    expected_macro_f1 = (expected_f1_class_0 + expected_f1_class_1) / len(CLASS_NAMES)
    assert metrics.macro_f1 == pytest.approx(expected_macro_f1)


def test_compute_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(total_loss=0.0, labels=[], predictions=[])


def test_inverse_frequency_weights():
    weights = inverse_frequency_weights([0, 0, 1, 2], num_classes=3)

    assert weights.dtype == torch.float32
    assert weights.tolist() == pytest.approx([4 / 6, 4 / 3, 4 / 3])


@pytest.mark.parametrize("labels", [[0, 0, 1], [0, 1, 2, 3]])
def test_inverse_frequency_weights_rejects_missing_or_unknown_classes(labels: list[int]):
    with pytest.raises(ValueError, match="at least one label"):
        inverse_frequency_weights(labels, num_classes=3)


def test_checkpoint_round_trip(tmp_path: Path):
    set_seed(0)
    model = build_model(pretrained=False)
    path = tmp_path / "nested" / "model.pt"
    validation = EpochMetrics(loss=0.4, accuracy=0.8, macro_f1=0.75)

    save_checkpoint(model, path, epoch=3, validation=validation)
    set_seed(1)
    restored = build_model(pretrained=False)
    checkpoint = load_checkpoint(restored, path)

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    assert checkpoint["epoch"] == 3
    assert checkpoint["validation"] == {"loss": 0.4, "accuracy": 0.8, "macro_f1": 0.75}


def test_load_checkpoint_rejects_other_class_names(tmp_path: Path):
    model = build_model(pretrained=False)
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "class_names": ["a", "b"]}, path)

    with pytest.raises(ValueError, match="other classes"):
        load_checkpoint(model, path)


def test_fit_writes_metrics_and_keeps_the_best_checkpoint(tmp_path: Path):
    set_seed(0)
    model = build_model(pretrained=False)
    run_dir = tmp_path / "run"

    result = fit(model, random_loader(1), random_loader(2), run_dir, adamw(model), epochs=3)

    with (run_dir / METRICS_FILE_NAME).open(encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert tuple(rows[0]) == METRICS_COLUMNS
    assert [int(row[0]) for row in rows[1:]] == [1, 2, 3]
    assert len(result.history) == 3
    best_f1 = max(record.validation.macro_f1 for record in result.history)
    assert result.best_validation.macro_f1 == best_f1
    assert result.history[result.best_epoch - 1].validation == result.best_validation
    assert result.checkpoint_path == run_dir / CHECKPOINT_FILE_NAME
    checkpoint = load_checkpoint(build_model(pretrained=False), result.checkpoint_path)
    assert checkpoint["epoch"] == result.best_epoch


def test_fit_keeps_the_first_epoch_when_later_epochs_are_not_better(tmp_path: Path):
    set_seed(0)
    model = build_model(pretrained=False)
    freeze_backbone(model)

    result = fit(model, random_loader(1), random_loader(2), tmp_path, adamw(model, 0.0), epochs=2)

    assert result.history[0].validation.macro_f1 == result.history[1].validation.macro_f1
    assert result.best_epoch == 1


def test_fit_with_class_weights(tmp_path: Path):
    set_seed(0)
    model = build_model(pretrained=False)
    weights = torch.ones(len(CLASS_NAMES))

    result = fit(model, random_loader(1), random_loader(2), tmp_path, adamw(model), 1, weights)

    assert result.best_epoch == 1


def test_fit_is_reproducible_with_the_same_seed(tmp_path: Path):
    results = []
    for name in ("first", "second"):
        set_seed(7)
        model = build_model(pretrained=False)
        results.append(
            fit(model, random_loader(1), random_loader(2), tmp_path / name, adamw(model), 2)
        )

    first, second = ([record.validation for record in r.history] for r in results)
    assert first == second


def test_fit_rejects_zero_epochs(tmp_path: Path):
    model = build_model(pretrained=False)

    with pytest.raises(ValueError, match="at least 1"):
        fit(model, [], [], tmp_path, adamw(model), epochs=0)


def test_fit_refuses_to_overwrite_a_run(tmp_path: Path):
    (tmp_path / METRICS_FILE_NAME).write_text("old\n", encoding="utf-8")

    model = build_model(pretrained=False)

    with pytest.raises(FileExistsError):
        fit(model, [], [], tmp_path, adamw(model), epochs=1)

    assert (tmp_path / METRICS_FILE_NAME).read_text(encoding="utf-8") == "old\n"


def test_fit_stops_early_when_validation_does_not_improve(tmp_path: Path):
    set_seed(0)
    model = build_model(pretrained=False)
    reported_epochs = []

    result = fit(
        model,
        random_loader(1),
        random_loader(2),
        tmp_path,
        adamw(model, 0.0),
        epochs=10,
        patience=2,
        report=lambda record: reported_epochs.append(record.epoch),
    )

    assert [record.epoch for record in result.history] == [1, 2, 3]
    assert reported_epochs == [1, 2, 3]
    assert result.best_epoch == 1


def test_fit_without_patience_runs_every_epoch(tmp_path: Path):
    set_seed(0)
    model = build_model(pretrained=False)

    result = fit(model, random_loader(1), random_loader(2), tmp_path, adamw(model, 0.0), 4)

    assert len(result.history) == 4


def test_fit_steps_the_scheduler_once_per_epoch(tmp_path: Path):
    set_seed(0)
    model = build_model(pretrained=False)
    optimizer = adamw(model, 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    fit(model, random_loader(1), random_loader(2), tmp_path, optimizer, 2, scheduler=scheduler)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5e-3)


def test_fit_rejects_zero_patience(tmp_path: Path):
    model = build_model(pretrained=False)

    with pytest.raises(ValueError, match="patience"):
        fit(model, [], [], tmp_path, adamw(model), epochs=1, patience=0)
