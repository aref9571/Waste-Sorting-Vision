from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from wastevision.training.loop import FitResult, load_checkpoint
from wastevision.training.model import build_model
from wastevision.training.train import train_full, train_head


def capture_fit_arguments(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured = {}

    def fake_fit(model, train_loader, validation_loader, run_dir, optimizer, epochs, **options):
        captured.update(model=model, optimizer=optimizer, epochs=epochs, **options)
        return FitResult([], 0, None, run_dir / "best.pt")

    monkeypatch.setattr("wastevision.training.train.fit", fake_fit)
    return captured


@pytest.fixture
def head_checkpoint(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> Path:
    images_dir, split_path = tiny_dataset
    result = train_head(
        images_dir, split_path, tmp_path / "head", epochs=1, batch_size=8, pretrained=False
    )
    return result.checkpoint_path


def test_train_head_on_a_tiny_dataset(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset
    run_dir = tmp_path / "run"

    result = train_head(images_dir, split_path, run_dir, epochs=2, batch_size=8, pretrained=False)

    assert [record.epoch for record in result.history] == [1, 2]
    assert (run_dir / "metrics.csv").exists()
    checkpoint = load_checkpoint(build_model(pretrained=False), result.checkpoint_path)
    assert checkpoint["epoch"] == result.best_epoch


def test_train_head_does_not_change_the_backbone(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    images_dir, split_path = tiny_dataset
    built_models = []

    def build_and_remember(pretrained: bool):
        model = build_model(pretrained=pretrained)
        built_models.append((model, {k: v.clone() for k, v in model.features.state_dict().items()}))
        return model

    monkeypatch.setattr("wastevision.training.train.build_model", build_and_remember)
    train_head(images_dir, split_path, tmp_path, epochs=1, batch_size=8, pretrained=False)

    model, features_before = built_models[0]
    for key, value in model.features.state_dict().items():
        assert torch.equal(value, features_before[key]), key


@pytest.mark.parametrize("use_class_weights", [False, True])
def test_train_head_uses_class_weights_only_when_asked(
    tiny_dataset: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_class_weights: bool,
):
    images_dir, split_path = tiny_dataset
    captured = capture_fit_arguments(monkeypatch)

    train_head(
        images_dir, split_path, tmp_path, pretrained=False, use_class_weights=use_class_weights
    )

    if use_class_weights:
        assert captured["class_weights"].tolist() == pytest.approx([1.0] * 6)
    else:
        assert captured["class_weights"] is None


def test_train_full_fine_tunes_every_layer(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, head_checkpoint: Path
):
    images_dir, split_path = tiny_dataset
    start = build_model(pretrained=False)
    load_checkpoint(start, head_checkpoint)

    result = train_full(
        images_dir, split_path, head_checkpoint, tmp_path / "full", epochs=2, batch_size=8
    )

    assert [record.epoch for record in result.history] == [1, 2]
    trained = build_model(pretrained=False)
    load_checkpoint(trained, result.checkpoint_path)
    assert not torch.equal(trained.features[0][0].weight, start.features[0][0].weight)
    assert not torch.equal(trained.classifier[-1].weight, start.classifier[-1].weight)


def test_train_full_starts_from_the_head_with_two_learning_rates(
    tiny_dataset: tuple[Path, Path],
    tmp_path: Path,
    head_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    images_dir, split_path = tiny_dataset
    expected = build_model(pretrained=False)
    load_checkpoint(expected, head_checkpoint)
    captured = capture_fit_arguments(monkeypatch)

    train_full(images_dir, split_path, head_checkpoint, tmp_path / "full", epochs=10)

    for key, value in expected.state_dict().items():
        assert torch.equal(captured["model"].state_dict()[key], value), key
    assert all(parameter.requires_grad for parameter in captured["model"].parameters())
    assert [group["lr"] for group in captured["optimizer"].param_groups] == [1e-4, 1e-3]
    scheduler = captured["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 10
    assert captured["epochs"] == 10
    assert captured["patience"] == 4
    assert captured["class_weights"].tolist() == pytest.approx([1.0] * 6)


def test_train_full_needs_a_head_checkpoint(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset

    with pytest.raises(FileNotFoundError, match="Head checkpoint"):
        train_full(images_dir, split_path, tmp_path / "missing.pt", tmp_path / "full")
