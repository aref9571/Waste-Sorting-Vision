from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from wastevision.training.loop import load_checkpoint
from wastevision.training.model import build_model
from wastevision.training.train import train_head


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
