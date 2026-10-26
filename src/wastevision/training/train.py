from pathlib import Path

from wastevision.config import Settings
from wastevision.data.image_dataset import make_data_loader, split_dataset
from wastevision.training.loop import FitResult, fit, set_seed
from wastevision.training.model import build_model, freeze_backbone

DEFAULT_HEAD_RUN_DIR = Path("runs/head")
HEAD_EPOCHS = 8
HEAD_LEARNING_RATE = 1e-3
BATCH_SIZE = 32
SEED = 42


def train_head(
    images_dir: Path,
    split_path: Path,
    run_dir: Path = DEFAULT_HEAD_RUN_DIR,
    epochs: int = HEAD_EPOCHS,
    learning_rate: float = HEAD_LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    pretrained: bool = True,
    settings: Settings | None = None,
) -> FitResult:
    set_seed(seed)
    model = build_model(pretrained=pretrained)
    freeze_backbone(model)
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
    return fit(model, train_loader, validation_loader, run_dir, epochs, learning_rate)
