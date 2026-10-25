from collections.abc import Sequence
from pathlib import Path

import torch
from matplotlib.figure import Figure

from wastevision.data.dataset import ImageRecord
from wastevision.data.image_dataset import load_rgb_image
from wastevision.data.transforms import denormalize, training_transforms

PREVIEW_CELL_INCHES = 1.6
PREVIEW_DPI = 80


def first_record_per_class(records: Sequence[ImageRecord]) -> list[ImageRecord]:
    chosen: dict[str, ImageRecord] = {}
    for record in sorted(records, key=lambda record: record.path):
        chosen.setdefault(record.class_name, record)
    return sorted(chosen.values(), key=lambda record: record.class_name)


def save_augmentation_preview(
    records: Sequence[ImageRecord], output: Path, augmentations: int = 5, seed: int = 0
) -> None:
    if not records:
        raise ValueError("No images to preview")
    columns = augmentations + 1
    figure = Figure(figsize=(columns * PREVIEW_CELL_INCHES, len(records) * PREVIEW_CELL_INCHES))
    axes = figure.subplots(len(records), columns, squeeze=False)
    transform = training_transforms()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for row, record in zip(axes, records, strict=True):
            image = load_rgb_image(record.path)
            row[0].imshow(image)
            row[0].set_title(f"{record.class_name}: original", fontsize=8)
            for axis in row[1:]:
                augmented = denormalize(transform(image))
                axis.imshow(augmented.permute(1, 2, 0).numpy())
            for axis in row:
                axis.set_axis_off()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=PREVIEW_DPI)
