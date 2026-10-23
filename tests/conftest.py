from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wastevision.labels import CLASS_NAMES

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "cardboard": (160, 110, 60),
    "glass": (40, 160, 70),
    "metal": (150, 150, 160),
    "paper": (240, 240, 230),
    "plastic": (30, 80, 200),
    "trash": (200, 30, 40),
}
SPLIT_LAYOUT: tuple[str, ...] = ("train", "train", "train", "validation", "test")


def noisy_image(color: tuple[int, int, int], seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-20, 21, size=(48, 64, 3))
    pixels = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "images"
    rows = ["file,class,split"]
    for class_index, class_name in enumerate(CLASS_NAMES):
        (images_dir / class_name).mkdir(parents=True)
        for image_index, split in enumerate(SPLIT_LAYOUT):
            file = f"{class_name}/{class_name}{image_index}.png"
            noisy_image(CLASS_COLORS[class_name], seed=class_index * 10 + image_index).save(
                images_dir / file
            )
            rows.append(f"{file},{class_name},{split}")
    split_path = tmp_path / "split.csv"
    split_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return images_dir, split_path
