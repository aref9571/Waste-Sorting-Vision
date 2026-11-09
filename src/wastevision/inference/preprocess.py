from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE

MAX_PIXEL_VALUE = 255.0


@dataclass(frozen=True)
class Preprocessing:
    resize: int = RESIZE_SIZE
    crop: int = IMAGE_SIZE
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD

    def __post_init__(self) -> None:
        if self.resize < self.crop:
            raise ValueError("resize must not be smaller than crop")
        if self.crop < 1:
            raise ValueError("crop must be at least 1")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std need one value per colour channel")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "Preprocessing":
        settings = metadata["input"]
        return cls(
            resize=settings["resize"],
            crop=settings["crop"],
            mean=tuple(settings["mean"]),
            std=tuple(settings["std"]),
        )


def resized_size(width: int, height: int, shorter_side: int) -> tuple[int, int]:
    if width <= height:
        return shorter_side, max(shorter_side, int(shorter_side * height / width))
    return max(shorter_side, int(shorter_side * width / height)), shorter_side


def center_crop(image: Image.Image, size: int) -> Image.Image:
    left = round((image.width - size) / 2.0)
    top = round((image.height - size) / 2.0)
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image: Image.Image, settings: Preprocessing | None = None) -> np.ndarray:
    settings = settings or Preprocessing()
    upright = ImageOps.exif_transpose(image).convert("RGB")
    size = resized_size(upright.width, upright.height, settings.resize)
    resized = upright.resize(size, Image.Resampling.BILINEAR)
    cropped = center_crop(resized, settings.crop)
    scaled = np.asarray(cropped, dtype=np.float32) / np.float32(MAX_PIXEL_VALUE)
    mean = np.asarray(settings.mean, dtype=np.float32)
    std = np.asarray(settings.std, dtype=np.float32)
    normalised = (scaled - mean) / std
    return np.ascontiguousarray(normalised.transpose(2, 0, 1), dtype=np.float32)
