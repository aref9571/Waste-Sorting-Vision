from collections.abc import Callable

import torch
from PIL import Image
from torchvision.transforms import v2

from wastevision.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_SIZE,
    Augmentation,
)

CROP_SCALE: tuple[float, float] = (0.7, 1.0)
ROTATION_DEGREES = 10
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.02


def to_normalized_tensor() -> list[v2.Transform]:
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def training_transforms(augmentation: Augmentation = Augmentation.STANDARD) -> v2.Compose:
    extra = [v2.RandAugment()] if augmentation is Augmentation.STRONG else []
    return v2.Compose(
        [
            v2.RandomResizedCrop(IMAGE_SIZE, scale=CROP_SCALE, antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(ROTATION_DEGREES),
            v2.ColorJitter(
                brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE
            ),
            *extra,
            *to_normalized_tensor(),
        ]
    )


def evaluation_transforms() -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(RESIZE_SIZE, antialias=True),
            v2.CenterCrop(IMAGE_SIZE),
            *to_normalized_tensor(),
        ]
    )


def perturbed_evaluation_transforms(
    perturbation: Callable[[Image.Image], Image.Image],
) -> v2.Compose:
    return v2.Compose([perturbation, *evaluation_transforms().transforms])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
