from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from torch import nn
from torch.nn import functional
from torchvision.models import MobileNetV3

from wastevision.data.image_dataset import load_rgb_image
from wastevision.data.transforms import denormalize, evaluation_transforms

OVERLAY_ALPHA = 0.4


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    lowest = heatmap.min()
    value_range = heatmap.max() - lowest
    if value_range == 0:
        return np.zeros_like(heatmap)
    return (heatmap - lowest) / value_range


def grad_cam(
    model: nn.Module, layer: nn.Module, image: torch.Tensor, class_index: int
) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("Expected one image with shape (channels, height, width)")
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_gradients(gradient: torch.Tensor) -> None:
        gradients.append(gradient.detach())

    def save_activations(module: nn.Module, inputs: object, output: torch.Tensor) -> None:
        activations.append(output.detach())
        output.register_hook(save_gradients)

    handle = layer.register_forward_hook(save_activations)
    model.eval()
    try:
        scores = model(image.unsqueeze(0))
        if not 0 <= class_index < scores.shape[1]:
            raise ValueError(f"Class index {class_index} is out of range")
        model.zero_grad()
        scores[0, class_index].backward()
    finally:
        handle.remove()
        model.zero_grad()

    channel_weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    heatmap = functional.relu((channel_weights * activations[0]).sum(dim=1, keepdim=True))
    heatmap = functional.interpolate(
        heatmap, size=tuple(image.shape[1:]), mode="bilinear", align_corners=False
    )
    return normalize_heatmap(heatmap[0, 0].numpy())


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = OVERLAY_ALPHA
) -> np.ndarray:
    if image.ndim != 3 or image.shape[:2] != heatmap.shape:
        raise ValueError("The heat map must have the image's height and width")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    colored = colormaps["jet"](heatmap)[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored, 0.0, 1.0)


def model_input(path: Path) -> torch.Tensor:
    return evaluation_transforms()(load_rgb_image(path))


def to_display_image(tensor: torch.Tensor) -> np.ndarray:
    return denormalize(tensor).permute(1, 2, 0).numpy()


def load_model_view(path: Path) -> np.ndarray:
    return to_display_image(model_input(path))


def explain_image(
    model: MobileNetV3, path: Path, class_index: int
) -> tuple[np.ndarray, np.ndarray]:
    tensor = model_input(path)
    heatmap = grad_cam(model, model.features[-1], tensor, class_index)
    image = to_display_image(tensor)
    return image, overlay_heatmap(image, heatmap)
