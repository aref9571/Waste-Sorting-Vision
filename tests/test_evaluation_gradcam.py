from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image
from torch import nn

from wastevision.config import IMAGE_SIZE
from wastevision.evaluation.gradcam import (
    explain_image,
    grad_cam,
    load_model_view,
    normalize_heatmap,
    overlay_heatmap,
)
from wastevision.training.model import build_model


def bright_corner_model() -> tuple[nn.Module, nn.Module]:
    layer = nn.Conv2d(1, 1, kernel_size=1, bias=False)
    classifier = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        classifier.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    model = nn.Sequential(layer, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), classifier)
    return model, layer


def bright_corner_image() -> torch.Tensor:
    image = torch.zeros(1, 8, 8)
    image[0, :4, :4] = 1.0
    return image


def test_grad_cam_has_the_input_size_and_values_between_zero_and_one():
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    image = torch.randn(3, 96, 128)

    heatmap = grad_cam(model, model.features[-1], image, class_index=2)

    assert heatmap.shape == (96, 128)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert heatmap.max() == pytest.approx(1.0)


def test_grad_cam_highlights_the_region_that_raises_the_class_score():
    model, layer = bright_corner_model()

    heatmap = grad_cam(model, layer, bright_corner_image(), class_index=0)

    assert heatmap[:4, :4].mean() > 0.9
    assert heatmap[4:, 4:].max() == 0.0


def test_grad_cam_ignores_regions_that_lower_the_class_score():
    model, layer = bright_corner_model()

    heatmap = grad_cam(model, layer, bright_corner_image(), class_index=1)

    assert heatmap.tolist() == np.zeros((8, 8)).tolist()


def test_grad_cam_removes_its_hooks_and_gradients():
    model, layer = bright_corner_model()

    grad_cam(model, layer, bright_corner_image(), class_index=0)

    assert not layer._forward_hooks
    assert not layer._backward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


def test_grad_cam_puts_the_model_in_eval_mode():
    model, layer = bright_corner_model()
    model.train()

    grad_cam(model, layer, bright_corner_image(), class_index=0)

    assert not model.training


def test_grad_cam_rejects_a_batch():
    model, layer = bright_corner_model()

    with pytest.raises(ValueError, match="one image"):
        grad_cam(model, layer, bright_corner_image().unsqueeze(0), class_index=0)


@pytest.mark.parametrize("class_index", [-1, 2])
def test_grad_cam_rejects_an_unknown_class(class_index: int):
    model, layer = bright_corner_model()

    with pytest.raises(ValueError, match="out of range"):
        grad_cam(model, layer, bright_corner_image(), class_index=class_index)
    assert not layer._forward_hooks
    assert not layer._backward_hooks


def test_normalize_heatmap_scales_to_zero_and_one():
    heatmap = normalize_heatmap(np.array([[2.0, 4.0], [6.0, 10.0]]))

    assert heatmap.tolist() == [[0.0, 0.25], [0.5, 1.0]]


def test_normalize_heatmap_of_a_constant_map_is_zero():
    heatmap = normalize_heatmap(np.full((3, 3), 5.0))

    assert heatmap.tolist() == np.zeros((3, 3)).tolist()


def test_overlay_heatmap_blends_colors_into_the_image():
    image = np.full((4, 5, 3), 0.5)
    heatmap = np.linspace(0, 1, 20).reshape(4, 5)

    overlay = overlay_heatmap(image, heatmap)

    assert overlay.shape == (4, 5, 3)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0
    assert not np.allclose(overlay, image)
    assert overlay_heatmap(image, heatmap, alpha=0.0).tolist() == image.tolist()


def test_overlay_heatmap_stays_within_zero_and_one_for_float32_images():
    image = np.ones((4, 5, 3), dtype=np.float32)
    heatmap = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)

    overlay = overlay_heatmap(image, heatmap)

    assert overlay.max() <= 1.0


def test_overlay_heatmap_rejects_a_different_size():
    with pytest.raises(ValueError, match="height and width"):
        overlay_heatmap(np.zeros((4, 5, 3)), np.zeros((5, 4)))


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_overlay_heatmap_rejects_an_invalid_alpha(alpha: float):
    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(np.zeros((2, 2, 3)), np.zeros((2, 2)), alpha=alpha)


def test_load_model_view_shows_the_center_crop(tiny_dataset: tuple[Path, Path]):
    images_dir, _ = tiny_dataset

    image = load_model_view(images_dir / "glass" / "glass0.png")

    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_explain_image_returns_the_image_and_its_overlay(tiny_dataset: tuple[Path, Path]):
    images_dir, _ = tiny_dataset
    torch.manual_seed(0)
    model = build_model(pretrained=False)

    image, overlay = explain_image(model, images_dir / "metal" / "metal0.png", class_index=2)

    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.shape == image.shape
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0


def test_load_model_view_uses_the_given_image_loader(tmp_path: Path):
    def load_white_image(path: Path) -> Image.Image:
        return Image.new("RGB", (300, 260), (255, 255, 255))

    image = load_model_view(tmp_path / "not-read.jpg", load_white_image)

    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert image.min() == pytest.approx(1.0, abs=1e-6)
