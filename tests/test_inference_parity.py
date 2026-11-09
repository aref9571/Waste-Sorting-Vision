import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from wastevision.data.transforms import evaluation_transforms
from wastevision.inference.preprocess import preprocess_image

SIZES = [(512, 384), (384, 512), (300, 300), (257, 256), (1000, 333), (2000, 1500), (224, 224)]


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_numpy_preprocessing_matches_the_torchvision_evaluation_transform(width: int, height: int):
    pixels = np.random.default_rng(width + height).integers(0, 256, (height, width, 3), np.uint8)
    image = Image.fromarray(pixels)

    expected = evaluation_transforms()(image).numpy()

    assert np.abs(preprocess_image(image) - expected).max() < 1e-5


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_other_modes_match_after_conversion_to_rgb(mode: str):
    pixels = np.random.default_rng(1).integers(0, 256, (90, 120, 3), np.uint8)
    image = Image.fromarray(pixels).convert(mode)

    expected = evaluation_transforms()(image.convert("RGB")).numpy()

    assert np.abs(preprocess_image(image) - expected).max() < 1e-5
