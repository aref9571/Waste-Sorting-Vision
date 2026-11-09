import io

import numpy as np
import pytest
from PIL import Image

from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from wastevision.inference.preprocess import (
    Preprocessing,
    center_crop,
    preprocess_image,
    resized_size,
)


def random_image(width: int, height: int, seed: int = 0, mode: str = "RGB") -> Image.Image:
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels).convert(mode)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (512, 384, (341, 256)),
        (384, 512, (256, 341)),
        (300, 300, (256, 256)),
        (1000, 333, (768, 256)),
        (100, 100, (256, 256)),
    ],
)
def test_resized_size_scales_the_shorter_side(width: int, height: int, expected: tuple[int, int]):
    assert resized_size(width, height, 256) == expected


def test_center_crop_takes_the_middle():
    pixels = np.arange(6 * 4, dtype=np.uint8).reshape(4, 6)
    cropped = center_crop(Image.fromarray(pixels), 2)

    assert np.array_equal(np.asarray(cropped), pixels[1:3, 2:4])


@pytest.mark.parametrize("size", [(64, 48), (48, 64), (300, 300), (1000, 400)])
def test_output_has_the_model_input_shape(size: tuple[int, int]):
    result = preprocess_image(random_image(*size))

    assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]
    assert np.isfinite(result).all()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "LA"])
def test_every_common_mode_gives_the_same_shape(mode: str):
    result = preprocess_image(random_image(70, 50, mode=mode))

    assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert np.isfinite(result).all()


def test_flat_colour_becomes_the_normalised_colour():
    colour = (200, 100, 50)
    image = Image.new("RGB", (9, 7), colour)

    result = preprocess_image(image)

    for channel in range(3):
        expected = (colour[channel] / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert np.allclose(result[channel], expected, atol=1e-5)


def test_pixels_are_scaled_normalised_and_moved_to_the_first_axis():
    pixels = np.array([[[0, 255, 51], [204, 0, 255]], [[51, 204, 0], [255, 51, 204]]], np.uint8)
    settings = Preprocessing(resize=2, crop=2, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))

    result = preprocess_image(Image.fromarray(pixels), settings)

    expected = np.array(
        [
            [[-2.0, 1.2], [-1.2, 2.0]],
            [[2.0, -2.0], [1.2, -1.2]],
            [[-1.2, 2.0], [-2.0, 1.2]],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected, atol=1e-6)


def test_resizing_gives_the_stored_golden_values():
    pixels = (np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)) * 2
    settings = Preprocessing(resize=2, crop=2, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

    result = preprocess_image(Image.fromarray(pixels), settings)

    expected = np.array(
        [
            [[0.192157, 0.239216], [0.490196, 0.537255]],
            [[0.200000, 0.247059], [0.498039, 0.545098]],
            [[0.207843, 0.254902], [0.505882, 0.552941]],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected, atol=1e-5)


def test_preprocessing_is_deterministic_and_leaves_the_input_alone():
    image = random_image(80, 60)
    before = np.asarray(image).copy()

    first = preprocess_image(image)
    second = preprocess_image(image)

    assert np.array_equal(first, second)
    assert np.array_equal(np.asarray(image), before)


def test_exif_orientation_is_applied():
    upright = random_image(60, 40, seed=3)
    sideways = upright.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    sideways.save(buffer, format="PNG", exif=exif)
    buffer.seek(0)
    stored = Image.open(buffer)

    assert stored.size == (40, 60)
    assert np.array_equal(preprocess_image(stored), preprocess_image(upright))


def test_preprocessing_reads_its_numbers_from_metadata():
    metadata = {
        "input": {"resize": 64, "crop": 32, "mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
    }

    settings = Preprocessing.from_metadata(metadata)

    assert settings == Preprocessing(64, 32, (0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert preprocess_image(random_image(100, 80), settings).shape == (3, 32, 32)


@pytest.mark.parametrize(
    "options",
    [
        {"resize": 100, "crop": 200},
        {"crop": 0, "resize": 10},
        {"mean": (0.5, 0.5)},
        {"std": (0.2, 0.2)},
        {"std": (0.2, 0.0, 0.2)},
    ],
)
def test_invalid_preprocessing_settings_are_rejected(options: dict):
    with pytest.raises(ValueError, match=r"resize|crop|mean|std"):
        Preprocessing(**options)
