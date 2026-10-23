import numpy as np
import pytest
from PIL import Image

from wastevision.baseline.features import (
    FEATURE_IMAGE_SIZE,
    HSV_BINS,
    LBP_BIN_COUNT,
    color_histogram,
    extract_features,
    extract_features_from_files,
    hog_descriptor,
    lbp_histogram,
    prepare_image,
    to_grayscale,
)


def random_image(mode: str, size: tuple[int, int], seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    channels = len(mode)
    pixels = rng.integers(0, 256, size=(size[1], size[0], channels), dtype=np.uint8)
    return Image.fromarray(pixels.squeeze(axis=2) if channels == 1 else pixels, mode=mode)


def expected_feature_length() -> int:
    rgb = prepare_image(Image.new("RGB", (10, 10)))
    gray = to_grayscale(rgb)
    return len(color_histogram(rgb)) + len(lbp_histogram(gray)) + len(hog_descriptor(gray))


@pytest.mark.parametrize(
    ("mode", "size"),
    [
        ("RGB", (512, 384)),
        ("RGB", (128, 128)),
        ("RGBA", (64, 200)),
        ("L", (300, 100)),
        ("RGB", (7, 5)),
    ],
)
def test_feature_vector_has_constant_length_and_finite_values(mode: str, size: tuple[int, int]):
    features = extract_features(random_image(mode, size))

    assert features.shape == (expected_feature_length(),)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_feature_length_combines_all_parts():
    assert expected_feature_length() == int(np.prod(HSV_BINS)) + LBP_BIN_COUNT + 1764


def test_uniform_image_gives_finite_features():
    features = extract_features(Image.new("RGB", (50, 50), (128, 128, 128)))

    assert np.isfinite(features).all()


def test_prepare_image_resizes_and_converts_to_rgb():
    rgb = prepare_image(random_image("L", (300, 100)))

    assert rgb.shape == (FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE, 3)
    assert rgb.dtype == np.uint8


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_color_histogram_sums_to_one(seed: int):
    histogram = color_histogram(prepare_image(random_image("RGB", (90, 60), seed)))

    assert histogram.shape == (int(np.prod(HSV_BINS)),)
    assert histogram.sum() == pytest.approx(1.0)
    assert (histogram >= 0).all()


def test_color_histogram_puts_pure_red_into_one_bin():
    histogram = color_histogram(prepare_image(Image.new("RGB", (20, 20), (255, 0, 0))))

    hue_bin, saturation_bin, value_bin = 0, HSV_BINS[1] - 1, HSV_BINS[2] - 1
    expected_index = (hue_bin * HSV_BINS[1] + saturation_bin) * HSV_BINS[2] + value_bin
    assert histogram[expected_index] == pytest.approx(1.0)


def test_lbp_histogram_sums_to_one():
    gray = to_grayscale(prepare_image(random_image("RGB", (128, 128))))

    histogram = lbp_histogram(gray)

    assert histogram.shape == (LBP_BIN_COUNT,)
    assert histogram.sum() == pytest.approx(1.0)


def test_features_differ_for_different_images():
    first = extract_features(random_image("RGB", (64, 64), seed=1))
    second = extract_features(random_image("RGB", (64, 64), seed=2))

    assert not np.allclose(first, second)


def test_extract_features_from_files_stacks_one_row_per_file(tmp_path):
    paths = []
    for index, mode in enumerate(["RGB", "L", "RGBA"]):
        path = tmp_path / f"image{index}.png"
        random_image(mode, (40, 30), seed=index).save(path)
        paths.append(path)

    features = extract_features_from_files(paths)

    assert features.shape == (3, expected_feature_length())
    np.testing.assert_array_equal(features[1], extract_features(Image.open(paths[1])))


def test_extract_features_from_files_fails_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_features_from_files([tmp_path / "missing.png"])
