import numpy as np
import pytest
from PIL import Image

from wastevision.evaluation.metrics import ClassificationReport
from wastevision.evaluation.robustness import (
    PERTURBATIONS,
    brighten,
    darken,
    gaussian_blur,
    jpeg_compress,
    robustness_rows,
    rotate,
)


def noisy_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8))


def pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.int64)


def report(accuracy: float, macro_f1: float) -> ClassificationReport:
    return ClassificationReport(accuracy, macro_f1, {}, np.zeros((1, 1)))


@pytest.mark.parametrize("perturbation", list(PERTURBATIONS.values()))
def test_every_perturbation_keeps_the_size_and_mode(perturbation):
    result = perturbation(noisy_image())

    assert result.size == (80, 60)
    assert result.mode == "RGB"


def test_perturbations_have_the_planned_names():
    assert list(PERTURBATIONS) == [
        "gaussian blur (radius 2 px)",
        "brightness +40%",
        "brightness -40%",
        "jpeg quality 30",
        "rotation 15 degrees",
    ]


def test_gaussian_blur_smooths_noise():
    image = noisy_image()

    blurred = gaussian_blur(image)

    assert pixels(blurred).std() < pixels(image).std() / 2


def test_gaussian_blur_keeps_a_uniform_image():
    image = Image.new("RGB", (40, 30), (120, 60, 30))

    assert pixels(gaussian_blur(image)).tolist() == pixels(image).tolist()


def test_brighten_and_darken_scale_pixel_values_by_forty_percent():
    image = Image.new("RGB", (4, 4), (100, 50, 200))

    assert pixels(brighten(image))[0, 0].tolist() == [140, 70, 255]
    assert pixels(darken(image))[0, 0].tolist() == [60, 30, 120]


def test_jpeg_compress_changes_a_detailed_image_and_is_deterministic():
    image = noisy_image()

    first = jpeg_compress(image)
    second = jpeg_compress(image)

    assert not np.array_equal(pixels(first), pixels(image))
    assert np.array_equal(pixels(first), pixels(second))


def test_jpeg_compress_accepts_an_image_with_transparency():
    image = Image.new("RGBA", (16, 16), (10, 20, 30, 128))

    assert jpeg_compress(image).mode == "RGB"


def test_rotate_fills_the_corners_with_black_and_keeps_the_center():
    image = Image.new("RGB", (80, 60), (255, 255, 255))

    rotated = pixels(rotate(image))

    assert rotated[0, 0].tolist() == [0, 0, 0]
    assert rotated[30, 40].tolist() == [255, 255, 255]


def test_robustness_rows_compute_the_drops_in_order():
    clean = report(accuracy=0.9, macro_f1=0.8)
    perturbed = {"blur": report(0.7, 0.5), "darker": report(0.95, 0.85)}

    rows = robustness_rows(clean, perturbed)

    assert [row.name for row in rows] == ["blur", "darker"]
    assert rows[0].accuracy == pytest.approx(0.7)
    assert rows[0].accuracy_drop == pytest.approx(0.2)
    assert rows[0].macro_f1_drop == pytest.approx(0.3)
    assert rows[1].accuracy_drop == pytest.approx(-0.05)
    assert rows[1].macro_f1_drop == pytest.approx(-0.05)


def test_robustness_rows_without_perturbations_are_empty():
    assert robustness_rows(report(0.9, 0.8), {}) == []
