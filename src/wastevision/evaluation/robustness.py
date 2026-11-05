from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter

from wastevision.evaluation.metrics import ClassificationReport

BLUR_RADIUS = 2.0
BRIGHTNESS_CHANGE = 0.4
JPEG_QUALITY = 30
ROTATION_DEGREES = 15

Perturbation = Callable[[Image.Image], Image.Image]


def gaussian_blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))


def brighten(image: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(1 + BRIGHTNESS_CHANGE)


def darken(image: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(1 - BRIGHTNESS_CHANGE)


def jpeg_compress(image: Image.Image) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def rotate(image: Image.Image) -> Image.Image:
    return image.rotate(ROTATION_DEGREES, resample=Image.Resampling.BILINEAR)


PERTURBATIONS: dict[str, Perturbation] = {
    f"gaussian blur (radius {BLUR_RADIUS:g} px)": gaussian_blur,
    f"brightness +{BRIGHTNESS_CHANGE:.0%}": brighten,
    f"brightness -{BRIGHTNESS_CHANGE:.0%}": darken,
    f"jpeg quality {JPEG_QUALITY}": jpeg_compress,
    f"rotation {ROTATION_DEGREES} degrees": rotate,
}


@dataclass(frozen=True)
class RobustnessRow:
    name: str
    accuracy: float
    macro_f1: float
    accuracy_drop: float
    macro_f1_drop: float


def robustness_rows(
    clean: ClassificationReport, perturbed: Mapping[str, ClassificationReport]
) -> list[RobustnessRow]:
    return [
        RobustnessRow(
            name=name,
            accuracy=report.accuracy,
            macro_f1=report.macro_f1,
            accuracy_drop=clean.accuracy - report.accuracy,
            macro_f1_drop=clean.macro_f1 - report.macro_f1,
        )
        for name, report in perturbed.items()
    ]
