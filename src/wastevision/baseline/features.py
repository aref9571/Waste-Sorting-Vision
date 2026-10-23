from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import hog, local_binary_pattern

FEATURE_IMAGE_SIZE = 128
HSV_BINS: tuple[int, int, int] = (8, 4, 4)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BIN_COUNT = LBP_POINTS + 2
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL: tuple[int, int] = (16, 16)
HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)


def prepare_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(
        (FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)


def color_histogram(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb).reshape(-1, 3)
    counts, _ = np.histogramdd(hsv, bins=HSV_BINS, range=((0.0, 1.0),) * 3)
    return counts.ravel() / counts.sum()


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    codes = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    counts = np.bincount(codes.astype(np.int64).ravel(), minlength=LBP_BIN_COUNT)
    return counts / counts.sum()


def hog_descriptor(gray: np.ndarray) -> np.ndarray:
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )


def extract_features(image: Image.Image) -> np.ndarray:
    rgb = prepare_image(image)
    gray = to_grayscale(rgb)
    parts = (color_histogram(rgb), lbp_histogram(gray), hog_descriptor(gray))
    return np.concatenate(parts).astype(np.float32)


def extract_features_from_files(paths: Iterable[Path]) -> np.ndarray:
    vectors = []
    for path in paths:
        with Image.open(path) as image:
            vectors.append(extract_features(image))
    return np.stack(vectors)
