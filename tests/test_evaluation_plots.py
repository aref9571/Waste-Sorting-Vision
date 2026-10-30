import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from wastevision.evaluation.calibration import reliability_bins
from wastevision.evaluation.plots import (
    save_confusion_matrix,
    save_image_grid,
    save_reliability_diagram,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reliability_diagram_is_written_and_reproducible(tmp_path: Path):
    bins = reliability_bins([0.95, 0.9, 0.55, 0.7], [True, True, False, True])
    first = tmp_path / "first" / "reliability.png"
    second = tmp_path / "second.png"

    save_reliability_diagram(bins, first)
    save_reliability_diagram(bins, second)

    assert first.read_bytes().startswith(PNG_SIGNATURE)
    assert file_hash(first) == file_hash(second)


def test_confusion_matrix_is_written_and_reproducible(tmp_path: Path):
    confusion = np.array([[5, 1, 0], [0, 4, 2], [1, 0, 6]])
    first = tmp_path / "first" / "confusion.png"
    second = tmp_path / "second.png"

    save_confusion_matrix(confusion, ("a", "b", "c"), first, "title")
    save_confusion_matrix(confusion, ("a", "b", "c"), second, "title")

    assert first.read_bytes().startswith(PNG_SIGNATURE)
    assert file_hash(first) == file_hash(second)


def test_confusion_matrix_must_match_the_class_names(tmp_path: Path):
    output = tmp_path / "confusion.png"

    with pytest.raises(ValueError, match="one row and one column per class"):
        save_confusion_matrix(np.zeros((2, 2), dtype=int), ("a", "b", "c"), output, "title")
    assert not output.exists()


def grid_images(count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random((12, 16, 3)) for _ in range(count)]


def test_image_grid_is_written_and_reproducible(tmp_path: Path):
    images = grid_images(5)
    titles = [f"image {index}" for index in range(5)]
    first = tmp_path / "first" / "grid.png"
    second = tmp_path / "second.png"

    save_image_grid(images, titles, first, columns=4)
    save_image_grid(images, titles, second, columns=4)

    assert first.read_bytes().startswith(PNG_SIGNATURE)
    assert file_hash(first) == file_hash(second)


@pytest.mark.parametrize(
    ("image_count", "titles", "columns", "message"),
    [
        (0, [], 4, "No images"),
        (2, ["only one"], 4, "one title per image"),
        (1, ["one"], 0, "columns"),
    ],
)
def test_image_grid_rejects_invalid_input(
    tmp_path: Path, image_count: int, titles: list[str], columns: int, message: str
):
    output = tmp_path / "grid.png"

    with pytest.raises(ValueError, match=message):
        save_image_grid(grid_images(image_count), titles, output, columns)
    assert not output.exists()
