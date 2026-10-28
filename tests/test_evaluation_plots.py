import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from wastevision.evaluation.calibration import reliability_bins
from wastevision.evaluation.plots import save_confusion_matrix, save_reliability_diagram

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
