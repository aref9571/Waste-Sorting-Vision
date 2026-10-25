from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from PIL import Image

from wastevision.data.dataset import ImageRecord, load_image_records
from wastevision.data.preview import first_record_per_class, save_augmentation_preview
from wastevision.labels import CLASS_NAMES


def test_first_record_per_class_picks_first_file_of_each_class(tmp_path: Path):
    records = [
        ImageRecord(tmp_path / "metal/metal2.jpg", "metal", "train"),
        ImageRecord(tmp_path / "glass/glass9.jpg", "glass", "train"),
        ImageRecord(tmp_path / "metal/metal1.jpg", "metal", "train"),
    ]

    chosen = first_record_per_class(records)

    assert [record.path.name for record in chosen] == ["glass9.jpg", "metal1.jpg"]


def test_save_augmentation_preview_writes_png_grid(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset
    records = first_record_per_class(load_image_records(split_path, images_dir, "train"))
    output = tmp_path / "img" / "preview.png"

    save_augmentation_preview(records, output, augmentations=3)

    with Image.open(output) as image:
        assert image.format == "PNG"
        width, height = image.size
    assert len(records) == len(CLASS_NAMES)
    assert height > width


def test_save_augmentation_preview_is_reproducible(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset
    records = first_record_per_class(load_image_records(split_path, images_dir, "train"))[:2]

    save_augmentation_preview(records, tmp_path / "first.png", seed=3)
    save_augmentation_preview(records, tmp_path / "second.png", seed=3)

    assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()


def test_save_augmentation_preview_rejects_empty_list(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        save_augmentation_preview([], tmp_path / "preview.png")
