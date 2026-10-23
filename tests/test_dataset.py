from pathlib import Path

import pytest

from wastevision.data.dataset import ImageRecord, SplitFileError, load_image_records

SPLIT_CSV = (
    "file,class,split\n"
    "glass/glass1.jpg,glass,train\n"
    "glass/glass2.jpg,glass,validation\n"
    "metal/metal1.jpg,metal,test\n"
    "trash/trash1.jpg,trash,train\n"
)


@pytest.fixture
def split_path(tmp_path: Path) -> Path:
    path = tmp_path / "split.csv"
    path.write_text(SPLIT_CSV, encoding="utf-8")
    return path


def test_load_image_records_reads_every_row(split_path: Path, tmp_path: Path):
    images_dir = tmp_path / "images"

    records = load_image_records(split_path, images_dir)

    assert records == [
        ImageRecord(images_dir / "glass/glass1.jpg", "glass", "train"),
        ImageRecord(images_dir / "glass/glass2.jpg", "glass", "validation"),
        ImageRecord(images_dir / "metal/metal1.jpg", "metal", "test"),
        ImageRecord(images_dir / "trash/trash1.jpg", "trash", "train"),
    ]


def test_load_image_records_filters_by_split(split_path: Path, tmp_path: Path):
    records = load_image_records(split_path, tmp_path, split="train")

    assert [record.path.name for record in records] == ["glass1.jpg", "trash1.jpg"]
    assert all(record.split == "train" for record in records)


def test_load_image_records_rejects_unknown_split_argument(split_path: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown split 'training'"):
        load_image_records(split_path, tmp_path, split="training")


def test_load_image_records_rejects_wrong_columns(tmp_path: Path):
    path = tmp_path / "split.csv"
    path.write_text("path,label\nglass/glass1.jpg,glass\n", encoding="utf-8")

    with pytest.raises(SplitFileError, match="must have the columns file,class,split"):
        load_image_records(path, tmp_path)


def test_load_image_records_rejects_empty_file(tmp_path: Path):
    path = tmp_path / "split.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(SplitFileError):
        load_image_records(path, tmp_path)


def test_load_image_records_rejects_unknown_class(tmp_path: Path):
    path = tmp_path / "split.csv"
    path.write_text("file,class,split\nwood/wood1.jpg,wood,train\n", encoding="utf-8")

    with pytest.raises(SplitFileError, match="Line 2: unknown class 'wood'"):
        load_image_records(path, tmp_path)


def test_load_image_records_rejects_unknown_split_value(tmp_path: Path):
    path = tmp_path / "split.csv"
    path.write_text(
        "file,class,split\nglass/glass1.jpg,glass,train\nglass/glass2.jpg,glass,holdout\n",
        encoding="utf-8",
    )

    with pytest.raises(SplitFileError, match="Line 3: unknown split 'holdout'"):
        load_image_records(path, tmp_path)


def test_committed_split_file_is_valid():
    split_path = Path(__file__).parents[1] / "splits" / "trashnet-split.csv"

    records = load_image_records(split_path, Path("images"))
    files = [record.path for record in records]

    assert len(files) == len(set(files)) == 2521
    assert {record.split for record in records} == {"train", "validation", "test"}
