from pathlib import Path

import pytest
from PIL import Image

from wastevision.data.real_photos import (
    REAL_SPLIT,
    RealPhotoError,
    count_photos_per_class,
    find_real_photos,
    load_photo_without_metadata,
)
from wastevision.labels import CLASS_NAMES

EXIF_ORIENTATION = 0x0112
EXIF_GPS_INFO = 0x8825
ROTATED_90_CLOCKWISE = 6


def save_photo(path: Path, size: tuple[int, int] = (24, 16)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 120, 150)).save(path)
    return path


def save_photo_with_exif(path: Path) -> Path:
    image = Image.new("RGB", (40, 20), (255, 255, 255))
    image.paste((255, 0, 0), (0, 0, 10, 20))
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = ROTATED_90_CLOCKWISE
    exif[EXIF_GPS_INFO] = {1: "N", 2: (52.0, 31.0, 12.0)}
    image.save(path, exif=exif.tobytes(), quality=95)
    return path


def test_find_real_photos_lists_every_photo_with_its_folder_as_label(tmp_path: Path):
    save_photo(tmp_path / "glass" / "b.jpg")
    save_photo(tmp_path / "glass" / "a.JPG")
    save_photo(tmp_path / "trash" / "wrapper.png")
    save_photo(tmp_path / "cardboard" / "box.jpeg")

    records = find_real_photos(tmp_path)

    assert [
        (record.path.relative_to(tmp_path).as_posix(), record.class_name) for record in records
    ] == [
        ("cardboard/box.jpeg", "cardboard"),
        ("glass/a.JPG", "glass"),
        ("glass/b.jpg", "glass"),
        ("trash/wrapper.png", "trash"),
    ]
    assert {record.split for record in records} == {REAL_SPLIT}


def test_find_real_photos_ignores_hidden_files_and_folders(tmp_path: Path):
    save_photo(tmp_path / "metal" / "can.webp")
    (tmp_path / "metal" / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / ".thumbnails").mkdir()

    records = find_real_photos(tmp_path)

    assert [record.path.name for record in records] == ["can.webp"]


def test_find_real_photos_rejects_an_unknown_class_folder(tmp_path: Path):
    save_photo(tmp_path / "plastics" / "bottle.jpg")

    with pytest.raises(RealPhotoError, match="plastics is not a class folder"):
        find_real_photos(tmp_path)


def test_find_real_photos_rejects_a_file_outside_the_class_folders(tmp_path: Path):
    save_photo(tmp_path / "bottle.jpg")

    with pytest.raises(RealPhotoError, match="is not a class folder"):
        find_real_photos(tmp_path)


@pytest.mark.parametrize("name", ["bottle.heic", "notes.txt"])
def test_find_real_photos_rejects_unsupported_files(tmp_path: Path, name: str):
    folder = tmp_path / "plastic"
    folder.mkdir()
    (folder / name).write_bytes(b"data")

    with pytest.raises(RealPhotoError, match="is not a supported photo"):
        find_real_photos(tmp_path)


def test_find_real_photos_rejects_a_nested_folder(tmp_path: Path):
    save_photo(tmp_path / "paper" / "more" / "sheet.jpg")

    with pytest.raises(RealPhotoError, match="is not a supported photo"):
        find_real_photos(tmp_path)


def test_find_real_photos_rejects_a_missing_folder(tmp_path: Path):
    with pytest.raises(RealPhotoError, match="is not a folder"):
        find_real_photos(tmp_path / "missing")


def test_find_real_photos_rejects_a_folder_without_photos(tmp_path: Path):
    (tmp_path / "glass").mkdir()

    with pytest.raises(RealPhotoError, match="contains no photos"):
        find_real_photos(tmp_path)


def test_count_photos_per_class_includes_empty_classes(tmp_path: Path):
    save_photo(tmp_path / "glass" / "a.jpg")
    save_photo(tmp_path / "glass" / "b.jpg")
    save_photo(tmp_path / "paper" / "c.jpg")

    counts = count_photos_per_class(find_real_photos(tmp_path))

    assert list(counts) == list(CLASS_NAMES)
    assert counts == {
        "cardboard": 0,
        "glass": 2,
        "metal": 0,
        "paper": 1,
        "plastic": 0,
        "trash": 0,
    }


def test_load_photo_without_metadata_turns_the_photo_upright(tmp_path: Path):
    path = save_photo_with_exif(tmp_path / "photo.jpg")

    photo = load_photo_without_metadata(path)

    assert photo.size == (20, 40)
    red, green, _ = photo.getpixel((10, 2))
    assert red > 200
    assert green < 60
    assert photo.getpixel((10, 37)) == pytest.approx((255, 255, 255), abs=30)


def test_load_photo_without_metadata_drops_exif_and_location(tmp_path: Path):
    path = save_photo_with_exif(tmp_path / "photo.jpg")
    with Image.open(path) as original:
        assert EXIF_GPS_INFO in original.getexif()

    photo = load_photo_without_metadata(path)
    output = tmp_path / "copy.jpg"
    photo.save(output)

    assert photo.info == {}
    assert len(photo.getexif()) == 0
    with Image.open(output) as saved:
        assert len(saved.getexif()) == 0


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_load_photo_without_metadata_converts_to_rgb(tmp_path: Path, mode: str):
    path = tmp_path / "photo.png"
    Image.new(mode, (12, 8)).save(path)

    photo = load_photo_without_metadata(path)

    assert photo.mode == "RGB"
    assert photo.size == (12, 8)
