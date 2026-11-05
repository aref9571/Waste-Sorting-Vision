from pathlib import Path

from PIL import Image, ImageOps

from wastevision.data.dataset import ImageRecord
from wastevision.labels import CLASS_NAMES

DEFAULT_REAL_DIR = Path("data/real")
REAL_SPLIT = "real"
PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class RealPhotoError(Exception):
    pass


def is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def photos_in_class_folder(folder: Path) -> list[Path]:
    photos: list[Path] = []
    for path in sorted(folder.iterdir()):
        if is_hidden(path):
            continue
        if not path.is_file() or path.suffix.lower() not in PHOTO_SUFFIXES:
            raise RealPhotoError(
                f"{path} is not a supported photo; use one of {', '.join(sorted(PHOTO_SUFFIXES))}"
            )
        photos.append(path)
    return photos


def find_real_photos(real_dir: Path) -> list[ImageRecord]:
    if not real_dir.is_dir():
        raise RealPhotoError(f"{real_dir} is not a folder")
    records: list[ImageRecord] = []
    for folder in sorted(real_dir.iterdir()):
        if is_hidden(folder):
            continue
        if not folder.is_dir() or folder.name not in CLASS_NAMES:
            raise RealPhotoError(
                f"{folder} is not a class folder; expected only {', '.join(CLASS_NAMES)}"
            )
        records.extend(
            ImageRecord(path=path, class_name=folder.name, split=REAL_SPLIT)
            for path in photos_in_class_folder(folder)
        )
    if not records:
        raise RealPhotoError(f"{real_dir} contains no photos")
    return records


def count_photos_per_class(records: list[ImageRecord]) -> dict[str, int]:
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for record in records:
        counts[record.class_name] += 1
    return counts


def load_photo_without_metadata(path: Path) -> Image.Image:
    with Image.open(path) as photo:
        upright = ImageOps.exif_transpose(photo).convert("RGB")
    return Image.frombytes("RGB", upright.size, upright.tobytes())
