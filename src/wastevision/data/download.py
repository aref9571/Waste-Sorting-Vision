import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from wastevision.labels import CLASS_NAMES

DATASET_URL = (
    "https://raw.githubusercontent.com/garythung/trashnet/"
    "6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip"
)
DATASET_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
ARCHIVE_NAME = "dataset-resized.zip"
IMAGES_DIR_NAME = "dataset-resized"
DEFAULT_RAW_DIR = Path("data/raw")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
EXPECTED_IMAGE_COUNTS: dict[str, int] = {
    "cardboard": 403,
    "glass": 501,
    "metal": 410,
    "paper": 594,
    "plastic": 482,
    "trash": 137,
}
CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60

Fetcher = Callable[[str, Path], None]


class ChecksumMismatchError(Exception):
    pass


class UnsafeArchiveError(Exception):
    pass


class DatasetError(Exception):
    pass


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> None:
    actual_sha256 = sha256_of_file(path)
    if actual_sha256 != expected_sha256:
        raise ChecksumMismatchError(
            f"SHA-256 of {path} is {actual_sha256}, expected {expected_sha256}"
        )


def fetch_url(url: str, destination: Path) -> None:
    with (
        urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        destination.open("wb") as file,
    ):
        shutil.copyfileobj(response, file, CHUNK_SIZE)


def ensure_safe_member(name: str, target_root: Path) -> None:
    member_path = PurePosixPath(name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise UnsafeArchiveError(f"Unsafe path in archive: {name!r}")
    if not (target_root / member_path).resolve().is_relative_to(target_root):
        raise UnsafeArchiveError(f"Archive member escapes the target folder: {name!r}")


def safe_extract(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            ensure_safe_member(member.filename, target_root)
        archive.extractall(target_root)


def images_present(images_dir: Path) -> bool:
    return all((images_dir / class_name).is_dir() for class_name in CLASS_NAMES)


def download_archive(url: str, archive_path: Path, expected_sha256: str, fetch: Fetcher) -> None:
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    fetch(url, partial_path)
    try:
        verify_checksum(partial_path, expected_sha256)
    except ChecksumMismatchError:
        partial_path.unlink()
        raise
    partial_path.replace(archive_path)


def extract_images(archive_path: Path, raw_dir: Path) -> Path:
    images_dir = raw_dir / IMAGES_DIR_NAME
    with tempfile.TemporaryDirectory(dir=raw_dir) as temporary_dir:
        safe_extract(archive_path, Path(temporary_dir))
        extracted_images_dir = Path(temporary_dir) / IMAGES_DIR_NAME
        if not images_present(extracted_images_dir):
            raise DatasetError(
                f"Archive {archive_path} does not contain the expected class folders"
            )
        extracted_images_dir.rename(images_dir)
    return images_dir


def download_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
    url: str = DATASET_URL,
    expected_sha256: str = DATASET_SHA256,
    fetch: Fetcher = fetch_url,
) -> Path:
    images_dir = raw_dir / IMAGES_DIR_NAME
    if images_present(images_dir):
        return images_dir
    if images_dir.exists():
        raise DatasetError(f"{images_dir} is incomplete; delete it and run the download again")
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / ARCHIVE_NAME
    if archive_path.exists():
        verify_checksum(archive_path, expected_sha256)
    else:
        download_archive(url, archive_path, expected_sha256, fetch)
    return extract_images(archive_path, raw_dir)


def count_images_per_class(images_dir: Path) -> dict[str, int]:
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        files = class_dir.iterdir() if class_dir.is_dir() else []
        counts[class_name] = sum(
            1 for file in files if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


def count_differences(counts: dict[str, int]) -> dict[str, tuple[int, int]]:
    return {
        class_name: (counts.get(class_name, 0), expected)
        for class_name, expected in EXPECTED_IMAGE_COUNTS.items()
        if counts.get(class_name, 0) != expected
    }
