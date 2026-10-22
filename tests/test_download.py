import hashlib
import zipfile
from pathlib import Path

import pytest

from wastevision.data.download import (
    IMAGES_DIR_NAME,
    ChecksumMismatchError,
    DatasetError,
    UnsafeArchiveError,
    count_differences,
    count_images_per_class,
    download_dataset,
    fetch_url,
    safe_extract,
    sha256_of_file,
    verify_checksum,
)
from wastevision.labels import CLASS_NAMES

FAKE_URL = "https://example.com/dataset.zip"


def build_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def dataset_members() -> dict[str, bytes]:
    members = {f"{IMAGES_DIR_NAME}/{name}/{name}1.jpg": b"image" for name in CLASS_NAMES}
    members["__MACOSX/._metadata"] = b"metadata"
    return members


def sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class FakeFetcher:
    def __init__(self, content: bytes):
        self.content = content
        self.calls: list[str] = []

    def __call__(self, url: str, destination: Path) -> None:
        self.calls.append(url)
        destination.write_bytes(self.content)


@pytest.fixture
def archive_bytes(tmp_path: Path) -> bytes:
    return build_zip(tmp_path / "source.zip", dataset_members()).read_bytes()


def test_sha256_of_file_matches_hashlib(tmp_path: Path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"waste sorting")

    assert sha256_of_file(path) == sha256_of_bytes(b"waste sorting")


def test_verify_checksum_accepts_matching_file(tmp_path: Path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"content")

    verify_checksum(path, sha256_of_bytes(b"content"))


def test_verify_checksum_rejects_mismatch(tmp_path: Path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"content")

    with pytest.raises(ChecksumMismatchError):
        verify_checksum(path, sha256_of_bytes(b"other content"))


def test_safe_extract_extracts_normal_archive(tmp_path: Path):
    archive = build_zip(tmp_path / "ok.zip", {"folder/file.txt": b"hello"})
    target = tmp_path / "target"
    target.mkdir()

    safe_extract(archive, target)

    assert (target / "folder" / "file.txt").read_bytes() == b"hello"


@pytest.mark.parametrize(
    "malicious_name",
    ["../escaped.txt", "folder/../../escaped.txt", "/tmp/escaped.txt"],
)
def test_safe_extract_rejects_zip_slip(tmp_path: Path, malicious_name: str):
    archive = build_zip(tmp_path / "evil.zip", {"good.txt": b"good", malicious_name: b"evil"})
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, target)

    assert not (tmp_path / "escaped.txt").exists()
    assert list(target.iterdir()) == []


def test_safe_extract_rejects_member_escaping_through_symlink(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / "link").symlink_to(outside, target_is_directory=True)
    archive = build_zip(tmp_path / "evil.zip", {"link/escaped.txt": b"evil"})

    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, target)

    assert not (outside / "escaped.txt").exists()


def test_download_dataset_downloads_verifies_and_extracts(tmp_path: Path, archive_bytes: bytes):
    raw_dir = tmp_path / "raw"
    fetcher = FakeFetcher(archive_bytes)

    images_dir = download_dataset(
        raw_dir, url=FAKE_URL, expected_sha256=sha256_of_bytes(archive_bytes), fetch=fetcher
    )

    assert fetcher.calls == [FAKE_URL]
    assert images_dir == raw_dir / IMAGES_DIR_NAME
    assert count_images_per_class(images_dir) == dict.fromkeys(CLASS_NAMES, 1)
    assert sorted(path.name for path in raw_dir.iterdir()) == [
        IMAGES_DIR_NAME,
        "dataset-resized.zip",
    ]


def test_download_dataset_skips_when_images_already_present(tmp_path: Path, archive_bytes: bytes):
    raw_dir = tmp_path / "raw"
    checksum = sha256_of_bytes(archive_bytes)
    download_dataset(
        raw_dir, url=FAKE_URL, expected_sha256=checksum, fetch=FakeFetcher(archive_bytes)
    )
    second_fetcher = FakeFetcher(archive_bytes)

    images_dir = download_dataset(
        raw_dir, url=FAKE_URL, expected_sha256=checksum, fetch=second_fetcher
    )

    assert second_fetcher.calls == []
    assert images_dir == raw_dir / IMAGES_DIR_NAME


def test_download_dataset_reuses_existing_archive(tmp_path: Path, archive_bytes: bytes):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "dataset-resized.zip").write_bytes(archive_bytes)
    fetcher = FakeFetcher(archive_bytes)

    images_dir = download_dataset(
        raw_dir, url=FAKE_URL, expected_sha256=sha256_of_bytes(archive_bytes), fetch=fetcher
    )

    assert fetcher.calls == []
    assert count_images_per_class(images_dir) == dict.fromkeys(CLASS_NAMES, 1)


def test_download_dataset_rejects_download_with_wrong_checksum(
    tmp_path: Path, archive_bytes: bytes
):
    raw_dir = tmp_path / "raw"

    with pytest.raises(ChecksumMismatchError):
        download_dataset(
            raw_dir,
            url=FAKE_URL,
            expected_sha256=sha256_of_bytes(b"something else"),
            fetch=FakeFetcher(archive_bytes),
        )

    assert list(raw_dir.iterdir()) == []


def test_download_dataset_rejects_existing_archive_with_wrong_checksum(
    tmp_path: Path, archive_bytes: bytes
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "dataset-resized.zip").write_bytes(b"corrupted")

    with pytest.raises(ChecksumMismatchError):
        download_dataset(
            raw_dir,
            url=FAKE_URL,
            expected_sha256=sha256_of_bytes(archive_bytes),
            fetch=FakeFetcher(archive_bytes),
        )

    assert not (raw_dir / IMAGES_DIR_NAME).exists()


def test_download_dataset_rejects_archive_without_class_folders(tmp_path: Path):
    archive_bytes = build_zip(tmp_path / "wrong.zip", {"other/file.jpg": b"x"}).read_bytes()
    raw_dir = tmp_path / "raw"

    with pytest.raises(DatasetError):
        download_dataset(
            raw_dir,
            url=FAKE_URL,
            expected_sha256=sha256_of_bytes(archive_bytes),
            fetch=FakeFetcher(archive_bytes),
        )

    assert not (raw_dir / IMAGES_DIR_NAME).exists()


def test_download_dataset_refuses_incomplete_images_folder(tmp_path: Path, archive_bytes: bytes):
    raw_dir = tmp_path / "raw"
    (raw_dir / IMAGES_DIR_NAME / "glass").mkdir(parents=True)

    with pytest.raises(DatasetError):
        download_dataset(
            raw_dir,
            url=FAKE_URL,
            expected_sha256=sha256_of_bytes(archive_bytes),
            fetch=FakeFetcher(archive_bytes),
        )


def test_fetch_url_copies_response_to_destination(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"remote content")
    destination = tmp_path / "destination.bin"

    fetch_url(source.as_uri(), destination)

    assert destination.read_bytes() == b"remote content"


def test_count_images_per_class_counts_only_images(tmp_path: Path):
    (tmp_path / "glass").mkdir()
    (tmp_path / "glass" / "a.jpg").write_bytes(b"x")
    (tmp_path / "glass" / "b.JPG").write_bytes(b"x")
    (tmp_path / "glass" / ".DS_Store").write_bytes(b"x")
    (tmp_path / "glass" / "notes.txt").write_bytes(b"x")

    counts = count_images_per_class(tmp_path)

    assert counts["glass"] == 2
    assert counts["paper"] == 0
    assert list(counts) == list(CLASS_NAMES)


def test_count_differences_is_empty_for_published_counts():
    published = {
        "cardboard": 403,
        "glass": 501,
        "metal": 410,
        "paper": 594,
        "plastic": 482,
        "trash": 137,
    }

    assert count_differences(published) == {}


def test_count_differences_reports_actual_and_expected():
    counts = {
        "cardboard": 403,
        "glass": 500,
        "metal": 410,
        "paper": 594,
        "plastic": 482,
    }

    assert count_differences(counts) == {"glass": (500, 501), "trash": (0, 137)}
