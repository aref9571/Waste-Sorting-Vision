import csv
import random
from collections import Counter
from pathlib import Path

import pytest

from wastevision.data.split import (
    SPLIT_COLUMNS,
    SPLIT_NAMES,
    ImageFile,
    SplitEntry,
    SplitError,
    choose_split,
    count_per_split,
    create_split,
    group_by_content,
    list_images,
    separate_conflicting_groups,
    target_sizes,
    write_split_csv,
)
from wastevision.labels import CLASS_NAMES

IMAGES_PER_CLASS = {
    "cardboard": 40,
    "glass": 50,
    "metal": 41,
    "paper": 59,
    "plastic": 48,
    "trash": 14,
}


def create_images(images_dir: Path, images_per_class: dict[str, int]) -> None:
    for class_name, count in images_per_class.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (class_dir / f"{class_name}{index}.jpg").write_bytes(f"{class_name}{index}".encode())


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    create_images(directory, IMAGES_PER_CLASS)
    return directory


def split_of(entries: list[SplitEntry]) -> dict[str, str]:
    return {entry.file: entry.split for entry in entries}


def test_list_images_finds_only_image_files(images_dir: Path):
    (images_dir / "glass" / "notes.txt").write_text("not an image")
    (images_dir / "glass" / "folder.jpg").mkdir()
    (images_dir / ".DS_Store").write_bytes(b"metadata")

    images = list_images(images_dir)

    assert len(images) == sum(IMAGES_PER_CLASS.values())
    assert all(image.file.endswith(".jpg") for image in images)
    assert {image.class_name for image in images} == set(CLASS_NAMES)


def test_list_images_rejects_missing_class_folder(tmp_path: Path):
    create_images(tmp_path, {"glass": 5})

    with pytest.raises(SplitError, match="Missing class folder"):
        list_images(tmp_path)


def test_group_by_content_groups_identical_files():
    images = [
        ImageFile("glass/a.jpg", "glass", "hash1"),
        ImageFile("glass/b.jpg", "glass", "hash2"),
        ImageFile("glass/c.jpg", "glass", "hash1"),
    ]

    groups = group_by_content(images)

    assert [[image.file for image in group] for group in groups] == [
        ["glass/a.jpg", "glass/c.jpg"],
        ["glass/b.jpg"],
    ]


def test_separate_conflicting_groups_excludes_groups_with_two_classes():
    same_class = [ImageFile("glass/a.jpg", "glass", "h1"), ImageFile("glass/b.jpg", "glass", "h1")]
    conflict = [ImageFile("glass/c.jpg", "glass", "h2"), ImageFile("metal/d.jpg", "metal", "h2")]

    kept, excluded = separate_conflicting_groups([same_class, conflict])

    assert kept == [same_class]
    assert excluded == conflict


@pytest.mark.parametrize(
    ("image_count", "expected"),
    [
        (100, {"train": 70, "validation": 15, "test": 15}),
        (403, {"train": 283, "validation": 60, "test": 60}),
        (4, {"train": 2, "validation": 1, "test": 1}),
    ],
)
def test_target_sizes(image_count: int, expected: dict[str, int]):
    assert target_sizes(image_count) == expected


def test_choose_split_fills_test_then_validation_then_train():
    targets = {"train": 7, "validation": 2, "test": 2}

    assert choose_split({"train": 0, "validation": 0, "test": 0}, targets, 1) == "test"
    assert choose_split({"train": 0, "validation": 0, "test": 2}, targets, 1) == "validation"
    assert choose_split({"train": 0, "validation": 2, "test": 2}, targets, 1) == "train"
    assert choose_split({"train": 0, "validation": 1, "test": 1}, targets, 2) == "train"


def test_split_proportions_are_within_one_image_per_class(images_dir: Path):
    counts = count_per_split(create_split(images_dir).entries)

    for class_name, image_count in IMAGES_PER_CLASS.items():
        for split, fraction in (("train", 0.70), ("validation", 0.15), ("test", 0.15)):
            assert abs(counts[class_name][split] - image_count * fraction) <= 1


def test_split_is_deterministic_for_the_same_seed(images_dir: Path):
    assert create_split(images_dir, seed=42) == create_split(images_dir, seed=42)


def test_split_changes_with_the_seed(images_dir: Path):
    assert create_split(images_dir, seed=1) != create_split(images_dir, seed=2)


def test_every_image_is_in_exactly_one_split(images_dir: Path):
    entries = create_split(images_dir).entries
    files = [entry.file for entry in entries]

    assert len(files) == len(set(files)) == sum(IMAGES_PER_CLASS.values())
    assert {entry.split for entry in entries} <= set(SPLIT_NAMES)


def test_every_class_appears_in_every_split(images_dir: Path):
    counts = count_per_split(create_split(images_dir).entries)

    assert all(count > 0 for class_counts in counts.values() for count in class_counts.values())


def test_entries_are_sorted_by_file(images_dir: Path):
    files = [entry.file for entry in create_split(images_dir).entries]

    assert files == sorted(files)


def test_duplicate_images_stay_in_the_same_split(images_dir: Path):
    for index in range(10):
        (images_dir / "paper" / f"paper{index}.jpg").write_bytes(b"duplicate content")
        (images_dir / "paper" / f"copy{index}.jpg").write_bytes(f"pair {index}".encode())
        (images_dir / "paper" / f"other{index}.jpg").write_bytes(f"pair {index}".encode())

    for seed in range(20):
        splits = split_of(create_split(images_dir, seed=seed).entries)
        assert len({splits[f"paper/paper{index}.jpg"] for index in range(10)}) == 1
        for index in range(10):
            assert splits[f"paper/copy{index}.jpg"] == splits[f"paper/other{index}.jpg"]


def test_identical_images_with_different_classes_are_excluded(images_dir: Path):
    (images_dir / "glass" / "glass0.jpg").write_bytes(b"same photo")
    (images_dir / "metal" / "metal0.jpg").write_bytes(b"same photo")

    result = create_split(images_dir)

    assert sorted(image.file for image in result.excluded) == [
        "glass/glass0.jpg",
        "metal/metal0.jpg",
    ]
    assert "glass/glass0.jpg" not in split_of(result.entries)
    assert "metal/metal0.jpg" not in split_of(result.entries)
    assert len(result.entries) == sum(IMAGES_PER_CLASS.values()) - 2


def test_split_rejects_class_with_too_few_images(tmp_path: Path):
    create_images(tmp_path, {**IMAGES_PER_CLASS, "trash": 3})

    with pytest.raises(SplitError, match="'trash' has too few images"):
        create_split(tmp_path)


def test_split_rejects_class_whose_duplicates_cannot_fill_every_split(tmp_path: Path):
    create_images(tmp_path, {**IMAGES_PER_CLASS, "trash": 6})
    for index in range(6):
        (tmp_path / "trash" / f"trash{index}.jpg").write_bytes(f"pair {index // 2}".encode())

    with pytest.raises(SplitError, match="'trash' has too few images"):
        create_split(tmp_path)


def test_write_split_csv_writes_header_and_rows(tmp_path: Path):
    entries = [
        SplitEntry("glass/glass1.jpg", "glass", "train"),
        SplitEntry("metal/metal1.jpg", "metal", "test"),
    ]
    split_path = tmp_path / "splits" / "split.csv"

    write_split_csv(entries, split_path)

    assert split_path.read_bytes() == (
        b"file,class,split\nglass/glass1.jpg,glass,train\nmetal/metal1.jpg,metal,test\n"
    )
    with split_path.open(newline="") as file:
        assert tuple(next(csv.reader(file))) == SPLIT_COLUMNS


def test_count_per_split_counts_every_class_and_split():
    entries = [
        SplitEntry("glass/a.jpg", "glass", "train"),
        SplitEntry("glass/b.jpg", "glass", "train"),
        SplitEntry("metal/c.jpg", "metal", "test"),
    ]

    counts = count_per_split(entries)

    assert counts["glass"] == {"train": 2, "validation": 0, "test": 0}
    assert counts["metal"] == {"train": 0, "validation": 0, "test": 1}
    assert set(counts) == set(CLASS_NAMES)


def test_split_does_not_depend_on_global_random_state(images_dir: Path):
    random.seed(1)
    first = create_split(images_dir)
    random.seed(2)
    second = create_split(images_dir)

    assert first == second
    assert Counter(entry.split for entry in first.entries)["train"] > 0
