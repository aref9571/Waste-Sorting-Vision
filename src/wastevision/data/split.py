import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from wastevision.data.download import IMAGE_SUFFIXES, sha256_of_file
from wastevision.labels import CLASS_NAMES

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
SPLIT_COLUMNS: tuple[str, ...] = ("file", "class", "split")
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_SEED = 42
DEFAULT_SPLIT_PATH = Path("splits/trashnet-split.csv")


class SplitError(Exception):
    pass


@dataclass(frozen=True)
class ImageFile:
    file: str
    class_name: str
    sha256: str


@dataclass(frozen=True)
class SplitEntry:
    file: str
    class_name: str
    split: str


@dataclass(frozen=True)
class SplitResult:
    entries: list[SplitEntry]
    excluded: list[ImageFile]


def list_images(images_dir: Path) -> list[ImageFile]:
    images = []
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        if not class_dir.is_dir():
            raise SplitError(f"Missing class folder {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(
                    ImageFile(
                        file=f"{class_name}/{path.name}",
                        class_name=class_name,
                        sha256=sha256_of_file(path),
                    )
                )
    return images


def group_by_content(images: list[ImageFile]) -> list[list[ImageFile]]:
    groups: dict[str, list[ImageFile]] = defaultdict(list)
    for image in images:
        groups[image.sha256].append(image)
    return list(groups.values())


def separate_conflicting_groups(
    groups: list[list[ImageFile]],
) -> tuple[list[list[ImageFile]], list[ImageFile]]:
    kept = []
    excluded = []
    for group in groups:
        if len({image.class_name for image in group}) > 1:
            excluded.extend(group)
        else:
            kept.append(group)
    return kept, excluded


def target_sizes(image_count: int) -> dict[str, int]:
    validation = round(image_count * VALIDATION_FRACTION)
    test = round(image_count * TEST_FRACTION)
    return {"train": image_count - validation - test, "validation": validation, "test": test}


def choose_split(sizes: dict[str, int], targets: dict[str, int], group_size: int) -> str:
    for split in ("test", "validation"):
        if sizes[split] + group_size <= targets[split]:
            return split
    return "train"


def assign_class_groups(
    groups: list[list[ImageFile]], rng: random.Random
) -> dict[str, list[ImageFile]]:
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    targets = target_sizes(sum(len(group) for group in groups))
    assigned: dict[str, list[ImageFile]] = {split: [] for split in SPLIT_NAMES}
    for group in shuffled:
        sizes = {split: len(members) for split, members in assigned.items()}
        assigned[choose_split(sizes, targets, len(group))].extend(group)
    return assigned


def create_split(images_dir: Path, seed: int = SPLIT_SEED) -> SplitResult:
    groups, excluded = separate_conflicting_groups(group_by_content(list_images(images_dir)))
    rng = random.Random(seed)
    entries = []
    for class_name in CLASS_NAMES:
        class_groups = [group for group in groups if group[0].class_name == class_name]
        assigned = assign_class_groups(class_groups, rng)
        if not all(assigned.values()):
            image_count = sum(len(group) for group in class_groups)
            raise SplitError(
                f"Class {class_name!r} has too few images ({image_count}) to fill every split"
            )
        for split, members in assigned.items():
            entries.extend(SplitEntry(image.file, class_name, split) for image in members)
    return SplitResult(entries=sorted(entries, key=lambda entry: entry.file), excluded=excluded)


def write_split_csv(entries: list[SplitEntry], split_path: Path) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(SPLIT_COLUMNS)
        for entry in entries:
            writer.writerow((entry.file, entry.class_name, entry.split))


def count_per_split(entries: list[SplitEntry]) -> dict[str, dict[str, int]]:
    counts = {class_name: dict.fromkeys(SPLIT_NAMES, 0) for class_name in CLASS_NAMES}
    for entry in entries:
        counts[entry.class_name][entry.split] += 1
    return counts
