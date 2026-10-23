import csv
from dataclasses import dataclass
from pathlib import Path

from wastevision.data.split import SPLIT_COLUMNS, SPLIT_NAMES
from wastevision.labels import CLASS_NAMES


class SplitFileError(Exception):
    pass


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str
    split: str


def parse_row(row: dict[str, str], images_dir: Path, line_number: int) -> ImageRecord:
    class_name = row["class"]
    split = row["split"]
    if class_name not in CLASS_NAMES:
        raise SplitFileError(f"Line {line_number}: unknown class {class_name!r}")
    if split not in SPLIT_NAMES:
        raise SplitFileError(f"Line {line_number}: unknown split {split!r}")
    return ImageRecord(path=images_dir / row["file"], class_name=class_name, split=split)


def load_image_records(
    split_path: Path, images_dir: Path, split: str | None = None
) -> list[ImageRecord]:
    if split is not None and split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}, expected one of {SPLIT_NAMES}")
    with split_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
            raise SplitFileError(f"{split_path} must have the columns {','.join(SPLIT_COLUMNS)}")
        records = [
            parse_row(row, images_dir, line_number)
            for line_number, row in enumerate(reader, start=2)
        ]
    if split is None:
        return records
    return [record for record in records if record.split == split]
