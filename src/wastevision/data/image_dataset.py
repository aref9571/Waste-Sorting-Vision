from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from wastevision.config import Settings
from wastevision.data.dataset import ImageRecord, load_image_records
from wastevision.data.transforms import evaluation_transforms, training_transforms
from wastevision.labels import CLASS_NAMES

ImageTransform = Callable[[Image.Image], torch.Tensor]


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class WasteImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: Sequence[ImageRecord], transform: ImageTransform) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = load_rgb_image(record.path)
        return self.transform(image), CLASS_NAMES.index(record.class_name)


def split_dataset(split_path: Path, images_dir: Path, split: str) -> WasteImageDataset:
    records = load_image_records(split_path, images_dir, split)
    transform = training_transforms() if split == "train" else evaluation_transforms()
    return WasteImageDataset(records, transform)


def make_data_loader(
    dataset: WasteImageDataset,
    batch_size: int,
    shuffle: bool,
    settings: Settings | None = None,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, int]]:
    settings = settings or Settings()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
