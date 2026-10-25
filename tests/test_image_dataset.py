from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from wastevision.config import IMAGE_SIZE, Settings
from wastevision.data.dataset import ImageRecord, SplitFileError
from wastevision.data.image_dataset import (
    WasteImageDataset,
    load_rgb_image,
    make_data_loader,
    split_dataset,
)
from wastevision.data.transforms import evaluation_transforms
from wastevision.labels import CLASS_NAMES


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_load_rgb_image_converts_every_mode(tmp_path: Path, mode: str):
    path = tmp_path / "image.png"
    Image.new(mode, (40, 30)).save(path)

    image = load_rgb_image(path)

    assert image.mode == "RGB"
    assert image.size == (40, 30)


def test_dataset_returns_tensor_and_class_index(tmp_path: Path):
    path = tmp_path / "metal.png"
    Image.new("L", (64, 48), 128).save(path)
    dataset = WasteImageDataset([ImageRecord(path, "metal", "test")], evaluation_transforms())

    tensor, label = dataset[0]

    assert len(dataset) == 1
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    assert label == CLASS_NAMES.index("metal")


def test_dataset_raises_for_missing_image(tmp_path: Path):
    dataset = WasteImageDataset(
        [ImageRecord(tmp_path / "missing.png", "glass", "train")], evaluation_transforms()
    )

    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_split_dataset_reads_only_the_requested_split(tiny_dataset: tuple[Path, Path]):
    images_dir, split_path = tiny_dataset

    train = split_dataset(split_path, images_dir, "train")
    validation = split_dataset(split_path, images_dir, "validation")

    assert len(train) == 3 * len(CLASS_NAMES)
    assert len(validation) == len(CLASS_NAMES)
    assert {record.split for record in train.records} == {"train"}


def test_split_dataset_augments_only_training_images(tiny_dataset: tuple[Path, Path]):
    images_dir, split_path = tiny_dataset
    train = split_dataset(split_path, images_dir, "train")
    test = split_dataset(split_path, images_dir, "test")
    torch.manual_seed(0)

    assert not torch.equal(train[0][0], train[0][0])
    assert torch.equal(test[0][0], test[0][0])


def test_split_dataset_rejects_unknown_split(tiny_dataset: tuple[Path, Path]):
    images_dir, split_path = tiny_dataset

    with pytest.raises(ValueError, match="Unknown split"):
        split_dataset(split_path, images_dir, "holdout")


def test_split_dataset_reports_invalid_split_file(tmp_path: Path):
    split_path = tmp_path / "split.csv"
    split_path.write_text("file,class,split\nx.png,bottle,train\n", encoding="utf-8")

    with pytest.raises(SplitFileError, match="unknown class"):
        split_dataset(split_path, tmp_path, "train")


def test_data_loader_returns_batches(tiny_dataset: tuple[Path, Path]):
    images_dir, split_path = tiny_dataset
    dataset = split_dataset(split_path, images_dir, "validation")

    images, labels = next(iter(make_data_loader(dataset, batch_size=4, shuffle=False)))

    assert images.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert labels.dtype == torch.int64
    assert labels.tolist() == [0, 1, 2, 3]


def test_data_loader_uses_num_workers_from_settings(tiny_dataset: tuple[Path, Path]):
    images_dir, split_path = tiny_dataset
    dataset = split_dataset(split_path, images_dir, "test")

    loader = make_data_loader(
        dataset, batch_size=2, shuffle=False, settings=Settings(num_workers=2)
    )

    assert loader.num_workers == 2


def test_data_loader_shuffle_is_reproducible_with_a_seed(tiny_dataset: tuple[Path, Path]):
    images_dir, split_path = tiny_dataset
    dataset = split_dataset(split_path, images_dir, "test")

    def label_order(seed: int) -> list[int]:
        loader = make_data_loader(dataset, batch_size=6, shuffle=True, seed=seed)
        return next(iter(loader))[1].tolist()

    assert label_order(1) == label_order(1)
    assert sorted(label_order(1)) == list(range(len(CLASS_NAMES)))
    assert any(label_order(seed) != label_order(1) for seed in range(2, 6))
