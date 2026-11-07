import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.export import Dim

from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE
from wastevision.data.download import DATASET_SHA256, sha256_of_file
from wastevision.data.image_dataset import WasteImageDataset, make_data_loader, split_dataset
from wastevision.evaluation.metrics import classification_report
from wastevision.evaluation.predictions import predict_probabilities
from wastevision.labels import CLASS_NAMES, category_for
from wastevision.training.loop import load_checkpoint
from wastevision.training.model import build_model

DEFAULT_MODELS_DIR = Path("models")
MODEL_FILE_NAME = "waste-mobilenetv3.onnx"
LABELS_FILE_NAME = "labels.json"
METADATA_FILE_NAME = "metadata.json"
ARCHITECTURE = "mobilenet_v3_small"
OPSET_VERSION = 18
INPUT_NAME = "image"
OUTPUT_NAME = "logits"
PARITY_IMAGE_COUNT = 50
PARITY_TOLERANCE = 1e-4
EVALUATION_BATCH_SIZE = 32


class ParityError(ValueError):
    pass


@dataclass(frozen=True)
class ParityResult:
    images: int
    max_abs_difference: float
    top1_matches: int

    @property
    def passed(self) -> bool:
        return self.max_abs_difference < PARITY_TOLERANCE and self.top1_matches == self.images


@dataclass(frozen=True)
class ExportResult:
    model_path: Path
    labels_path: Path
    metadata_path: Path
    parity: ParityResult
    metadata: dict[str, Any]


def export_onnx(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        (example,),
        path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: Dim("batch")},),
        opset_version=OPSET_VERSION,
        external_data=False,
        verbose=False,
    )
    strip_debug_metadata(path)


def strip_debug_metadata(path: Path) -> None:
    exported = onnx.load(path)
    exported.doc_string = ""
    for node in exported.graph.node:
        node.doc_string = ""
        del node.metadata_props[:]
    onnx.save(exported, path)


def torch_logits(model: nn.Module, images: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(images).numpy()


def onnx_logits(model_path: Path, images: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session.run([OUTPUT_NAME], {INPUT_NAME: images})[0]


def check_parity(model: nn.Module, model_path: Path, images: torch.Tensor) -> ParityResult:
    if len(images) == 0:
        raise ValueError("Cannot check parity without any images")
    expected = torch_logits(model, images)
    actual = onnx_logits(model_path, images.numpy())
    return ParityResult(
        images=len(images),
        max_abs_difference=float(np.abs(actual - expected).max()),
        top1_matches=int((actual.argmax(axis=1) == expected.argmax(axis=1)).sum()),
    )


def first_images(dataset: WasteImageDataset, count: int) -> torch.Tensor:
    return torch.stack([dataset[index][0] for index in range(min(count, len(dataset)))])


def split_metrics(model: nn.Module, dataset: WasteImageDataset) -> dict[str, float | int]:
    loader = make_data_loader(dataset, EVALUATION_BATCH_SIZE, shuffle=False)
    predictions = predict_probabilities(model, loader)
    report = classification_report(
        predictions.labels.tolist(), predictions.predicted.tolist(), CLASS_NAMES
    )
    return {
        "images": len(predictions.labels),
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
    }


def labels_document() -> list[dict[str, Any]]:
    return [
        {"index": index, "class": class_name, "category": category_for(class_name)}
        for index, class_name in enumerate(CLASS_NAMES)
    ]


def build_metadata(
    model_path: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    split_path: Path,
    test_metrics: dict[str, float | int],
    parity: ParityResult,
    exported_on: date,
) -> dict[str, Any]:
    validation = checkpoint["validation"]
    return {
        "model_file": model_path.name,
        "model_sha256": sha256_of_file(model_path),
        "architecture": ARCHITECTURE,
        "opset": OPSET_VERSION,
        "input": {
            "name": INPUT_NAME,
            "layout": "NCHW",
            "color": "RGB",
            "resize": RESIZE_SIZE,
            "crop": IMAGE_SIZE,
            "value_range": [0.0, 1.0],
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "output": {"name": OUTPUT_NAME},
        "class_names": list(CLASS_NAMES),
        "dataset_sha256": DATASET_SHA256,
        "split_sha256": sha256_of_file(split_path),
        "checkpoint_sha256": sha256_of_file(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "metrics": {
            "validation": {
                "accuracy": validation["accuracy"],
                "macro_f1": validation["macro_f1"],
            },
            "test": test_metrics,
        },
        "parity": asdict(parity),
        "exported_on": exported_on.isoformat(),
    }


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def export_model(
    checkpoint_path: Path,
    images_dir: Path,
    split_path: Path,
    models_dir: Path = DEFAULT_MODELS_DIR,
    exported_on: date | None = None,
) -> ExportResult:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
    model = build_model(pretrained=False)
    checkpoint = load_checkpoint(model, checkpoint_path)
    model.eval()
    test_dataset = split_dataset(split_path, images_dir, "test")

    model_path = models_dir / MODEL_FILE_NAME
    export_onnx(model, model_path)
    parity = check_parity(model, model_path, first_images(test_dataset, PARITY_IMAGE_COUNT))
    if not parity.passed:
        model_path.unlink()
        raise ParityError(
            f"ONNX model differs from PyTorch: max difference {parity.max_abs_difference:.2e},"
            f" top-1 equal for {parity.top1_matches} of {parity.images} images"
        )

    metadata = build_metadata(
        model_path,
        checkpoint_path,
        checkpoint,
        split_path,
        split_metrics(model, test_dataset),
        parity,
        exported_on or date.today(),
    )
    labels_path = models_dir / LABELS_FILE_NAME
    metadata_path = models_dir / METADATA_FILE_NAME
    write_json(labels_path, labels_document())
    write_json(metadata_path, metadata)
    return ExportResult(model_path, labels_path, metadata_path, parity, metadata)
