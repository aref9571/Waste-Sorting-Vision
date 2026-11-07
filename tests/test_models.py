import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE
from wastevision.data.download import DATASET_SHA256, sha256_of_file
from wastevision.data.split import DEFAULT_SPLIT_PATH
from wastevision.labels import CATEGORY_BY_CLASS, CLASS_NAMES

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "waste-mobilenetv3.onnx"
SPLIT_PATH = Path(__file__).resolve().parents[1] / DEFAULT_SPLIT_PATH


def read_json(name: str):
    return json.loads((MODELS_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def session() -> ort.InferenceSession:
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_committed_model_returns_one_score_per_class(
    session: ort.InferenceSession, batch_size: int
):
    images = np.random.default_rng(0).standard_normal(
        (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
    )

    (logits,) = session.run(["logits"], {"image": images})

    assert logits.shape == (batch_size, len(CLASS_NAMES))
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()


def test_committed_model_declares_its_input_and_output(session: ort.InferenceSession):
    (model_input,) = session.get_inputs()
    (model_output,) = session.get_outputs()

    assert model_input.name == "image"
    assert model_input.shape[1:] == [3, IMAGE_SIZE, IMAGE_SIZE]
    assert model_output.name == "logits"
    assert model_output.shape[1:] == [len(CLASS_NAMES)]


def test_labels_file_matches_the_class_list():
    labels = read_json("labels.json")

    assert [entry["class"] for entry in labels] == list(CLASS_NAMES)
    assert [entry["index"] for entry in labels] == list(range(len(CLASS_NAMES)))
    assert all(entry["category"] == CATEGORY_BY_CLASS[entry["class"]] for entry in labels)


def test_metadata_matches_the_model_and_the_preprocessing():
    metadata = read_json("metadata.json")

    assert metadata["model_file"] == MODEL_PATH.name
    assert metadata["model_sha256"] == sha256_of_file(MODEL_PATH)
    assert metadata["class_names"] == list(CLASS_NAMES)
    assert metadata["input"]["resize"] == RESIZE_SIZE
    assert metadata["input"]["crop"] == IMAGE_SIZE
    assert metadata["input"]["mean"] == list(IMAGENET_MEAN)
    assert metadata["input"]["std"] == list(IMAGENET_STD)
    assert metadata["dataset_sha256"] == DATASET_SHA256
    assert metadata["split_sha256"] == sha256_of_file(SPLIT_PATH)
    assert metadata["parity"]["max_abs_difference"] < 1e-4
    assert metadata["parity"]["top1_matches"] == metadata["parity"]["images"]
