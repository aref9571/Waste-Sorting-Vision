import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from wastevision.evaluation.threshold import METADATA_KEY
from wastevision.inference.preprocess import Preprocessing, preprocess_image
from wastevision.labels import category_for

DEFAULT_MODELS_DIR = Path("models")
MODEL_FILE_NAME = "waste-mobilenetv3.onnx"
METADATA_FILE_NAME = "metadata.json"
TOP_COUNT = 3


@dataclass(frozen=True)
class Prediction:
    top: tuple[tuple[str, float], ...]
    category: str
    unsure: bool

    @property
    def label(self) -> str:
        return self.top[0][0]

    @property
    def confidence(self) -> float:
        return self.top[0][1]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Predictor:
    def __init__(
        self,
        model_path: Path,
        metadata: dict[str, Any],
        threshold: float | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model {model_path} does not exist")
        self.class_names: tuple[str, ...] = tuple(metadata["class_names"])
        self.preprocessing = Preprocessing.from_metadata(metadata)
        self.threshold = self.resolve_threshold(metadata, threshold)
        self.input_name: str = metadata["input"]["name"]
        self.output_name: str = metadata["output"]["name"]
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @staticmethod
    def resolve_threshold(metadata: dict[str, Any], threshold: float | None) -> float:
        if threshold is None and METADATA_KEY not in metadata:
            raise ValueError("The metadata has no confidence threshold")
        value = threshold if threshold is not None else metadata[METADATA_KEY]["threshold"]
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return float(value)

    @classmethod
    def from_directory(cls, models_dir: Path = DEFAULT_MODELS_DIR) -> "Predictor":
        metadata = load_metadata(models_dir / METADATA_FILE_NAME)
        return cls(models_dir / metadata.get("model_file", MODEL_FILE_NAME), metadata)

    def probabilities(self, image: Image.Image) -> np.ndarray:
        batch = preprocess_image(image, self.preprocessing)[np.newaxis]
        (logits,) = self.session.run([self.output_name], {self.input_name: batch})
        return softmax(logits[0].astype(np.float64))

    def predict(self, image: Image.Image) -> Prediction:
        probabilities = self.probabilities(image)
        order = np.argsort(-probabilities, kind="stable")[:TOP_COUNT]
        top = tuple((self.class_names[index], float(probabilities[index])) for index in order)
        return Prediction(
            top=top,
            category=category_for(top[0][0]),
            unsure=top[0][1] < self.threshold,
        )
