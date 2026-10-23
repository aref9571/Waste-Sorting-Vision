from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wastevision.baseline.features import extract_features_from_files
from wastevision.data.dataset import load_image_records
from wastevision.labels import CLASS_NAMES

PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {"C": [0.001, 0.01, 0.1, 1.0]},
    "svm_rbf": {"C": [1.0, 10.0, 100.0], "gamma": ["scale", 0.0001]},
}


class BaselineError(Exception):
    pass


@dataclass(frozen=True)
class ClassScores:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassScores]


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    params: dict[str, Any]
    validation: EvaluationResult


@dataclass(frozen=True)
class SelectionResult:
    pipeline: Pipeline
    best: CandidateResult
    candidates: list[CandidateResult]


def build_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=5000, **params)
    elif model_name == "svm_rbf":
        model = SVC(kernel="rbf", **params)
    else:
        raise ValueError(f"Unknown model {model_name!r}")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def evaluate(pipeline: Pipeline, features: np.ndarray, labels: np.ndarray) -> EvaluationResult:
    predictions = pipeline.predict(features)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(CLASS_NAMES), zero_division=0
    )
    per_class = {
        class_name: ClassScores(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, class_name in enumerate(CLASS_NAMES)
    }
    return EvaluationResult(
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(
            f1_score(
                labels, predictions, labels=list(CLASS_NAMES), average="macro", zero_division=0
            )
        ),
        per_class=per_class,
    )


def select_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> SelectionResult:
    candidates = []
    best_pipeline = None
    best_result = None
    for model_name, grid in PARAMETER_GRIDS.items():
        for params in ParameterGrid(grid):
            pipeline = build_pipeline(model_name, params)
            pipeline.fit(train_features, train_labels)
            result = CandidateResult(
                model_name, params, evaluate(pipeline, validation_features, validation_labels)
            )
            candidates.append(result)
            if best_result is None or result.validation.macro_f1 > best_result.validation.macro_f1:
                best_pipeline, best_result = pipeline, result
    return SelectionResult(pipeline=best_pipeline, best=best_result, candidates=candidates)


def load_split_features(
    split_path: Path, images_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray]:
    records = load_image_records(split_path, images_dir, split=split)
    if not records:
        raise BaselineError(f"No images in the {split} split of {split_path}")
    features = extract_features_from_files(record.path for record in records)
    labels = np.array([record.class_name for record in records])
    return features, labels


def save_model(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    if not path.is_file():
        raise BaselineError(f"No baseline model at {path}. Run 'wastevision baseline train' first.")
    return joblib.load(path)
