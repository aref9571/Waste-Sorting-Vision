from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassScores:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassScores]
    confusion: np.ndarray


def check_label_indices(
    labels: Sequence[int], predictions: Sequence[int], num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if true.ndim != 1 or true.shape != predicted.shape:
        raise ValueError("labels and predictions must be flat sequences of the same length")
    if len(true) == 0:
        raise ValueError("Cannot compute metrics without any labels")
    for values in (true, predicted):
        if values.min() < 0 or values.max() >= num_classes:
            raise ValueError(f"Class indices must be between 0 and {num_classes - 1}")
    return true, predicted


def confusion_matrix(
    labels: Sequence[int], predictions: Sequence[int], num_classes: int
) -> np.ndarray:
    true, predicted = check_label_indices(labels, predictions, num_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros(len(numerator), dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def classification_report(
    labels: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> ClassificationReport:
    confusion = confusion_matrix(labels, predictions, len(class_names))
    true_positives = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    precision = safe_divide(true_positives, predicted_counts)
    recall = safe_divide(true_positives, support)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    per_class = {
        class_name: ClassScores(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, class_name in enumerate(class_names)
    }
    return ClassificationReport(
        accuracy=float(true_positives.sum() / confusion.sum()),
        macro_f1=float(f1.mean()),
        per_class=per_class,
        confusion=confusion,
    )
