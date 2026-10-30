from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusedPair:
    true_class: int
    predicted_class: int
    count: int


def check_predictions(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or labels.shape != (probabilities.shape[0],):
        raise ValueError("Expected one label per row of probabilities")


def check_count(count: int) -> None:
    if count < 1:
        raise ValueError("count must be at least 1")


def worst_mistakes(labels: np.ndarray, probabilities: np.ndarray, count: int) -> list[int]:
    check_predictions(labels, probabilities)
    check_count(count)
    confidences = probabilities.max(axis=1)
    wrong = np.flatnonzero(probabilities.argmax(axis=1) != labels)
    by_confidence = wrong[np.argsort(-confidences[wrong], kind="stable")]
    return by_confidence[:count].tolist()


def most_confident_correct(labels: np.ndarray, probabilities: np.ndarray, class_index: int) -> int:
    confidences = probabilities.max(axis=1)
    correct = np.flatnonzero((probabilities.argmax(axis=1) == labels) & (labels == class_index))
    if correct.size == 0:
        raise ValueError(f"No correct prediction for class {class_index}")
    return int(correct[np.argmax(confidences[correct])])


def gradcam_pairs(
    labels: np.ndarray, probabilities: np.ndarray, count: int
) -> list[tuple[int, int]]:
    check_predictions(labels, probabilities)
    check_count(count)
    pairs: list[tuple[int, int]] = []
    used_classes: set[int] = set()
    for mistake in worst_mistakes(labels, probabilities, len(labels)):
        true_class = int(labels[mistake])
        if true_class in used_classes:
            continue
        try:
            correct = most_confident_correct(labels, probabilities, true_class)
        except ValueError:
            continue
        pairs.append((correct, mistake))
        used_classes.add(true_class)
        if len(pairs) == count:
            break
    return pairs


def most_confused_pairs(confusion: np.ndarray, count: int) -> list[ConfusedPair]:
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("The confusion matrix must be square")
    check_count(count)
    pairs = [
        ConfusedPair(true_class, predicted_class, int(confusion[true_class, predicted_class]))
        for true_class in range(confusion.shape[0])
        for predicted_class in range(confusion.shape[1])
        if true_class != predicted_class and confusion[true_class, predicted_class] > 0
    ]
    pairs.sort(key=lambda pair: pair.count, reverse=True)
    return pairs[:count]
