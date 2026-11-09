import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TARGET_ACCURACY = 0.95
METADATA_KEY = "confidence_threshold"


class ThresholdError(ValueError):
    pass


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    target_accuracy: float
    images: int
    answered: int
    coverage: float
    accuracy: float


def coverage_and_accuracy(
    confidences: Sequence[float], correct: Sequence[bool], threshold: float
) -> tuple[int, float, float]:
    confidence_values = np.asarray(confidences, dtype=np.float64)
    correct_values = np.asarray(correct, dtype=bool)
    if confidence_values.ndim != 1 or confidence_values.shape != correct_values.shape:
        raise ThresholdError("confidences and correct must be flat sequences of the same length")
    if len(confidence_values) == 0:
        raise ThresholdError("Cannot measure coverage without any predictions")
    answered = confidence_values >= threshold
    count = int(answered.sum())
    coverage = count / len(confidence_values)
    accuracy = float(correct_values[answered].mean()) if count else 0.0
    return count, coverage, accuracy


def select_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
) -> ThresholdChoice:
    if not 0.0 < target_accuracy <= 1.0:
        raise ThresholdError("target_accuracy must be in (0, 1]")
    confidence_values = np.asarray(confidences, dtype=np.float64)
    coverage_and_accuracy(confidences, correct, 0.0)
    for candidate in np.unique(confidence_values):
        answered, coverage, accuracy = coverage_and_accuracy(confidences, correct, candidate)
        if accuracy >= target_accuracy:
            return ThresholdChoice(
                threshold=float(candidate),
                target_accuracy=target_accuracy,
                images=len(confidence_values),
                answered=answered,
                coverage=coverage,
                accuracy=accuracy,
            )
    raise ThresholdError(f"No threshold reaches an accuracy of {target_accuracy}")


def add_threshold_to_metadata(metadata_path: Path, choice: ThresholdChoice) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[METADATA_KEY] = asdict(choice)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
