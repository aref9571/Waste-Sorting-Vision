from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_NUM_BINS = 10


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def bin_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, num_bins + 1)


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], num_bins: int = DEFAULT_NUM_BINS
) -> list[CalibrationBin]:
    confidence_values = np.asarray(confidences, dtype=np.float64)
    correct_values = np.asarray(correct, dtype=bool)
    if confidence_values.ndim != 1 or confidence_values.shape != correct_values.shape:
        raise ValueError("confidences and correct must be flat sequences of the same length")
    if len(confidence_values) == 0:
        raise ValueError("Cannot measure calibration without any predictions")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    if not np.all((confidence_values >= 0.0) & (confidence_values <= 1.0)):
        raise ValueError("Confidences must be between 0 and 1")

    edges = bin_edges(num_bins)
    indices = np.digitize(confidence_values, edges[1:-1], right=True)
    bins = []
    for index in range(num_bins):
        in_bin = indices == index
        count = int(in_bin.sum())
        bins.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_confidence=float(confidence_values[in_bin].mean()) if count else 0.0,
                accuracy=float(correct_values[in_bin].mean()) if count else 0.0,
            )
        )
    return bins


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float:
    total = sum(calibration_bin.count for calibration_bin in bins)
    if total == 0:
        raise ValueError("Cannot measure calibration without any predictions")
    return sum(
        calibration_bin.count
        / total
        * abs(calibration_bin.accuracy - calibration_bin.mean_confidence)
        for calibration_bin in bins
    )
