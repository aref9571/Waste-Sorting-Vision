from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from wastevision.evaluation.calibration import CalibrationBin, expected_calibration_error

PLOT_DPI = 100


def save_figure(figure: Figure, output: Path) -> None:
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=PLOT_DPI)


def save_reliability_diagram(bins: Sequence[CalibrationBin], output: Path) -> None:
    ece = expected_calibration_error(bins)
    lowers = [calibration_bin.lower for calibration_bin in bins]
    widths = [calibration_bin.upper - calibration_bin.lower for calibration_bin in bins]
    accuracies = [calibration_bin.accuracy for calibration_bin in bins]
    counts = [calibration_bin.count for calibration_bin in bins]

    figure = Figure(figsize=(5, 5))
    axis = figure.subplots()
    axis.bar(lowers, accuracies, width=widths, align="edge", edgecolor="black", label="accuracy")
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfect calibration")
    for lower, width, accuracy, count in zip(lowers, widths, accuracies, counts, strict=True):
        if count:
            axis.text(lower + width / 2, accuracy + 0.01, str(count), ha="center", fontsize=7)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("confidence (highest predicted probability)")
    axis.set_ylabel("accuracy in bin")
    axis.set_title(f"Reliability diagram, ECE = {ece:.3f}")
    axis.legend(loc="upper left")
    save_figure(figure, output)


def save_confusion_matrix(
    confusion: np.ndarray, class_names: Sequence[str], output: Path, title: str
) -> None:
    if confusion.shape != (len(class_names), len(class_names)):
        raise ValueError("The confusion matrix must have one row and one column per class")
    positions = range(len(class_names))

    figure = Figure(figsize=(6, 5.5))
    axis = figure.subplots()
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    threshold = confusion.max() / 2
    for row in positions:
        for column in positions:
            value = int(confusion[row, column])
            color = "white" if value > threshold else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=color)
    axis.set_xticks(positions, labels=class_names, rotation=45, ha="right")
    axis.set_yticks(positions, labels=class_names)
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    axis.set_title(title)
    save_figure(figure, output)
