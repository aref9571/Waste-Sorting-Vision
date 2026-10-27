import csv
from collections.abc import Sequence
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

PLOTTED_COLUMNS: tuple[str, ...] = (
    "train_loss",
    "validation_loss",
    "train_macro_f1",
    "validation_macro_f1",
)
METRICS_FILE_NAME = "metrics.csv"
CURVES_DPI = 100


def read_run_metrics(run_dir: Path) -> dict[str, list[float]]:
    path = run_dir / METRICS_FILE_NAME
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [name for name in PLOTTED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing the columns {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no epochs")
    return {name: [float(row[name]) for row in rows] for name in PLOTTED_COLUMNS}


def join_runs(runs: Sequence[dict[str, list[float]]]) -> dict[str, list[float]]:
    return {name: [value for run in runs for value in run[name]] for name in PLOTTED_COLUMNS}


def draw_run_boundaries(
    axis: Axes, run_dirs: Sequence[Path], runs: Sequence[dict[str, list[float]]]
) -> None:
    start = 0
    for run_dir, run in zip(run_dirs, runs, strict=True):
        if start > 0:
            axis.axvline(start + 0.5, color="grey", linestyle="--", linewidth=1)
        axis.text(
            start + 1,
            1.01,
            run_dir.name,
            transform=axis.get_xaxis_transform(),
            fontsize=8,
            color="grey",
            verticalalignment="bottom",
        )
        start += len(run["train_loss"])


def save_training_curves(run_dirs: Sequence[Path], output: Path) -> None:
    if not run_dirs:
        raise ValueError("No runs to plot")
    runs = [read_run_metrics(run_dir) for run_dir in run_dirs]
    joined = join_runs(runs)
    epochs = list(range(1, len(joined["train_loss"]) + 1))

    figure = Figure(figsize=(10, 4))
    loss_axis, f1_axis = figure.subplots(1, 2)
    panels = ((loss_axis, "loss", "Cross-entropy loss"), (f1_axis, "macro_f1", "Macro-F1"))
    for axis, metric, title in panels:
        axis.plot(epochs, joined[f"train_{metric}"], marker="o", markersize=3, label="train")
        axis.plot(
            epochs, joined[f"validation_{metric}"], marker="o", markersize=3, label="validation"
        )
        draw_run_boundaries(axis, run_dirs, runs)
        axis.set_title(title, pad=16)
        axis.set_xlabel("epoch (all runs in order)")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=CURVES_DPI)
