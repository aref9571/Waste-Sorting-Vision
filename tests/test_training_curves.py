from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from wastevision.training.curves import join_runs, read_run_metrics, save_training_curves

HEADER = (
    "epoch,train_loss,train_accuracy,train_macro_f1,"
    "validation_loss,validation_accuracy,validation_macro_f1,seconds"
)


def write_run(run_dir: Path, rows: list[str]) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return run_dir


def test_read_run_metrics(tmp_path: Path):
    run_dir = write_run(
        tmp_path / "head",
        ["1,0.9,0.6,0.5,0.8,0.7,0.6,10.0", "2,0.5,0.8,0.75,0.6,0.8,0.7,9.5"],
    )

    metrics = read_run_metrics(run_dir)

    assert metrics == {
        "train_loss": [0.9, 0.5],
        "validation_loss": [0.8, 0.6],
        "train_macro_f1": [0.5, 0.75],
        "validation_macro_f1": [0.6, 0.7],
    }


def test_join_runs_keeps_the_order_of_the_runs():
    first = {"train_loss": [3.0, 2.0], "validation_loss": [3.5, 2.5]}
    second = {"train_loss": [1.0], "validation_loss": [1.5]}
    first |= {"train_macro_f1": [0.1, 0.2], "validation_macro_f1": [0.1, 0.2]}
    second |= {"train_macro_f1": [0.3], "validation_macro_f1": [0.3]}

    joined = join_runs([first, second])

    assert joined["train_loss"] == [3.0, 2.0, 1.0]
    assert joined["validation_macro_f1"] == [0.1, 0.2, 0.3]


def test_read_run_metrics_rejects_missing_columns(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("epoch,train_loss\n1,0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="validation_loss"):
        read_run_metrics(run_dir)


def test_read_run_metrics_rejects_a_run_without_epochs(tmp_path: Path):
    run_dir = write_run(tmp_path / "run", [])

    with pytest.raises(ValueError, match="no epochs"):
        read_run_metrics(run_dir)


def test_read_run_metrics_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_run_metrics(tmp_path)


def test_save_training_curves_writes_a_reproducible_png(tmp_path: Path):
    head = write_run(tmp_path / "head", ["1,0.9,0.6,0.5,0.8,0.7,0.6,10.0"])
    full = write_run(tmp_path / "full", ["1,0.4,0.9,0.85,0.5,0.85,0.8,30.0"])
    first = tmp_path / "out" / "curves.png"
    second = tmp_path / "curves-again.png"

    save_training_curves([head, full], first)
    save_training_curves([head, full], second)

    assert first.read_bytes().startswith(b"\x89PNG")
    assert first.read_bytes() == second.read_bytes()


def test_save_training_curves_needs_a_run(tmp_path: Path):
    with pytest.raises(ValueError, match="No runs"):
        save_training_curves([], tmp_path / "curves.png")
