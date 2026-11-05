import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wastevision import __version__, cli
from wastevision.cli import app
from wastevision.data.download import EXPECTED_IMAGE_COUNTS, ChecksumMismatchError, sha256_of_file
from wastevision.labels import CLASS_NAMES

runner = CliRunner()


def test_version_command_prints_package_version():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_unknown_command_fails():
    result = runner.invoke(app, ["does-not-exist"])

    assert result.exit_code != 0


def create_class_folders(images_dir: Path, images_per_class: dict[str, int]) -> None:
    for class_name, count in images_per_class.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(count):
            (class_dir / f"{class_name}{index}.jpg").write_bytes(b"x")


def test_data_download_prints_counts_per_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    images_dir = tmp_path / "images"
    create_class_folders(images_dir, dict.fromkeys(CLASS_NAMES, 2))
    received_dirs = []

    def fake_download_dataset(raw_dir: Path) -> Path:
        received_dirs.append(raw_dir)
        return images_dir

    monkeypatch.setattr(cli, "download_dataset", fake_download_dataset)

    result = runner.invoke(app, ["data", "download", "--raw-dir", str(tmp_path / "raw")])

    assert result.exit_code == 0
    assert received_dirs == [tmp_path / "raw"]
    assert "glass: 2" in result.stdout
    assert "total: 12" in result.stdout
    assert "Warning: glass has 2 images, expected 501." in result.stdout


def test_data_download_reports_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def failing_download_dataset(raw_dir: Path) -> Path:
        raise ChecksumMismatchError("bad checksum")

    monkeypatch.setattr(cli, "download_dataset", failing_download_dataset)

    result = runner.invoke(app, ["data", "download", "--raw-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "Error: bad checksum" in result.stderr


def test_data_download_confirms_published_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "download_dataset", lambda raw_dir: tmp_path)
    monkeypatch.setattr(cli, "count_images_per_class", lambda images_dir: EXPECTED_IMAGE_COUNTS)

    result = runner.invoke(app, ["data", "download", "--raw-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "total: 2527" in result.stdout
    assert "Counts match the published TrashNet counts." in result.stdout
    assert "Warning" not in result.stdout


def create_distinct_images(images_dir: Path, images_per_class: int) -> None:
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(images_per_class):
            (class_dir / f"{class_name}{index}.jpg").write_text(f"{class_name}{index}")


def test_data_split_writes_csv_and_prints_counts(tmp_path: Path):
    images_dir = tmp_path / "images"
    create_distinct_images(images_dir, 20)
    output = tmp_path / "splits" / "split.csv"

    result = runner.invoke(
        app, ["data", "split", "--images-dir", str(images_dir), "--output", str(output)]
    )

    assert result.exit_code == 0
    assert "glass" in result.stdout
    assert f"Wrote 120 images to {output}" in result.stdout
    assert f"SHA-256: {sha256_of_file(output)}" in result.stdout
    assert len(output.read_text().splitlines()) == 121


def test_data_split_reports_excluded_images(tmp_path: Path):
    images_dir = tmp_path / "images"
    create_distinct_images(images_dir, 20)
    (images_dir / "metal" / "metal0.jpg").write_text("glass0")

    result = runner.invoke(
        app,
        ["data", "split", "--images-dir", str(images_dir), "--output", str(tmp_path / "s.csv")],
    )

    assert result.exit_code == 0
    assert "Excluded glass/glass0.jpg: identical image with another class label." in result.stdout
    assert "Excluded metal/metal0.jpg: identical image with another class label." in result.stdout
    assert "Wrote 118 images" in result.stdout


def test_data_split_reports_errors(tmp_path: Path):
    result = runner.invoke(
        app, ["data", "split", "--images-dir", str(tmp_path), "--output", str(tmp_path / "s.csv")]
    )

    assert result.exit_code == 1
    assert "Error: Missing class folder" in result.stderr


def test_cli_import_does_not_load_training_libraries():
    libraries = ("sklearn", "skimage", "torch", "torchvision", "matplotlib")
    code = f"import sys, wastevision.cli; print([name in sys.modules for name in {libraries!r}])"

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == str([False] * len(libraries))


def test_baseline_train_then_evaluate(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset
    model_path = tmp_path / "runs" / "model.joblib"
    options = [
        "--images-dir",
        str(images_dir),
        "--split",
        str(split_path),
        "--model",
        str(model_path),
    ]

    train_result = runner.invoke(app, ["baseline", "train", *options])

    assert train_result.exit_code == 0
    assert "logistic_regression" in train_result.stdout
    assert "svm_rbf" in train_result.stdout
    assert "Best on validation:" in train_result.stdout
    assert f"Saved pipeline to {model_path}" in train_result.stdout
    assert model_path.is_file()

    evaluate_result = runner.invoke(app, ["baseline", "evaluate", *options])

    assert evaluate_result.exit_code == 0
    assert "Test split (6 images):" in evaluate_result.stdout
    assert "macro-F1:" in evaluate_result.stdout
    assert "trash" in evaluate_result.stdout


def test_baseline_train_reports_errors(tmp_path: Path):
    result = runner.invoke(app, ["baseline", "train", "--split", str(tmp_path / "missing.csv")])

    assert result.exit_code == 1
    assert "Error:" in result.stderr


def test_baseline_evaluate_reports_missing_model(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    images_dir, split_path = tiny_dataset

    result = runner.invoke(
        app,
        [
            "baseline",
            "evaluate",
            "--images-dir",
            str(images_dir),
            "--split",
            str(split_path),
            "--model",
            str(tmp_path / "missing.joblib"),
        ],
    )

    assert result.exit_code == 1
    assert "Run 'wastevision baseline train' first." in result.stderr


def test_data_preview_writes_augmentation_grid(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    images_dir, split_path = tiny_dataset
    output = tmp_path / "augmentations.png"

    result = runner.invoke(
        app,
        [
            "data",
            "preview",
            "--images-dir",
            str(images_dir),
            "--split",
            str(split_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert f"Wrote augmentation preview of {len(CLASS_NAMES)} images" in result.stdout


def test_data_preview_reports_errors(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")

    result = runner.invoke(app, ["data", "preview", "--split", str(tmp_path / "missing.csv")])

    assert result.exit_code == 1
    assert "Error:" in result.stderr


def fake_fit_result(run_dir: Path, epochs: int):
    from wastevision.training.loop import EpochMetrics, EpochRecord, FitResult

    metrics = EpochMetrics(loss=0.5, accuracy=0.8, macro_f1=0.75)
    history = [EpochRecord(epoch, metrics, metrics, 1.0) for epoch in range(1, epochs + 1)]
    return FitResult(history, 1, metrics, run_dir / "best.pt")


def test_train_head_uses_the_stage_defaults(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("torch")
    calls = []

    def fake_train_head(images_dir, split_path, run_dir, **options):
        calls.append((run_dir, options))
        options["report"](fake_fit_result(run_dir, 1).history[0])
        return fake_fit_result(run_dir, options["epochs"])

    monkeypatch.setattr("wastevision.training.train.train_head", fake_train_head)

    result = runner.invoke(app, ["train", "--stage", "head"])

    assert result.exit_code == 0, result.output
    run_dir, options = calls[0]
    assert run_dir == Path("runs/head")
    assert options["epochs"] == 8
    assert options["use_class_weights"] is True
    assert "epoch   1" in result.stdout
    assert "Best epoch 1: val accuracy 0.8000, val macro-F1 0.7500" in result.stdout
    assert "Stopped early" not in result.stdout
    assert f"Saved checkpoint to {Path('runs/head/best.pt')}" in result.stdout


def test_train_full_passes_its_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pytest.importorskip("torch")
    calls = []

    def fake_train_full(images_dir, split_path, head_checkpoint, run_dir, **options):
        calls.append((head_checkpoint, run_dir, options))
        return fake_fit_result(run_dir, 6)

    monkeypatch.setattr("wastevision.training.train.train_full", fake_train_full)

    result = runner.invoke(
        app,
        [
            "train",
            "--stage",
            "full",
            "--no-class-weights",
            "--head-checkpoint",
            str(tmp_path / "head.pt"),
            "--run-dir",
            str(tmp_path / "full"),
        ],
    )

    assert result.exit_code == 0, result.output
    head_checkpoint, run_dir, options = calls[0]
    assert head_checkpoint == tmp_path / "head.pt"
    assert run_dir == tmp_path / "full"
    assert options["epochs"] == 25
    assert options["use_class_weights"] is False
    assert "Stopped early after 6 epochs" in result.stdout


def test_train_full_default_run_folder(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("torch")
    calls = []

    def fake_train_full(images_dir, split_path, head_checkpoint, run_dir, **options):
        calls.append((head_checkpoint, run_dir, options["epochs"]))
        return fake_fit_result(run_dir, options["epochs"])

    monkeypatch.setattr("wastevision.training.train.train_full", fake_train_full)

    result = runner.invoke(app, ["train", "--stage", "full", "--epochs", "3"])

    assert result.exit_code == 0, result.output
    assert calls == [(Path("runs/head/best.pt"), Path("runs/full"), 3)]


def test_train_reports_a_missing_head_checkpoint(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    pytest.importorskip("torch")
    images_dir, split_path = tiny_dataset

    result = runner.invoke(
        app,
        [
            "train",
            "--stage",
            "full",
            "--images-dir",
            str(images_dir),
            "--split",
            str(split_path),
            "--head-checkpoint",
            str(tmp_path / "missing.pt"),
        ],
    )

    assert result.exit_code == 1
    assert "Error: Head checkpoint" in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [["train"], ["train", "--stage", "middle"], ["train", "--stage", "head", "--epochs", "0"]],
)
def test_train_rejects_invalid_options(arguments: list[str]):
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2


def test_curves_writes_the_plot(tmp_path: Path):
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "head"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text(
        "epoch,train_loss,train_accuracy,train_macro_f1,"
        "validation_loss,validation_accuracy,validation_macro_f1,seconds\n"
        "1,0.9,0.6,0.5,0.8,0.7,0.6,10.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "curves.png"

    result = runner.invoke(app, ["curves", "--run", str(run_dir), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert f"Wrote training curves to {output}" in result.stdout


def test_curves_reports_errors(tmp_path: Path):
    pytest.importorskip("matplotlib")

    result = runner.invoke(app, ["curves", "--run", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "Error:" in result.stderr


def evaluate_arguments(images_dir: Path, split_path: Path, checkpoint: Path, tmp_path: Path):
    return [
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--images-dir",
        str(images_dir),
        "--split",
        str(split_path),
        "--confusion-output",
        str(tmp_path / "img" / "confusion.png"),
        "--reliability-output",
        str(tmp_path / "img" / "reliability.png"),
    ]


def test_evaluate_prints_test_metrics_and_writes_the_plots(
    tiny_dataset: tuple[Path, Path], tmp_path: Path
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from wastevision.training.loop import EpochMetrics, save_checkpoint
    from wastevision.training.model import build_model

    images_dir, split_path = tiny_dataset
    checkpoint = tmp_path / "best.pt"
    torch.manual_seed(0)
    save_checkpoint(build_model(pretrained=False), checkpoint, 1, EpochMetrics(1.0, 0.5, 0.5))

    result = runner.invoke(app, evaluate_arguments(images_dir, split_path, checkpoint, tmp_path))

    assert result.exit_code == 0, result.output
    assert f"Test split ({len(CLASS_NAMES)} images):" in result.stdout
    assert "macro-F1:" in result.stdout
    assert "Confusion matrix (rows: true class, columns: predicted class):" in result.stdout
    assert "0.9 to 1.0" in result.stdout
    assert "expected calibration error:" in result.stdout
    assert (tmp_path / "img" / "confusion.png").is_file()
    assert (tmp_path / "img" / "reliability.png").is_file()


def test_evaluate_reports_a_missing_checkpoint(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    images_dir, split_path = tiny_dataset

    result = runner.invoke(
        app, evaluate_arguments(images_dir, split_path, tmp_path / "missing.pt", tmp_path)
    )

    assert result.exit_code == 1
    assert "Error: Checkpoint" in result.stderr
    assert not (tmp_path / "img").exists()


def all_test_split(images_dir: Path, tmp_path: Path) -> Path:
    rows = ["file,class,split"]
    for class_name in CLASS_NAMES:
        for image_index in range(5):
            rows.append(f"{class_name}/{class_name}{image_index}.png,{class_name},test")
    split_path = tmp_path / "all-test.csv"
    split_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return split_path


def fake_predictions(mistakes: dict[int, tuple[int, float]]):
    import numpy as np

    from wastevision.evaluation.predictions import Predictions

    labels = np.repeat(np.arange(len(CLASS_NAMES)), 5)
    probabilities = np.full((len(labels), len(CLASS_NAMES)), 0.02)
    probabilities[np.arange(len(labels)), labels] = 0.9
    for index, (predicted, confidence) in mistakes.items():
        probabilities[index] = (1 - confidence) / (len(CLASS_NAMES) - 1)
        probabilities[index, predicted] = confidence
    return Predictions(labels, probabilities)


def error_analysis_arguments(images_dir: Path, split_path: Path, checkpoint: Path, tmp_path: Path):
    return [
        "error-analysis",
        "--checkpoint",
        str(checkpoint),
        "--images-dir",
        str(images_dir),
        "--split",
        str(split_path),
        "--mistakes-output",
        str(tmp_path / "img" / "mistakes.png"),
        "--gradcam-output",
        str(tmp_path / "img" / "gradcam.png"),
    ]


def run_error_analysis(
    tiny_dataset: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mistakes: dict[int, tuple[int, float]],
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from wastevision.evaluation import predictions
    from wastevision.training.loop import EpochMetrics, save_checkpoint
    from wastevision.training.model import build_model

    images_dir, _ = tiny_dataset
    split_path = all_test_split(images_dir, tmp_path)
    checkpoint = tmp_path / "best.pt"
    torch.manual_seed(0)
    save_checkpoint(build_model(pretrained=False), checkpoint, 1, EpochMetrics(1.0, 0.5, 0.5))
    requested_splits: list[str] = []

    def predict_split(checkpoint_path, images, split_file, split):
        requested_splits.append(split)
        return fake_predictions(mistakes)

    monkeypatch.setattr(predictions, "predict_split", predict_split)
    result = runner.invoke(
        app, error_analysis_arguments(images_dir, split_path, checkpoint, tmp_path)
    )
    assert requested_splits == ["test"]
    return result


def test_error_analysis_lists_mistakes_and_writes_both_figures(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paper_index = CLASS_NAMES.index("paper") * 5
    plastic_index = CLASS_NAMES.index("plastic") * 5
    mistakes = {
        paper_index: (CLASS_NAMES.index("trash"), 0.95),
        plastic_index: (CLASS_NAMES.index("glass"), 0.8),
    }

    result = run_error_analysis(tiny_dataset, tmp_path, monkeypatch, mistakes)

    assert result.exit_code == 0, result.output
    assert "Test split (30 images), 2 mistakes" in result.stdout
    assert "paper -> trash: 1" in result.stdout
    assert "plastic -> glass: 1" in result.stdout
    lines = result.stdout.splitlines()
    first_mistake = next(line for line in lines if line.startswith("paper/paper0.png"))
    assert first_mistake.split() == ["paper/paper0.png", "paper", "trash", "0.9500"]
    assert lines.index(first_mistake) < next(
        index for index, line in enumerate(lines) if line.startswith("plastic/plastic0.png")
    )
    assert (tmp_path / "img" / "mistakes.png").is_file()
    assert (tmp_path / "img" / "gradcam.png").is_file()
    assert "Wrote 2 mistakes" in result.stdout
    assert "Wrote 2 Grad-CAM example pairs" in result.stdout


def test_error_analysis_without_mistakes_writes_no_figures(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    result = run_error_analysis(tiny_dataset, tmp_path, monkeypatch, mistakes={})

    assert result.exit_code == 0, result.output
    assert "No mistakes to show." in result.stdout
    assert not (tmp_path / "img").exists()


def test_error_analysis_reports_a_missing_checkpoint(
    tiny_dataset: tuple[Path, Path], tmp_path: Path
):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    images_dir, split_path = tiny_dataset

    result = runner.invoke(
        app, error_analysis_arguments(images_dir, split_path, tmp_path / "missing.pt", tmp_path)
    )

    assert result.exit_code == 1
    assert "Error: Checkpoint" in result.stderr
    assert not (tmp_path / "img").exists()


def test_error_analysis_without_a_correct_image_of_the_class_skips_grad_cam(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paper_start = CLASS_NAMES.index("paper") * 5
    mistakes = {
        index: (CLASS_NAMES.index("trash"), 0.7) for index in range(paper_start, paper_start + 5)
    }

    result = run_error_analysis(tiny_dataset, tmp_path, monkeypatch, mistakes)

    assert result.exit_code == 0, result.output
    assert "paper -> trash: 5" in result.stdout
    assert "Wrote 5 mistakes" in result.stdout
    assert "No class has both a correct and a wrong prediction." in result.stdout
    assert (tmp_path / "img" / "mistakes.png").is_file()
    assert not (tmp_path / "img" / "gradcam.png").exists()


def save_untrained_checkpoint(path: Path) -> None:
    torch = pytest.importorskip("torch")
    from wastevision.training.loop import EpochMetrics, save_checkpoint
    from wastevision.training.model import build_model

    torch.manual_seed(0)
    save_checkpoint(build_model(pretrained=False), path, 1, EpochMetrics(1.0, 0.5, 0.5))


def test_robustness_prints_every_perturbation(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    pytest.importorskip("torch")
    from wastevision.evaluation.robustness import PERTURBATIONS

    images_dir, split_path = tiny_dataset
    checkpoint = tmp_path / "best.pt"
    save_untrained_checkpoint(checkpoint)

    result = runner.invoke(
        app,
        [
            "robustness",
            "--checkpoint",
            str(checkpoint),
            "--images-dir",
            str(images_dir),
            "--split",
            str(split_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Test split ({len(CLASS_NAMES)} images)" in result.stdout
    assert "clean: accuracy" in result.stdout
    for name in PERTURBATIONS:
        assert name in result.stdout


def test_robustness_reports_a_missing_checkpoint(tiny_dataset: tuple[Path, Path], tmp_path: Path):
    pytest.importorskip("torch")
    images_dir, split_path = tiny_dataset

    result = runner.invoke(
        app,
        [
            "robustness",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--images-dir",
            str(images_dir),
            "--split",
            str(split_path),
        ],
    )

    assert result.exit_code == 1
    assert "Error: Checkpoint" in result.stderr
