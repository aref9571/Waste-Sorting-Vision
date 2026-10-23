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
    code = "import sys, wastevision.cli; print('sklearn' in sys.modules, 'skimage' in sys.modules)"

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "False False"


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
