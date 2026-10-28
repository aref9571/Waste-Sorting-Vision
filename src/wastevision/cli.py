from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from wastevision import __version__
from wastevision.data.dataset import SplitFileError, load_image_records
from wastevision.data.download import (
    DEFAULT_RAW_DIR,
    IMAGES_DIR_NAME,
    ChecksumMismatchError,
    DatasetError,
    UnsafeArchiveError,
    count_differences,
    count_images_per_class,
    download_dataset,
    sha256_of_file,
)
from wastevision.data.split import (
    DEFAULT_SPLIT_PATH,
    SPLIT_NAMES,
    SplitError,
    count_per_split,
    create_split,
    write_split_csv,
)
from wastevision.labels import CLASS_NAMES

if TYPE_CHECKING:
    from wastevision.baseline.train import EvaluationResult
    from wastevision.evaluation.calibration import CalibrationBin
    from wastevision.evaluation.metrics import ClassificationReport
    from wastevision.training.loop import EpochRecord

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare the dataset.")
app.add_typer(data_app, name="data")
baseline_app = typer.Typer(no_args_is_help=True, help="Train and evaluate the classical baseline.")
app.add_typer(baseline_app, name="baseline")

DEFAULT_IMAGES_DIR = DEFAULT_RAW_DIR / IMAGES_DIR_NAME
DEFAULT_MODEL_PATH = Path("runs/baseline/model.joblib")
DEFAULT_PREVIEW_PATH = Path("docs/img/augmentations.png")
DEFAULT_HEAD_CHECKPOINT = Path("runs/head/best.pt")
DEFAULT_CURVES_PATH = Path("docs/img/training-curves.png")
DEFAULT_FULL_CHECKPOINT = Path("runs/full/best.pt")
DEFAULT_CONFUSION_PATH = Path("docs/img/confusion-matrix.png")
DEFAULT_RELIABILITY_PATH = Path("docs/img/reliability-diagram.png")


class TrainingStage(StrEnum):
    HEAD = "head"
    FULL = "full"


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(__version__)


@data_app.command("download")
def download(
    raw_dir: Annotated[
        Path, typer.Option(help="Folder for the archive and the extracted images.")
    ] = DEFAULT_RAW_DIR,
) -> None:
    try:
        images_dir = download_dataset(raw_dir)
    except (ChecksumMismatchError, UnsafeArchiveError, DatasetError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    counts = count_images_per_class(images_dir)
    typer.echo(f"Images in {images_dir}:")
    for class_name, count in counts.items():
        typer.echo(f"  {class_name}: {count}")
    typer.echo(f"  total: {sum(counts.values())}")

    differences = count_differences(counts)
    if not differences:
        typer.echo("Counts match the published TrashNet counts.")
    for class_name, (actual, expected) in differences.items():
        typer.echo(f"Warning: {class_name} has {actual} images, expected {expected}.")


@data_app.command("split")
def split(
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    output: Annotated[Path, typer.Option(help="CSV file to write.")] = DEFAULT_SPLIT_PATH,
) -> None:
    try:
        result = create_split(images_dir)
        write_split_csv(result.entries, output)
    except (SplitError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    for image in result.excluded:
        typer.echo(f"Excluded {image.file}: identical image with another class label.")
    typer.echo(f"{'class':<10} " + " ".join(f"{name:>10}" for name in SPLIT_NAMES))
    for class_name, counts in count_per_split(result.entries).items():
        typer.echo(f"{class_name:<10} " + " ".join(f"{counts[name]:>10}" for name in SPLIT_NAMES))
    typer.echo(f"Wrote {len(result.entries)} images to {output}")
    typer.echo(f"SHA-256: {sha256_of_file(output)}")


@data_app.command("preview")
def preview(
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
    output: Annotated[Path, typer.Option(help="PNG file to write.")] = DEFAULT_PREVIEW_PATH,
) -> None:
    from wastevision.data.preview import first_record_per_class, save_augmentation_preview

    try:
        records = first_record_per_class(load_image_records(split_path, images_dir, "train"))
        save_augmentation_preview(records, output)
    except (SplitFileError, ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Wrote augmentation preview of {len(records)} images to {output}")


def echo_evaluation(result: "EvaluationResult | ClassificationReport") -> None:
    typer.echo(f"accuracy: {result.accuracy:.4f}")
    typer.echo(f"macro-F1: {result.macro_f1:.4f}")
    typer.echo(f"{'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for class_name, scores in result.per_class.items():
        typer.echo(
            f"{class_name:<10} {scores.precision:>10.4f} {scores.recall:>10.4f}"
            f" {scores.f1:>10.4f} {scores.support:>10}"
        )


@baseline_app.command("train")
def baseline_train(
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
    model_path: Annotated[
        Path, typer.Option("--model", help="File for the trained pipeline.")
    ] = DEFAULT_MODEL_PATH,
) -> None:
    from wastevision.baseline.train import (
        BaselineError,
        load_split_features,
        save_model,
        select_baseline,
    )

    try:
        train_features, train_labels = load_split_features(split_path, images_dir, "train")
        validation_features, validation_labels = load_split_features(
            split_path, images_dir, "validation"
        )
    except (BaselineError, SplitFileError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    selection = select_baseline(
        train_features, train_labels, validation_features, validation_labels
    )
    typer.echo(f"{'model':<20} {'parameters':<30} {'val accuracy':>12} {'val macro-F1':>12}")
    for candidate in selection.candidates:
        typer.echo(
            f"{candidate.model_name:<20} {candidate.params!s:<30}"
            f" {candidate.validation.accuracy:>12.4f} {candidate.validation.macro_f1:>12.4f}"
        )
    best = selection.best
    typer.echo(f"Best on validation: {best.model_name} {best.params}")
    echo_evaluation(best.validation)
    save_model(selection.pipeline, model_path)
    typer.echo(f"Saved pipeline to {model_path}")


@baseline_app.command("evaluate")
def baseline_evaluate(
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
    model_path: Annotated[
        Path, typer.Option("--model", help="Trained pipeline to evaluate.")
    ] = DEFAULT_MODEL_PATH,
) -> None:
    from wastevision.baseline.train import BaselineError, evaluate, load_model, load_split_features

    try:
        pipeline = load_model(model_path)
        test_features, test_labels = load_split_features(split_path, images_dir, "test")
    except (BaselineError, SplitFileError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Test split ({len(test_labels)} images):")
    echo_evaluation(evaluate(pipeline, test_features, test_labels))


def echo_epoch(record: "EpochRecord") -> None:
    typer.echo(
        f"epoch {record.epoch:>3}  train loss {record.train.loss:.4f}"
        f"  val loss {record.validation.loss:.4f}  val accuracy {record.validation.accuracy:.4f}"
        f"  val macro-F1 {record.validation.macro_f1:.4f}  ({record.seconds:.0f} s)"
    )


@app.command()
def train(
    stage: Annotated[
        TrainingStage,
        typer.Option(help="head: train only the classifier; full: fine-tune every layer."),
    ],
    epochs: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum number of epochs [default: 8 for head, 25 for full]."),
    ] = None,
    class_weights: Annotated[
        bool, typer.Option(help="Weight the loss by inverse class frequency.")
    ] = True,
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
    run_dir: Annotated[
        Path | None, typer.Option(help="Folder for metrics.csv and best.pt [default: runs/STAGE].")
    ] = None,
    head_checkpoint: Annotated[
        Path, typer.Option(help="Head checkpoint that the full stage starts from.")
    ] = DEFAULT_HEAD_CHECKPOINT,
) -> None:
    from wastevision.training.train import FULL_EPOCHS, HEAD_EPOCHS, train_full, train_head

    run_dir = run_dir or Path("runs") / stage.value
    default_epochs = HEAD_EPOCHS if stage is TrainingStage.HEAD else FULL_EPOCHS
    max_epochs = epochs or default_epochs
    try:
        if stage is TrainingStage.HEAD:
            result = train_head(
                images_dir,
                split_path,
                run_dir,
                epochs=max_epochs,
                use_class_weights=class_weights,
                report=echo_epoch,
            )
        else:
            result = train_full(
                images_dir,
                split_path,
                head_checkpoint,
                run_dir,
                epochs=max_epochs,
                use_class_weights=class_weights,
                report=echo_epoch,
            )
    except (SplitFileError, ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if len(result.history) < max_epochs:
        typer.echo(f"Stopped early after {len(result.history)} epochs")
    best = result.best_validation
    typer.echo(
        f"Best epoch {result.best_epoch}: val accuracy {best.accuracy:.4f},"
        f" val macro-F1 {best.macro_f1:.4f}"
    )
    typer.echo(f"Saved checkpoint to {result.checkpoint_path}")


@app.command()
def curves(
    run_dirs: Annotated[
        list[Path],
        typer.Option("--run", help="Run folder with a metrics.csv; repeat to join runs in order."),
    ],
    output: Annotated[Path, typer.Option(help="PNG file to write.")] = DEFAULT_CURVES_PATH,
) -> None:
    from wastevision.training.curves import save_training_curves

    try:
        save_training_curves(run_dirs, output)
    except (ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Wrote training curves to {output}")


def echo_confusion_matrix(report: "ClassificationReport") -> None:
    typer.echo("Confusion matrix (rows: true class, columns: predicted class):")
    typer.echo(f"{'':<10} " + " ".join(f"{name:>10}" for name in CLASS_NAMES))
    for class_name, row in zip(CLASS_NAMES, report.confusion, strict=True):
        typer.echo(f"{class_name:<10} " + " ".join(f"{int(count):>10}" for count in row))


def echo_calibration(bins: "list[CalibrationBin]", ece: float) -> None:
    typer.echo(f"{'confidence':<12} {'images':>8} {'mean confidence':>16} {'accuracy':>10}")
    for calibration_bin in bins:
        typer.echo(
            f"{calibration_bin.lower:.1f} to {calibration_bin.upper:.1f}"
            f" {calibration_bin.count:>8} {calibration_bin.mean_confidence:>16.4f}"
            f" {calibration_bin.accuracy:>10.4f}"
        )
    typer.echo(f"expected calibration error: {ece:.4f}")


@app.command()
def evaluate(
    checkpoint: Annotated[
        Path, typer.Option(help="Trained network checkpoint to evaluate.")
    ] = DEFAULT_FULL_CHECKPOINT,
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
    confusion_output: Annotated[
        Path, typer.Option(help="PNG file for the confusion matrix.")
    ] = DEFAULT_CONFUSION_PATH,
    reliability_output: Annotated[
        Path, typer.Option(help="PNG file for the reliability diagram.")
    ] = DEFAULT_RELIABILITY_PATH,
) -> None:
    from wastevision.evaluation.calibration import expected_calibration_error, reliability_bins
    from wastevision.evaluation.metrics import classification_report
    from wastevision.evaluation.plots import save_confusion_matrix, save_reliability_diagram
    from wastevision.evaluation.predictions import predict_split

    try:
        predictions = predict_split(checkpoint, images_dir, split_path, "test")
    except (SplitFileError, ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    report = classification_report(
        predictions.labels.tolist(), predictions.predicted.tolist(), CLASS_NAMES
    )
    bins = reliability_bins(predictions.confidences.tolist(), predictions.correct.tolist())
    typer.echo(f"Test split ({len(predictions.labels)} images):")
    echo_evaluation(report)
    echo_confusion_matrix(report)
    echo_calibration(bins, expected_calibration_error(bins))

    save_confusion_matrix(
        report.confusion, CLASS_NAMES, confusion_output, "Neural network, test split"
    )
    save_reliability_diagram(bins, reliability_output)
    typer.echo(f"Wrote confusion matrix to {confusion_output}")
    typer.echo(f"Wrote reliability diagram to {reliability_output}")
