from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer

from wastevision import __version__
from wastevision.config import Augmentation
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
from wastevision.data.real_photos import (
    DEFAULT_REAL_DIR,
    RealPhotoError,
    count_photos_per_class,
    find_real_photos,
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
    from wastevision.evaluation.error_analysis import ConfusedPair
    from wastevision.evaluation.metrics import ClassificationReport
    from wastevision.evaluation.predictions import Predictions
    from wastevision.evaluation.robustness import RobustnessRow
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
DEFAULT_MISTAKES_PATH = Path("docs/img/misclassified.png")
DEFAULT_GRADCAM_PATH = Path("docs/img/gradcam.png")
DEFAULT_REAL_CONFUSION_PATH = Path("docs/img/real-confusion-matrix.png")
DEFAULT_MODELS_DIR = Path("models")
GRID_COLUMNS = 4
MIN_PHOTOS_PER_CLASS = 10


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
    augmentation: Annotated[
        Augmentation,
        typer.Option(help="standard: crop, flip, rotation, colour; strong: adds RandAugment."),
    ] = Augmentation.STANDARD,
    label_smoothing: Annotated[
        float,
        typer.Option(min=0.0, help="Label smoothing of the loss, below 1."),
    ] = 0.1,
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
    if label_smoothing >= 1.0:
        raise typer.BadParameter("must be below 1", param_hint="'--label-smoothing'")
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
                augmentation=augmentation,
                label_smoothing=label_smoothing,
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
                augmentation=augmentation,
                label_smoothing=label_smoothing,
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


def echo_confused_pairs(pairs: "list[ConfusedPair]") -> None:
    typer.echo("Most frequent confusions (true class -> predicted class):")
    for pair in pairs:
        typer.echo(
            f"  {CLASS_NAMES[pair.true_class]} -> {CLASS_NAMES[pair.predicted_class]}: {pair.count}"
        )


def describe_prediction(probabilities: np.ndarray, label: int) -> str:
    predicted = int(np.argmax(probabilities))
    return (
        f"true {CLASS_NAMES[label]}\n"
        f"predicted {CLASS_NAMES[predicted]} ({probabilities[predicted]:.2f})"
    )


@app.command("error-analysis")
def error_analysis(
    checkpoint: Annotated[
        Path, typer.Option(help="Trained network checkpoint to analyse.")
    ] = DEFAULT_FULL_CHECKPOINT,
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
    mistakes: Annotated[int, typer.Option(min=1, help="Number of mistakes to show.")] = 12,
    examples: Annotated[
        int, typer.Option(min=1, help="Number of classes with a Grad-CAM example pair.")
    ] = 3,
    mistakes_output: Annotated[
        Path, typer.Option(help="PNG file for the grid of mistakes.")
    ] = DEFAULT_MISTAKES_PATH,
    gradcam_output: Annotated[
        Path, typer.Option(help="PNG file for the Grad-CAM examples.")
    ] = DEFAULT_GRADCAM_PATH,
) -> None:
    from wastevision.evaluation.error_analysis import (
        gradcam_pairs,
        most_confused_pairs,
        worst_mistakes,
    )
    from wastevision.evaluation.gradcam import explain_image, load_model_view
    from wastevision.evaluation.metrics import classification_report
    from wastevision.evaluation.plots import save_image_grid
    from wastevision.evaluation.predictions import load_trained_model, predict_split

    try:
        model = load_trained_model(checkpoint)
        predictions = predict_split(checkpoint, images_dir, split_path, "test")
        records = load_image_records(split_path, images_dir, "test")
    except (SplitFileError, ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    labels = predictions.labels
    probabilities = predictions.probabilities
    report = classification_report(labels.tolist(), predictions.predicted.tolist(), CLASS_NAMES)
    typer.echo(f"Test split ({len(labels)} images), {int((~predictions.correct).sum())} mistakes")
    echo_confused_pairs(most_confused_pairs(report.confusion, count=len(CLASS_NAMES)))

    worst = worst_mistakes(labels, probabilities, mistakes)
    if not worst:
        typer.echo("No mistakes to show.")
        return
    typer.echo(f"{'file':<28} {'true':<10} {'predicted':<10} {'confidence':>10}")
    for index in worst:
        predicted = int(predictions.predicted[index])
        typer.echo(
            f"{records[index].path.relative_to(images_dir).as_posix():<28}"
            f" {CLASS_NAMES[labels[index]]:<10} {CLASS_NAMES[predicted]:<10}"
            f" {predictions.confidences[index]:>10.4f}"
        )
    save_image_grid(
        [load_model_view(records[index].path) for index in worst],
        [describe_prediction(probabilities[index], labels[index]) for index in worst],
        mistakes_output,
        GRID_COLUMNS,
    )
    typer.echo(f"Wrote {len(worst)} mistakes to {mistakes_output}")

    pairs = gradcam_pairs(labels, probabilities, examples)
    if not pairs:
        typer.echo("No class has both a correct and a wrong prediction.")
        return
    images: list[np.ndarray] = []
    titles: list[str] = []
    for pair in pairs:
        for index in pair:
            predicted = int(predictions.predicted[index])
            image, overlay = explain_image(model, records[index].path, predicted)
            description = describe_prediction(probabilities[index], labels[index])
            images.extend([image, overlay])
            titles.extend([description, f"Grad-CAM for {CLASS_NAMES[predicted]}"])
    save_image_grid(images, titles, gradcam_output, GRID_COLUMNS)
    typer.echo(f"Wrote {len(pairs)} Grad-CAM example pairs to {gradcam_output}")


def report_for(predictions: "Predictions") -> "ClassificationReport":
    from wastevision.evaluation.metrics import classification_report

    return classification_report(
        predictions.labels.tolist(), predictions.predicted.tolist(), CLASS_NAMES
    )


def echo_robustness(rows: "list[RobustnessRow]") -> None:
    typer.echo(f"{'perturbation':<30} {'accuracy':>10} {'drop':>8} {'macro-F1':>10} {'drop':>8}")
    for row in rows:
        typer.echo(
            f"{row.name:<30} {row.accuracy:>10.4f} {row.accuracy_drop:>8.4f}"
            f" {row.macro_f1:>10.4f} {row.macro_f1_drop:>8.4f}"
        )


@app.command()
def robustness(
    checkpoint: Annotated[
        Path, typer.Option(help="Trained network checkpoint to evaluate.")
    ] = DEFAULT_FULL_CHECKPOINT,
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
) -> None:
    from wastevision.data.transforms import evaluation_transforms, perturbed_evaluation_transforms
    from wastevision.evaluation.predictions import load_trained_model, predict_records
    from wastevision.evaluation.robustness import PERTURBATIONS, robustness_rows

    try:
        model = load_trained_model(checkpoint)
        records = load_image_records(split_path, images_dir, "test")
        clean_predictions = predict_records(model, records, evaluation_transforms())
        perturbed_predictions = {
            name: predict_records(model, records, perturbed_evaluation_transforms(perturbation))
            for name, perturbation in PERTURBATIONS.items()
        }
    except (SplitFileError, ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    clean = report_for(clean_predictions)
    typer.echo(f"Test split ({len(records)} images)")
    typer.echo(f"clean: accuracy {clean.accuracy:.4f}, macro-F1 {clean.macro_f1:.4f}")
    echo_robustness(
        robustness_rows(
            clean,
            {name: report_for(predictions) for name, predictions in perturbed_predictions.items()},
        )
    )


def echo_photo_counts(counts: dict[str, int]) -> None:
    for class_name, count in counts.items():
        typer.echo(f"  {class_name}: {count}")
        if count < MIN_PHOTOS_PER_CLASS:
            typer.echo(
                f"Warning: {class_name} has fewer than {MIN_PHOTOS_PER_CLASS} photos,"
                " so its scores are very uncertain."
            )


@app.command("evaluate-real")
def evaluate_real(
    checkpoint: Annotated[
        Path, typer.Option(help="Trained network checkpoint to evaluate.")
    ] = DEFAULT_FULL_CHECKPOINT,
    real_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of real photos per class.")
    ] = DEFAULT_REAL_DIR,
    confusion_output: Annotated[
        Path, typer.Option(help="PNG file for the confusion matrix.")
    ] = DEFAULT_REAL_CONFUSION_PATH,
    examples_output: Annotated[
        Path | None,
        typer.Option(help="PNG file for one example prediction per class [default: none]."),
    ] = None,
) -> None:
    from wastevision.data.preview import first_record_per_class
    from wastevision.data.real_photos import load_photo_without_metadata
    from wastevision.data.transforms import evaluation_transforms
    from wastevision.evaluation.gradcam import load_model_view
    from wastevision.evaluation.plots import save_confusion_matrix, save_image_grid
    from wastevision.evaluation.predictions import load_trained_model, predict_records

    try:
        records = find_real_photos(real_dir)
        model = load_trained_model(checkpoint)
        predictions = predict_records(
            model, records, evaluation_transforms(), load_photo_without_metadata
        )
    except (RealPhotoError, ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Real photos ({len(records)} images) in {real_dir}:")
    echo_photo_counts(count_photos_per_class(records))
    report = report_for(predictions)
    echo_evaluation(report)
    echo_confusion_matrix(report)
    typer.echo(f"mean confidence: {predictions.confidences.mean():.4f}")
    typer.echo(f"{'file':<36} {'true':<10} {'predicted':<10} {'confidence':>10}")
    for record, label, probabilities in zip(
        records, predictions.labels, predictions.probabilities, strict=True
    ):
        predicted = int(np.argmax(probabilities))
        typer.echo(
            f"{record.path.relative_to(real_dir).as_posix():<36} {CLASS_NAMES[label]:<10}"
            f" {CLASS_NAMES[predicted]:<10} {probabilities[predicted]:>10.4f}"
        )

    save_confusion_matrix(
        report.confusion, CLASS_NAMES, confusion_output, "Neural network, real photos"
    )
    typer.echo(f"Wrote confusion matrix to {confusion_output}")
    if examples_output is None:
        return
    examples = first_record_per_class(records)
    indices = [records.index(record) for record in examples]
    save_image_grid(
        [load_model_view(record.path, load_photo_without_metadata) for record in examples],
        [
            describe_prediction(predictions.probabilities[index], predictions.labels[index])
            for index in indices
        ],
        examples_output,
        GRID_COLUMNS,
    )
    typer.echo(f"Wrote {len(examples)} example predictions to {examples_output}")


@app.command("export")
def export(
    checkpoint: Annotated[
        Path, typer.Option(help="Trained network checkpoint to export.")
    ] = DEFAULT_FULL_CHECKPOINT,
    images_dir: Annotated[
        Path, typer.Option(help="Folder with one subfolder of images per class.")
    ] = DEFAULT_IMAGES_DIR,
    split_path: Annotated[Path, typer.Option("--split", help="Split CSV.")] = DEFAULT_SPLIT_PATH,
    models_dir: Annotated[
        Path, typer.Option(help="Folder for the ONNX model, labels.json and metadata.json.")
    ] = DEFAULT_MODELS_DIR,
) -> None:
    from wastevision.training.export import ParityError, export_model

    try:
        result = export_model(checkpoint, images_dir, split_path, models_dir)
    except (ParityError, SplitFileError, ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    parity = result.parity
    test = result.metadata["metrics"]["test"]
    typer.echo(
        f"Parity on {parity.images} test images: max logit difference"
        f" {parity.max_abs_difference:.2e}, top-1 equal for {parity.top1_matches}"
    )
    typer.echo(
        f"Test split ({test['images']} images): accuracy {test['accuracy']:.4f},"
        f" macro-F1 {test['macro_f1']:.4f}"
    )
    size_mb = result.model_path.stat().st_size / 1_000_000
    typer.echo(f"Wrote {result.model_path} ({size_mb:.1f} MB)")
    typer.echo(f"Wrote {result.labels_path}")
    typer.echo(f"Wrote {result.metadata_path}")
