from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from wastevision import __version__
from wastevision.data.dataset import SplitFileError
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

if TYPE_CHECKING:
    from wastevision.baseline.train import EvaluationResult

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare the dataset.")
app.add_typer(data_app, name="data")
baseline_app = typer.Typer(no_args_is_help=True, help="Train and evaluate the classical baseline.")
app.add_typer(baseline_app, name="baseline")

DEFAULT_IMAGES_DIR = DEFAULT_RAW_DIR / IMAGES_DIR_NAME
DEFAULT_MODEL_PATH = Path("runs/baseline/model.joblib")


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


def echo_evaluation(result: "EvaluationResult") -> None:
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
