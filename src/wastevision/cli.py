from pathlib import Path
from typing import Annotated

import typer

from wastevision import __version__
from wastevision.data.download import (
    DEFAULT_RAW_DIR,
    ChecksumMismatchError,
    DatasetError,
    UnsafeArchiveError,
    count_differences,
    count_images_per_class,
    download_dataset,
)

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare the dataset.")
app.add_typer(data_app, name="data")


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
