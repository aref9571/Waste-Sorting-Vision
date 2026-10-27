# Waste Sorting Vision

Waste Sorting Vision is an image classification service that tells you which recycling category a photographed household item belongs to: cardboard, glass, metal and cans, paper, plastic, or other waste. It compares a classical image-processing baseline with a fine-tuned convolutional neural network, and evaluates both honestly, including on real photos taken outside the lab. The final model will be served through a tested FastAPI service and a simple web page, packaged with Docker.

## Status

In development.

| Part | Status |
|:--|:--|
| Dataset download, checksum and fixed train/validation/test split | Done |
| Image preprocessing and augmentation | Done |
| Classical baseline (colour, texture and shape features with logistic regression or SVM), evaluated on the test split | Done |
| Convolutional neural network: model and training loop | In progress, not yet available from the command line |
| Evaluation of the neural network and comparison with the baseline | Planned |
| FastAPI service, web page and Docker image | Planned |

## Development setup

Requirements: Python 3.12 and Git.

```bash
git clone https://github.com/aref9571/waste-sorting-vision.git
cd waste-sorting-vision
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The default install contains only the serving dependencies. Training needs the `train` extra, which adds PyTorch and the other training libraries:

```bash
pip install -e ".[train,dev]"
```

Run the checks:

```bash
ruff format --check .
ruff check .
pytest
```

Show the installed version:

```bash
wastevision version
```

## Dataset

The models are trained on the TrashNet dataset. Download it (about 43 MB, checked against a SHA-256 checksum) into `data/raw/`:

```bash
wastevision data download
```

The fixed train/validation/test split is committed as [splits/trashnet-split.csv](splits/trashnet-split.csv). It can be rebuilt from the downloaded images with:

```bash
wastevision data split
```

See [docs/dataset.md](docs/dataset.md) for the source, checksum, class counts and known bias, and [NOTICE.md](NOTICE.md) for the credit and licence.

For the neural network, images are resized and cropped to 224 × 224 pixels and normalised with the ImageNet statistics; training images are also randomly cropped, flipped, rotated and colour-shifted. The preprocessing section of [docs/dataset.md](docs/dataset.md) lists the settings and shows examples, which can be recreated with:

```bash
wastevision data preview
```

## Classical baseline

A classical image-processing baseline (HSV colour histogram, local binary patterns and HOG features with a logistic regression or RBF SVM) sets the reference point for the neural network. It needs the `train` extra and the downloaded dataset:

```bash
wastevision baseline train
wastevision baseline evaluate
```

On the test split it reaches an accuracy of 0.730 and a macro-F1 of 0.720. See [docs/evaluation.md](docs/evaluation.md) for the features, the model selection on validation and the per-class results.
