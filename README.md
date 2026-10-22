# Waste Sorting Vision

Waste Sorting Vision is an image classification service that tells you which recycling category a photographed household item belongs to: cardboard, glass, metal and cans, paper, plastic, or other waste. It compares a classical image-processing baseline with a fine-tuned convolutional neural network, evaluates both honestly (including on real photos taken outside the lab), and serves the final model through a tested FastAPI service and a simple web page, packaged with Docker.

## Status

In development. The project setup is in place; the dataset pipeline, models, evaluation and service are still being built.

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

See [docs/dataset.md](docs/dataset.md) for the source, checksum, class counts and known bias, and [NOTICE.md](NOTICE.md) for the credit and licence.
