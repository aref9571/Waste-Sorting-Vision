# Waste Sorting Vision

Waste Sorting Vision is an image classification service that tells you which recycling category a photographed household item belongs to: cardboard, glass, metal and cans, paper, plastic, or other waste. It compares a classical image-processing baseline with a fine-tuned convolutional neural network, and evaluates both honestly, including on real photos taken outside the lab. The final model will be served through a tested FastAPI service and a simple web page, packaged with Docker.

## Status

In development.

| Part | Status |
|:--|:--|
| Dataset download, checksum and fixed train/validation/test split | Done |
| Image preprocessing and augmentation | Done |
| Classical baseline (colour, texture and shape features with logistic regression or SVM), evaluated on the test split | Done |
| Convolutional neural network: two-stage training with `wastevision train`, results on the validation split | Done |
| Evaluation of the neural network on the test split, calibration and comparison with the baseline | Done |
| Error analysis with Grad-CAM explanations | Done |
| Robustness checks and a test on real-world photos (RealWaste) | Done |
| Comparison of two improvements on the validation split; label smoothing kept | Done |
| ONNX export of the final model with a parity check against PyTorch | Done |
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

## Neural network

The main model is MobileNetV3-Small, pretrained on ImageNet and fine-tuned in two stages: first only the new classification layer, then the whole network with a lower learning rate for the pretrained layers. It needs the `train` extra and the downloaded dataset:

```bash
wastevision train --stage head
wastevision train --stage full
```

On the validation split it reaches an accuracy of 0.926 and a macro-F1 of 0.919. Evaluate the trained model on the test split:

```bash
wastevision evaluate
```

On the test split it reaches an accuracy of 0.868 and a macro-F1 of 0.819 (baseline: 0.730 and 0.720). It is better than the baseline for every class except trash, the smallest class. See [docs/evaluation.md](docs/evaluation.md) for the settings, the class-weight comparison, the training curves, the per-class test results, the confusion matrix, the calibration and the error analysis.

Show the most confident mistakes and Grad-CAM maps of the regions the network used:

```bash
wastevision error-analysis --mistakes 16
```

Check how the model copes with changed test images (blur, brightness, JPEG compression, rotation) and with real-world photos from a waste facility (the RealWaste dataset, downloaded separately into `data/real/`):

```bash
wastevision robustness
wastevision evaluate-real
```

The changed test images cost at most about 2 points of accuracy. On the 3,587 RealWaste photos of the six matching classes, accuracy falls to 0.484 and macro-F1 to 0.456, while the model stays confident (mean confidence 0.837). The test split overstates real-world accuracy; see [docs/evaluation.md](docs/evaluation.md#real-world-photos-and-robustness).

The numbers above are for version 1 of the model. Two improvements were then compared on the validation split: stronger augmentation (RandAugment) made it worse, label smoothing 0.1 made it slightly better and is now the default of `wastevision train` (version 2). Version 2 reaches a test accuracy of 0.892 and a macro-F1 of 0.867, with trash F1 rising from 0.450 to 0.640. On RealWaste it is less accurate (0.432) but far less overconfident (mean confidence 0.540, calibration error 0.108 instead of 0.353). The test split has been used twice, once for each version. See [docs/evaluation.md](docs/evaluation.md#model-versions).

## Exported model

The final model (version 2) is committed in ONNX format, so the service can run it with `onnxruntime` and does not need PyTorch:

| File | Content |
|:--|:--|
| [models/waste-mobilenetv3.onnx](models/waste-mobilenetv3.onnx) | The network (6.1 MB, ONNX opset 18, any batch size, input `image` of shape N × 3 × 224 × 224, output `logits` with one score per class) |
| [models/labels.json](models/labels.json) | Class order and the user-facing category of each class |
| [models/metadata.json](models/metadata.json) | Preprocessing (resize, crop, mean and standard deviation), SHA-256 checksums of the dataset archive, the split, the checkpoint and the ONNX file, validation and test metrics, parity result and export date |

Export it again from a trained checkpoint (needs the `train` extra and the downloaded dataset):

```bash
wastevision export
```

The command compares the ONNX model with the PyTorch model on the first 50 test images and stops with an error unless the largest logit difference is below 0.0001 and the predicted class is the same for every image. For the committed model the largest difference is 0.000016 and all 50 predictions are equal.
