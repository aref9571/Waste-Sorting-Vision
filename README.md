# Waste Sorting Vision

Waste Sorting Vision is an image classification service that tells you which recycling category a photographed household item belongs to: cardboard, glass, metal and cans, paper, plastic, or other waste. It compares a classical image-processing baseline with a fine-tuned convolutional neural network, and evaluates both honestly, including on real photos taken outside the lab. The final model is served through a tested FastAPI service and a simple web page, and will be packaged with Docker.

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
| Image preprocessing without PyTorch, ONNX predictor and "unsure" confidence threshold | Done |
| Classification API (FastAPI) with upload validation and health checks | Done |
| Web page with photo upload, result and sorting hints | Done |
| Docker image | Planned |

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

## Predicting with the exported model

The predictor needs only `onnxruntime`, numpy and Pillow:

```python
from PIL import Image

from wastevision.inference.predictor import Predictor

predictor = Predictor.from_directory()
prediction = predictor.predict(Image.open("photo.jpg"))
print(prediction.category, prediction.top, prediction.unsure)
```

`prediction.top` lists the three most likely classes with their probabilities, `prediction.category` is the sorting category of the most likely class, and `prediction.unsure` is true when the highest probability is below the confidence threshold stored in `models/metadata.json`. The threshold (0.4131) was chosen on the validation split so that at least 95% of the answered validation images are correct; it answers 94.7% of the validation images and 91.3% of the test images, with 95.3% and 93.9% correct. See [docs/evaluation.md](docs/evaluation.md#confidence-threshold-unsure-answer). Choose it again with:

```bash
wastevision threshold
```

## Classification API

Start the service on port 8000 (it loads the model from `models/` once at start-up):

```bash
uvicorn wastevision.web.app:create_app --factory --port 8000
```

Classify a photo:

```bash
curl -F "image=@photo.jpg;type=image/jpeg" http://localhost:8000/api/v1/classify
```

Example response (the numbers depend on the photo):

```json
{
  "category": "Glass",
  "label": "glass",
  "confidence": 0.87,
  "unsure": false,
  "top3": [
    {"label": "glass", "probability": 0.87},
    {"label": "plastic", "probability": 0.06},
    {"label": "metal", "probability": 0.03}
  ]
}
```

| Endpoint | Purpose |
|:--|:--|
| `POST /api/v1/classify` | Multipart form with one field `image` (JPEG, PNG or WebP) |
| `GET /health/live` | The process is running |
| `GET /health/ready` | The model is loaded; answers 503 if it could not be loaded |

Uploads are checked before the image is decoded: the content type must be `image/jpeg`, `image/png` or `image/webp` and match the file content (otherwise 415), the file must not be larger than 5 MB (413), and the image must not have more than 40 megapixels (400). The pixel count is read from the file header, and Pillow's own decompression-bomb limit is set to the same value. Then Pillow verifies the file and decodes it again; a file that cannot be read gives 400. Errors are returned as problem details (RFC 9457, `application/problem+json`) with a short description; they never repeat the uploaded content.

The service is configured with environment variables:

| Variable | Default | Meaning |
|:--|:--|:--|
| `WASTEVISION_MODELS_DIR` | `models` | Folder with the ONNX model and `metadata.json` |
| `WASTEVISION_MAX_UPLOAD_BYTES` | `5242880` (5 MB) | Largest accepted upload |
| `WASTEVISION_MAX_IMAGE_PIXELS` | `40000000` | Largest accepted width × height |
| `WASTEVISION_RATE_LIMIT_PER_MINUTE` | `0` (off) | Classification requests allowed per client address per minute |

The rate limit is basic protection only: it counts requests per client address in one-minute windows inside a single process, so every worker process counts separately, the count is lost on restart, and behind a reverse proxy the client address is only correct if uvicorn is set up to trust the proxy's forwarded headers. The size limit is checked after the upload has been received, so a public deployment should also limit the request body size in its reverse proxy. The API has no authentication, because it is a public demo that stores nothing.

## Web page

The same service serves a web page at `http://localhost:8000/` (start it with the command above). Choose or take a photo, press Classify, and the page shows the category, a confidence bar, the three most likely classes and a short generic sorting hint. If the confidence is below the threshold stored in `models/metadata.json`, the page says it is not sure and shows only the best guess.

The page is a normal HTML form: it works without JavaScript, and the result is a page returned by the server. A small optional script only shows a preview of the chosen photo. Uploads go through the same validation, limits and rate limit as the API; problems are shown on the page with the same status codes. The uploaded photo is not stored and not shown again.

The sorting hints are general advice. Recycling rules differ between cities and collection services, so check your local rules.

## Licence

The code is released under the MIT License, see [LICENSE](LICENSE). The datasets used for training and evaluation have their own licences, listed in [NOTICE.md](NOTICE.md).
