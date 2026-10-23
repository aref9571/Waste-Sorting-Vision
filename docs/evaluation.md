# Evaluation

All results use the fixed split in [splits/trashnet-split.csv](../splits/trashnet-split.csv) (SHA-256 `29db88fff705dfe6ee2ca24eec9e99fe4d494f4ecea9bdb36b06a39f4907e419`, 1,765 train, 378 validation and 378 test images, see [dataset.md](dataset.md)). Models and settings are chosen on the validation split only. Each final model is evaluated once on the test split. The main metric is macro-F1, because the classes are imbalanced (the test split has 21 trash images and 89 paper images).

## Classical baseline

The baseline shows how far hand-crafted image features get before any deep learning is used.

### Features

Every image is converted to RGB and resized to 128 × 128 pixels. Three descriptors are concatenated into one vector of 1,902 values:

| Descriptor | Settings | Values | What it captures |
|---|---|---|---|
| HSV colour histogram | 8 hue × 4 saturation × 4 value bins, normalised to sum 1 | 128 | Overall colour, independent of where it is in the image |
| Local binary patterns (LBP) | Grayscale, uniform patterns, 8 neighbours, radius 1, normalised histogram | 10 | Fine surface texture (smooth, edges, corners, spots) |
| Histogram of oriented gradients (HOG) | Grayscale, 9 orientations, 16 × 16 pixel cells, 2 × 2 cell blocks | 1,764 | Shape: the direction of edges in each region of the image |

The features are computed with scikit-image ([src/wastevision/baseline/features.py](../src/wastevision/baseline/features.py)).

### Model selection on validation

Each candidate is a scikit-learn pipeline that standardises the features and then fits the classifier on the train split. It is scored on the validation split, and the candidate with the highest validation macro-F1 is kept (on a tie, the first one in the table).

| Model | Parameters | Validation accuracy | Validation macro-F1 |
|---|---|---|---|
| Logistic regression | C = 0.001 | 0.672 | 0.657 |
| **Logistic regression** | **C = 0.01** | **0.704** | **0.699** |
| Logistic regression | C = 0.1 | 0.693 | 0.682 |
| Logistic regression | C = 1 | 0.693 | 0.680 |
| SVM, RBF kernel | C = 1, gamma = scale | 0.672 | 0.653 |
| SVM, RBF kernel | C = 1, gamma = 0.0001 | 0.611 | 0.541 |
| SVM, RBF kernel | C = 10, gamma = scale | 0.698 | 0.693 |
| SVM, RBF kernel | C = 10, gamma = 0.0001 | 0.688 | 0.678 |
| SVM, RBF kernel | C = 100, gamma = scale | 0.698 | 0.693 |
| SVM, RBF kernel | C = 100, gamma = 0.0001 | 0.696 | 0.693 |

The selected model is logistic regression with C = 0.01, a strongly regularised linear model. The best RBF SVMs are close behind (0.693), so the extra flexibility of a non-linear kernel does not help with these features.

### Test result

The selected pipeline was evaluated once on the test split (378 images):

| Metric | Value |
|---|---|
| Accuracy | 0.730 |
| Macro-F1 | 0.720 |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| cardboard | 0.833 | 0.833 | 0.833 | 60 |
| glass | 0.692 | 0.720 | 0.706 | 75 |
| metal | 0.638 | 0.607 | 0.622 | 61 |
| paper | 0.817 | 0.854 | 0.835 | 89 |
| plastic | 0.627 | 0.653 | 0.640 | 72 |
| trash | 0.857 | 0.571 | 0.686 | 21 |

The test scores are slightly higher than the validation scores. With 378 images, one image changes accuracy by about 0.3 percentage points, so a difference of this size is within the normal variation between two samples of the same data.

Cardboard and paper are recognised best: they have typical colours and flat textures. Glass, metal and plastic are confused with each other most often: the largest errors in the test confusion matrix are glass predicted as plastic (11 images) and metal predicted as glass or plastic (9 images each). A likely reason is that these materials are shiny, often transparent or grey, and their appearance depends on reflections that colour, texture and edge histograms describe poorly. Trash has a high precision but a low recall: the model predicts it only 14 times for 21 trash images, which fits the small number of trash images in training.

### Reproduce

```bash
pip install -e ".[train,dev]"
wastevision data download
wastevision baseline train
wastevision baseline evaluate
```

`baseline train` prints the validation table and saves the selected pipeline to `runs/baseline/model.joblib` (not committed). `baseline evaluate` loads it and prints the test metrics. Both commands are deterministic: running them again gives the same numbers.

Measured on 2026-10-23 on a 4-core Intel Xeon CPU at 2.10 GHz with Python 3.12.3, scikit-learn 1.9.1 and scikit-image 0.26.0: feature extraction takes about 11 ms per image, `baseline train` about 70 seconds in total (feature extraction for train and validation plus all ten candidates), `baseline evaluate` about 6 seconds.
