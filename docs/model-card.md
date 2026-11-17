# Model card: waste-mobilenetv3 (version 2)

This card describes the model that the service uses: what it is for, what it was trained and measured on, how well it works, and where it should not be trusted. All numbers come from [evaluation.md](evaluation.md) and [models/metadata.json](../models/metadata.json). Where a number was not measured for this version, the card says so.

## Model details

| Item | Value |
|---|---|
| Task | Image classification into six waste classes: cardboard, glass, metal, paper, plastic, trash |
| User-facing categories | Cardboard, Glass, Metal and cans, Paper, Plastic, Other waste ([models/labels.json](../models/labels.json)) |
| Architecture | MobileNetV3-Small, pretrained on ImageNet (torchvision `IMAGENET1K_V1` weights), last layer replaced by a six-class layer |
| Training | Two stages on the train split: classifier only, then the whole network. AdamW, class-weighted cross-entropy with label smoothing 0.1, early stopping on validation macro-F1, seed 42 |
| Version | 2 (label smoothing). Version 1 (no label smoothing) is described in the evaluation for comparison only |
| Format | ONNX, opset 18, 6.1 MB, served with `onnxruntime` on CPU |
| Input | One RGB image. Resize the shorter side to 256 pixels, centre crop 224 × 224, scale to [0, 1], normalise with the ImageNet mean and standard deviation |
| Output | Six scores (logits). The service turns them into probabilities and returns the best class, the three most likely classes and an "unsure" flag |
| Model file SHA-256 | `a8c5c99f973d970248db2a72a4fae839ababdf9d52901c9ed65353e72bb99eb3` |
| Exported on | 2026-11-07 |
| Code licence | MIT, see [LICENSE](../LICENSE) |

The ONNX file gives the same answers as the PyTorch checkpoint: on 50 test images the largest logit difference is 1.6e-5 and all 50 predicted classes are equal.

## Intended use

- Helping a person decide which household recycling category a single item belongs to, as a suggestion to check against local rules.
- Learning and demonstration: comparing a classical image-processing baseline with transfer learning, measuring calibration and domain shift, and serving a model without PyTorch.

The best-case input is one item, clearly visible, on a plain light background, like the training photos.

## Not intended for

- Industrial or municipal sorting lines, or any process where a wrong answer has a cost without a person checking it.
- Safety-critical decisions. The model has six classes only. It does not know batteries, electronics, medical waste, chemicals, food waste, textiles or any other hazardous or special waste, and it will still answer with one of the six classes for them.
- Deciding whether an item is recyclable. The label "trash" in the training data means "does not belong to the other five classes", not "not recyclable", and recycling rules differ between cities and collection services.
- Photos with several items, or scenes where the item is small or partly hidden.
- Identifying or analysing people. The model has no person category, and the service does not store uploaded photos.

## Training data

TrashNet (Thung and Yang, Stanford CS 229, 2016, MIT licence): 2,527 photos of single items on a white posterboard, 512 × 384 pixels, taken with phones. Six exact duplicates with conflicting labels were removed, so the split holds 2,521 images: 1,765 for training, 378 for validation and 378 for testing, stratified by class, with seed 42, and with exact duplicates kept together ([dataset.md](dataset.md)).

Known problems of this data:

- **One setting.** Almost every photo has the same white background and similar light. The model has learned to expect that look, and the error analysis found that it depends on it.
- **Small and unbalanced.** Trash has only 95 training images, against 416 for paper. Its results are the least reliable: each of the 21 test trash images moves its recall by about 5 points.
- **Labels mix material and use.** Trash contains items such as snack wrappers and crumpled paper-like items. From a photo alone, a recyclable plastic bottle and a non-recyclable plastic wrapper can look the same.
- **Narrow origin.** The photos come from one research project. They do not represent the packaging, bins or waste of every country, and the dataset has no information about where each item came from.

## Evaluation data

- **Test split:** 378 TrashNet images, held out. It has been used twice, once for each model version, and nothing was chosen from it ([evaluation.md](evaluation.md#version-2-on-the-test-split-and-on-realwaste)).
- **RealWaste** (Single et al., 2023, CC BY-NC-SA 4.0): 3,587 photos from a waste facility in Australia, items lying alone on a grey floor, many dirty, crushed or broken. Only the six classes that match this project were used. Its labels do not always match TrashNet's (for example, plastic bags and dirty food tubs are plastic there). The photos were used for measurement only, never for training or choosing the model, and they are not part of this repository.
- The project does not contain photos taken by its author. No measurement exists for household photos taken with a phone on ordinary backgrounds, and the RealWaste result is the closest available guide.

## Results

### TrashNet test split (378 images)

| Metric | Version 1 | Version 2 (this model) |
|---|---|---|
| Accuracy | 0.868 | **0.892** |
| Macro-F1 (main metric) | 0.819 | **0.867** |
| Validation accuracy / macro-F1 | 0.926 / 0.919 | 0.929 / 0.928 |
| Expected calibration error | 0.069 | 0.147 |

The classical baseline (colour, texture and shape features with logistic regression) reaches 0.730 accuracy and 0.720 macro-F1 on the same split.

| Class | Test images | Version 2 test F1 | Version 2 RealWaste F1 |
|---|---|---|---|
| cardboard | 60 | 0.984 | 0.326 |
| glass | 75 | 0.873 | 0.351 |
| metal | 61 | 0.915 | 0.563 |
| paper | 89 | 0.906 | 0.532 |
| plastic | 72 | 0.884 | 0.468 |
| trash | 21 | 0.640 | 0.313 |

Paper predicted as trash is still the most frequent mistake on the test split (10 images). Glass, plastic and metal are confused with each other, because shiny, clear or grey materials look alike in a photo.

### Real-world photos (RealWaste, 3,587 images)

| Metric | Version 1 | Version 2 (this model) |
|---|---|---|
| Accuracy | 0.484 | 0.432 |
| Macro-F1 | 0.456 | 0.426 |
| Mean confidence | 0.837 | 0.540 |
| Expected calibration error | 0.353 | 0.108 |

On photos from a different place the model loses about half of its accuracy. It predicts trash for 1,243 of the 3,587 photos, because dirty, crushed or irregular items of every material look like trash to it. Part of the drop comes from different label definitions between the two datasets, but not all of it. Label smoothing did not fix this: version 2 is less accurate than version 1 on RealWaste, though much better calibrated.

### Robustness

Blur, brightness (±40%), JPEG quality 30 and a 15 degree rotation each cost at most about 2 points of accuracy. This was measured for version 1 only and was not repeated for version 2, to keep the second use of the test split to a single evaluation. Version 2 was not tested on photos with changed backgrounds.

## Unsure behaviour

The service answers "unsure" when the highest class probability is below **0.4131**. The threshold was chosen on the validation split only: the lowest value for which at least 95% of the answered validation images are correct.

| | Images | Answered | Coverage | Accuracy of answered images |
|---|---|---|---|---|
| Validation (threshold chosen here) | 378 | 358 | 0.947 | 0.953 |
| Test (measurement only) | 378 | 345 | 0.913 | 0.939 |

Read this with care:

- The threshold is low because label smoothing makes the model underconfident. A probability of 0.41 is already a clear answer for this model.
- On the test split the answered images are 93.9% correct, below the 95% target. A small shortfall on new images is expected.
- The threshold was not measured on RealWaste. On photos from other places, the model is wrong far more often at the same confidence, so "not unsure" does not mean "correct".
- Some wrong answers have high confidence. For version 1, 16 of 50 test mistakes had a confidence above 0.9. The flag reduces wrong answers but cannot remove them.

## Limitations

- **Domain shift** is the main weakness. Expect much lower accuracy than the test split shows for photos with ordinary backgrounds, several items, dirt, damage or unusual packaging.
- **Six classes only.** Anything else is forced into one of them.
- **Trash is the weakest class** and its definition is not material-based.
- **No local rules.** The model cannot know what a city accepts. The web page shows generic hints for each category and says to check local rules.
- **Single measurements.** Each number comes from one trained model with seed 42. There is no estimate of the variation between training runs, and the choices between versions rest on differences of one to a few images (for example label smoothing: 0.9286 against 0.9259 validation accuracy, one image out of 378).
- **Measured on one machine.** Speed and the training time below were measured on a 4-core CPU only ([performance.md](performance.md)).

## Ethical considerations

- **Harm from wrong answers.** A wrong suggestion puts an item in the wrong bin. Wrongly sorted items can contaminate a recycling batch, and a confident wrong answer can mislead a person who trusts it. The service reports its three best guesses and an unsure flag, and the page describes the answer as a suggestion.
- **Honest limits.** The accuracy of 0.892 describes TrashNet-like photos. Quoting it for real use would be misleading, so this card and the evaluation give the RealWaste result next to it.
- **Privacy.** Uploaded photos are checked, decoded in memory, classified and not stored. The service has no accounts and no authentication, so it should not be exposed publicly without a limit on request size at the proxy. A photo can show people or places in its background. The service does not look for them and keeps nothing, but a deployment that logs or stores uploads would need its own privacy review.
- **Data rights.** TrashNet is MIT licensed and RealWaste is CC BY-NC-SA 4.0. Neither dataset's images are redistributed here. RealWaste was used only for measurement, and its licence does not allow commercial use, so anyone who wants to evaluate or train with it commercially must check the licence themselves. The ImageNet weights of the starting network come from torchvision, whose terms apply to them.
- **Fairness.** The data contains no people, so there is no group of people for which it can be unfair in the usual sense. The unequal part is geographic and material: the model works best on items and packaging that look like the 2016 TrashNet photos and worst on what that collection did not show.

## Energy use

The final model (version 2) trains in about 9 minutes on a 4-core Intel Xeon CPU at 2.10 GHz, with no GPU: about 2 minutes for stage 1 and 7 minutes for stage 2 (measured 2026-11-06, PyTorch 2.14.1). The power draw was not measured. If the whole machine drew 100 W, which is assumed and probably on the high side for four cores, nine minutes would use about 0.015 kWh.

Developing the model took more than that. Four variants were trained for the comparisons (each at most 16 minutes in both stages), plus the classical baseline (about 70 seconds) and several evaluation runs of seconds to a few minutes each. This adds up to at most about 70 minutes of CPU time, or about 0.12 kWh at the same assumed 100 W. Serving costs little: a 12 megapixel photo takes about 44 ms and a small image about 4 ms on one CPU worker, and the Docker image does not contain PyTorch ([performance.md](performance.md)).

These are estimates from measured time and an assumed power, not measurements of energy. They exclude the data centre and the energy used to make the hardware.

## Sustainability and its limits

The project's purpose is to help people sort waste correctly: the less recyclable material that ends up in the wrong bin, the more can be recovered. The model is small enough to run on a CPU and the training cost is small, which fits the purpose.

Its contribution is limited, and this should be said plainly. A classifier of single items on a white background addresses only a small part of the sorting problem. Whether waste is recycled depends on local collection rules, how clean the item is, the market for the material and the sorting plants, and none of that is visible in a photo. The real-world result, an accuracy below 0.5 on photos from another place, shows that the model is a demonstration of method and measurement, not a tool ready to run unchecked. A sensible next step would be training on varied backgrounds and conditions and measuring on household photos, which this project has not done.

## Licence and credit

- Code and the model files in this repository: MIT, copyright 2026 aref ([LICENSE](../LICENSE)).
- Dataset credit and licences: [NOTICE.md](../NOTICE.md). TrashNet: Gary Thung and Mindy Yang, https://github.com/garythung/trashnet. RealWaste: S. Single, S. Iranmanesh and R. Raad, Information 14(12), 633, 2023.

## Reproduce

```bash
pip install -e ".[train,dev]"
wastevision data download
wastevision train --stage head
wastevision train --stage full
wastevision evaluate
wastevision threshold
wastevision export
```

Details, options and the commands for the other measurements are in [evaluation.md](evaluation.md).
