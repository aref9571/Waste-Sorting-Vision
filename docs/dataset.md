# Dataset

## Source

The project uses the resized version of the TrashNet dataset by Gary Thung and Mindy Yang (Stanford CS 229, 2016).

- Repository: https://github.com/garythung/trashnet
- Archive used here: `data/dataset-resized.zip` from that repository, pinned to commit `6fa2b878c6c1b4304b91109070ce0edf9279bb31`:
  https://raw.githubusercontent.com/garythung/trashnet/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip
- The dataset README also links a copy on Hugging Face (https://huggingface.co/datasets/garythung/trashnet). This project uses the GitHub archive because a fixed commit makes the download reproducible.

## Licence and citation

TrashNet is distributed under the MIT License (Copyright (c) 2017 Gary Thung). The full licence text and the credit are in [NOTICE.md](../NOTICE.md). The authors ask users to cite the repository:

> Gary Thung and Mindy Yang. trashnet. https://github.com/garythung/trashnet

The images are not stored in this repository. They are downloaded into `data/`, which is ignored by Git.

## Download

```bash
wastevision data download
```

The command:

1. downloads the archive to `data/raw/dataset-resized.zip` (skipped if the archive is already there),
2. checks its SHA-256 checksum and stops with an error if it does not match,
3. checks every path in the archive before extracting anything and refuses absolute paths, `..` segments and paths that would end up outside the target folder (zip-slip protection),
4. extracts the images to `data/raw/dataset-resized/<class>/`,
5. prints the number of images per class and compares it with the published counts.

If `data/raw/dataset-resized/` already contains all class folders, nothing is downloaded again. Use `--raw-dir` to choose another folder.

## Checksum

| File | Size | SHA-256 |
|---|---|---|
| `dataset-resized.zip` | 42,834,870 bytes | `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2` |

The same value is the constant `DATASET_SHA256` in `src/wastevision/data/download.py`. Check it by hand with:

```bash
sha256sum data/raw/dataset-resized.zip
```

## Contents (measured on 2026-10-22)

| Class | Images |
|---|---:|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |
| **total** | **2,527** |

These counts were measured from the extracted files with `wastevision data download` and match the counts published by the dataset authors.

- All 2,527 images are RGB JPEG files of exactly 512 × 384 pixels (landscape).
- The classes are imbalanced: `trash` has 137 images, about a quarter of `paper` (594). This is why macro-F1 is the main metric.
- The archive also contains macOS metadata (`__MACOSX/` and `.DS_Store` files). The `__MACOSX/` folder is not kept after extraction, and only image files are counted.

The image format and size were measured with Pillow:

```bash
python -c "
from collections import Counter
from pathlib import Path
from PIL import Image
sizes = Counter()
for path in Path('data/raw/dataset-resized').glob('*/*.jpg'):
    with Image.open(path) as image:
        sizes[(image.format, image.mode, image.size)] += 1
print(sizes)
"
```

## Known bias

Every photo shows a single object on a white posterboard, taken with an iPhone in similar lighting. Real photos of household waste have cluttered backgrounds, several objects, different lighting and different cameras, so a model trained only on TrashNet is expected to perform worse on them. The project measures this on a separate set of real photos instead of assuming the TrashNet results carry over.
