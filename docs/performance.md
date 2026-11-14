# Performance

This page records how fast the classification service answers, where the time goes, and what a change to JPEG decoding bought. All numbers were measured with the commands below; they depend on the machine and should be measured again on other hardware.

## Setup

| Item | Value |
|---|---|
| Machine | 4 virtual CPU cores (Intel Xeon, one thread per core), 16 GB memory, Linux, no GPU |
| Software | Python 3.12.3, onnxruntime 1.32.0 (CPU provider), Pillow 13.0.0, NumPy 2.5.4, FastAPI 0.162.7, uvicorn 0.54.0 |
| Server | `uvicorn wastevision.web.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1`: one process, one worker, no rate limit |
| Client | `scripts/benchmark.py` on the same machine, one request at a time, over `127.0.0.1` |
| Requests | 10 warm-up requests (not counted), then 200 measured requests per image |
| Model | `models/waste-mobilenetv3.onnx` (version 2) |
| Not measured | Docker (no container daemon was available), other hardware, concurrent clients |

The test images are made by the script itself (`--synthetic WIDTHxHEIGHT`): a smooth colour gradient with seeded Gaussian noise, saved as JPEG with quality 85. They are not photos of waste, so their file sizes are only roughly comparable with phone photos. The 12 megapixel image (4000 × 3000) is 3.1 MB, under the service's 5 MB upload limit. The 64 × 48 image is `tests/fixtures/sample.png`.

## Commands

```bash
uvicorn wastevision.web.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1

python scripts/benchmark.py service --image tests/fixtures/sample.png
python scripts/benchmark.py service --synthetic 640x480
python scripts/benchmark.py service --synthetic 1920x1080
python scripts/benchmark.py service --synthetic 4000x3000

python scripts/benchmark.py model --synthetic 4000x3000
```

`service` posts the same image to `/api/v1/classify` and times each request from sending to the end of the answer. `model` does not use the server: it loads the model like the service does and times four steps separately, each 200 times on the same image: decoding the JPEG, preprocessing (rotate by EXIF, resize, crop, normalise), the ONNX inference alone, and `Predictor.predict` (preprocessing plus inference plus the probabilities). Both print JSON with the image size, the number of requests, the mean, p50, p95, p99 and maximum in milliseconds, and the throughput. Options: `--requests`, `--warmup`, `--url`, `--models-dir`.

## Where the time goes

Model-only timings in milliseconds (p50), with the full-resolution decoding that the service used before the change below:

| Image | Decode | Preprocess | Inference | Predict (preprocess + inference) |
|---|---|---|---|---|
| 64 × 48 PNG | 0.07 | 1.04 | 1.06 | 2.23 |
| 640 × 480 JPEG (0.08 MB) | 1.47 | 2.15 | 0.97 | 3.36 |
| 1920 × 1080 JPEG (0.5 MB) | 9.02 | 8.50 | 1.22 | 11.08 |
| 4000 × 3000 JPEG (3.1 MB) | 59.58 | 78.17 | 1.56 | 73.91 |

The network itself needs 1 to 1.6 ms. Everything else is image handling. For a 12 megapixel photo, decoding plus resizing takes about 140 ms and the network 1.6 ms, so about 99% of the time goes to getting the pixels ready. The run for 4000 × 3000 shows this directly: the model is cheap, the photo is large.

(The `predict` column is a separate run of its own, so it does not equal the sum of the other columns.)

## Service latency

Before the decoding change (full-resolution decode of every upload), 200 requests, one at a time:

| Image | p50 | p95 | p99 | max | mean | Requests per second |
|---|---|---|---|---|---|---|
| 64 × 48 PNG | 3.85 | 5.09 | 5.31 | 7.98 | 4.19 | 238.7 |
| 640 × 480 JPEG | 6.38 | 6.83 | 6.98 | 7.55 | 6.43 | 155.5 |
| 1920 × 1080 JPEG | 33.05 | 40.59 | 44.69 | 45.02 | 33.50 | 29.8 |
| 4000 × 3000 JPEG | 149.72 | 187.53 | 214.32 | 289.76 | 148.44 | 6.7 |

After the change (JPEG uploads are decoded at a reduced size, see below):

| Image | p50 | p95 | p99 | max | mean | Requests per second |
|---|---|---|---|---|---|---|
| 64 × 48 PNG | 3.60 | 3.86 | 4.45 | 6.87 | 3.65 | 274.1 |
| 640 × 480 JPEG | 6.49 | 8.52 | 10.29 | 10.99 | 6.67 | 149.9 |
| 1920 × 1080 JPEG | 11.50 | 14.16 | 16.21 | 19.83 | 11.86 | 84.3 |
| 4000 × 3000 JPEG | 43.56 | 51.28 | 57.98 | 60.25 | 44.47 | 22.5 |

All times are in milliseconds. "Requests per second" is 200 divided by the total time of the measured requests with a single client, so it describes this one-at-a-time test, not the most the server can take.

What this shows:

- Small images are not affected: 64 × 48 and 640 × 480 are within noise (3.9 to 3.6 ms and 6.4 to 6.5 ms).
- A 12 megapixel JPEG is about 3.4 times faster at p50 (149.7 to 43.6 ms) and its p99 fell from 214 ms to 58 ms. A 1080p JPEG is about 2.9 times faster (33.1 to 11.5 ms).
- Even after the change, a 12 megapixel upload costs about 43 ms against about 4 ms for a small image. The rest is the work that remains (checking the upload, decoding at the reduced size, resizing, the network and the web framework); it was not broken down further after the change.
- Percentiles tell more than the mean here. For the 12 megapixel image before the change the mean was 148 ms and the p50 150 ms, but the slowest request took 290 ms and the p99 was 214 ms. A user who is unlucky sees the tail, not the middle.

## The change: decode large JPEGs at a reduced size

The network only sees a 224 × 224 crop of an image resized to 256 pixels on the shorter side. Decoding 12 million pixels to throw most of them away is wasteful. JPEG can be decoded at 1/2, 1/4 or 1/8 of the size directly in the decoder (Pillow `Image.draft`), which is much cheaper than decoding in full and shrinking afterwards.

`decode_image` in `src/wastevision/web/validation.py` now takes the shorter side the model needs (256, read from the model's preprocessing settings) and asks Pillow for the smallest reduction that keeps both sides at least 256 pixels. A 4000 × 3000 photo is decoded at 500 × 375. Then the normal preprocessing runs. Smaller images, and PNG and WebP uploads, are decoded in full as before. The checks that come first (type, size in bytes, pixel limit, a readable file) are unchanged, and a damaged JPEG is still rejected.

### Cost of the change

Reduced decoding does not give exactly the same pixels as decoding in full and resizing, so predictions can change slightly. It was measured on the 378 test-split images of the dataset (not used to choose anything): each was enlarged to 2048 × 1536 pixels, saved as JPEG with quality 90, and classified once decoded in full and once decoded at a reduced size (512 × 384).

| Measure | Result |
|---|---|
| Same predicted class | 376 of 378 (99.5%) |
| Same "unsure" decision | 374 of 378 (98.9%) |
| Mean change of the confidence | 0.008 |
| Largest change of the confidence | 0.071 |

So about one image in 200 changes class, and about one in 100 changes the unsure decision. These images are enlarged dataset photos and therefore smoother than real phone photos, which probably makes the test favourable to the change; the real difference on phone photos was not measured. Images at most 512 × 384 (the size of the dataset images) are never reduced, so the numbers in `docs/evaluation.md` are not affected. The preprocessing used for training and for the parity test is unchanged.

## Limits of these numbers

- One machine, one kind of synthetic image, a client on the same machine that takes some of the CPU.
- One request at a time. Latency with several clients at once was not measured, and neither was the largest load the server can take.
- The server was not run inside Docker, so the container's overhead is not included.
- The first requests after start-up are slower; the 10 warm-up requests hide that.
