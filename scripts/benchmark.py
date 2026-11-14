import argparse
import json
import math
import statistics
import time
import urllib.request
import uuid
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from wastevision.inference.predictor import Predictor
from wastevision.inference.preprocess import preprocess_image

DEFAULT_URL = "http://127.0.0.1:8000/api/v1/classify"
DEFAULT_REQUESTS = 200
DEFAULT_WARMUP = 10
SYNTHETIC_SEED = 7
SYNTHETIC_JPEG_QUALITY = 85


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile needs at least one value")
    rank = math.ceil(fraction * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


def summarise(durations: list[float]) -> dict[str, float]:
    ordered = sorted(durations)
    return {
        "count": len(ordered),
        "mean_ms": round(statistics.fmean(ordered) * 1000, 2),
        "p50_ms": round(percentile(ordered, 0.50) * 1000, 2),
        "p95_ms": round(percentile(ordered, 0.95) * 1000, 2),
        "p99_ms": round(percentile(ordered, 0.99) * 1000, 2),
        "max_ms": round(ordered[-1] * 1000, 2),
        "throughput_per_s": round(len(ordered) / sum(ordered), 1),
    }


def synthetic_jpeg(width: int, height: int) -> bytes:
    generator = np.random.default_rng(SYNTHETIC_SEED)
    columns = np.linspace(0, 255, width, dtype=np.float32)
    rows = np.linspace(0, 255, height, dtype=np.float32)
    base = np.stack(
        [
            np.add.outer(rows * 0.3, columns * 0.7),
            np.add.outer(rows * 0.6, columns * 0.2),
            np.add.outer(rows * 0.1, columns * 0.9),
        ],
        axis=-1,
    )
    noise = generator.normal(0, 12, size=(height, width, 3)).astype(np.float32)
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=SYNTHETIC_JPEG_QUALITY)
    return buffer.getvalue()


def parse_size(text: str) -> tuple[int, int]:
    width, separator, height = text.lower().partition("x")
    if not separator or not width.isdigit() or not height.isdigit():
        raise argparse.ArgumentTypeError("size must look like 4000x3000")
    return int(width), int(height)


def image_content_type(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        return Image.MIME[image.format]


def multipart_body(data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; filename="benchmark"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + data + tail, f"multipart/form-data; boundary={boundary}"


def post_once(url: str, body: bytes, header: str) -> float:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": header})
    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        response.read()
    return time.perf_counter() - start


def measure(action: Callable[[], object], requests: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        action()
    durations = []
    for _ in range(requests):
        start = time.perf_counter()
        action()
        durations.append(time.perf_counter() - start)
    return durations


def benchmark_service(data: bytes, url: str, requests: int, warmup: int) -> dict[str, float]:
    body, header = multipart_body(data, image_content_type(data))
    return summarise(measure(lambda: post_once(url, body, header), requests, warmup))


def benchmark_model(
    data: bytes, models_dir: Path, requests: int, warmup: int
) -> dict[str, dict[str, float]]:
    predictor = Predictor.from_directory(models_dir)
    with Image.open(BytesIO(data)) as opened:
        opened.load()
        image = opened.copy()
    tensor = preprocess_image(image, predictor.preprocessing)[np.newaxis]
    feeds = {predictor.input_name: tensor}
    outputs = [predictor.output_name]

    def decode() -> None:
        with Image.open(BytesIO(data)) as opened:
            opened.load()

    return {
        "decode": summarise(measure(decode, requests, warmup)),
        "preprocess": summarise(
            measure(lambda: preprocess_image(image, predictor.preprocessing), requests, warmup)
        ),
        "inference": summarise(
            measure(lambda: predictor.session.run(outputs, feeds), requests, warmup)
        ),
        "predict": summarise(measure(lambda: predictor.predict(image), requests, warmup)),
    }


def load_image_bytes(path: Path | None, size: tuple[int, int]) -> bytes:
    if path is not None:
        return path.read_bytes()
    return synthetic_jpeg(*size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latency benchmark for the classification service")
    parser.add_argument("mode", choices=["service", "model"])
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--synthetic", type=parse_size, default=(4000, 3000))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    data = load_image_bytes(arguments.image, arguments.synthetic)
    with Image.open(BytesIO(data)) as opened:
        details = {"width": opened.width, "height": opened.height, "bytes": len(data)}
    if arguments.mode == "service":
        results = benchmark_service(data, arguments.url, arguments.requests, arguments.warmup)
    else:
        results = benchmark_model(data, arguments.models_dir, arguments.requests, arguments.warmup)
    print(json.dumps({"image": details, "results": results}, indent=2))


if __name__ == "__main__":
    main()
