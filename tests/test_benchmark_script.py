import argparse
import importlib.util
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentile_uses_the_nearest_rank(benchmark):
    values = [float(number) for number in range(1, 101)]
    assert benchmark.percentile(values, 0.50) == 50.0
    assert benchmark.percentile(values, 0.95) == 95.0
    assert benchmark.percentile(values, 0.99) == 99.0


def test_percentile_of_one_value_is_that_value(benchmark):
    assert benchmark.percentile([0.25], 0.99) == 0.25


def test_percentile_rejects_an_empty_list(benchmark):
    with pytest.raises(ValueError, match="at least one value"):
        benchmark.percentile([], 0.5)


def test_summarise_reports_milliseconds_and_throughput(benchmark):
    summary = benchmark.summarise([0.010, 0.020, 0.030, 0.040])
    assert summary["count"] == 4
    assert summary["mean_ms"] == 25.0
    assert summary["p50_ms"] == 20.0
    assert summary["max_ms"] == 40.0
    assert summary["throughput_per_s"] == 40.0


def test_summarise_shows_a_slow_outlier_in_the_tail_not_the_median(benchmark):
    summary = benchmark.summarise([0.010] * 99 + [1.0])
    assert summary["p50_ms"] == 10.0
    assert summary["p99_ms"] == 10.0
    assert summary["max_ms"] == 1000.0
    assert summary["mean_ms"] > summary["p50_ms"]


def test_parse_size_reads_width_and_height(benchmark):
    assert benchmark.parse_size("4000x3000") == (4000, 3000)


@pytest.mark.parametrize("text", ["4000", "axb", "4000x", "x300", "-1x5"])
def test_parse_size_rejects_bad_text(benchmark, text):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_size(text)


def test_synthetic_jpeg_has_the_requested_size_and_is_repeatable(benchmark):
    first = benchmark.synthetic_jpeg(320, 240)
    assert first == benchmark.synthetic_jpeg(320, 240)
    with Image.open(BytesIO(first)) as image:
        assert image.format == "JPEG"
        assert image.size == (320, 240)


def test_multipart_body_wraps_the_image_bytes(benchmark):
    body, header = benchmark.multipart_body(b"DATA", "image/png")
    boundary = header.split("boundary=")[1]
    assert header.startswith("multipart/form-data")
    assert body.startswith(f"--{boundary}\r\n".encode())
    assert b'name="image"' in body
    assert b"Content-Type: image/png" in body
    assert b"\r\n\r\nDATA\r\n" in body
    assert body.endswith(f"--{boundary}--\r\n".encode())


def test_image_content_type_follows_the_real_format(benchmark):
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    assert benchmark.image_content_type(buffer.getvalue()) == "image/png"


def test_measure_runs_warmup_without_recording_it(benchmark):
    calls = []
    durations = benchmark.measure(lambda: calls.append(1), requests=5, warmup=2)
    assert len(calls) == 7
    assert len(durations) == 5
