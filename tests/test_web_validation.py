import struct
import zlib
from http import HTTPStatus
from io import BytesIO

import pytest
from PIL import Image

from wastevision.web.problems import ProblemError
from wastevision.web.validation import (
    check_pixel_count,
    check_upload_size,
    decode_image,
    expected_format,
    validate_upload,
)

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000


def encoded_image(image_format: str, size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (30, 80, 200)).save(buffer, format=image_format)
    return buffer.getvalue()


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def png_header_only(width: int, height: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(b"\x00" * 16))
        + png_chunk(b"IEND", b"")
    )


@pytest.fixture(autouse=True)
def default_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", MAX_PIXELS)


@pytest.mark.parametrize(
    ("content_type", "image_format"),
    [
        ("image/jpeg", "JPEG"),
        ("image/png", "PNG"),
        ("image/webp", "WEBP"),
        ("IMAGE/PNG", "PNG"),
        ("image/jpeg; charset=binary", "JPEG"),
    ],
)
def test_expected_format_accepts_the_allowed_content_types(content_type: str, image_format: str):
    assert expected_format(content_type) == image_format


@pytest.mark.parametrize(
    "content_type", ["image/gif", "image/svg+xml", "text/plain", "application/pdf", "", None]
)
def test_expected_format_rejects_other_content_types(content_type: str | None):
    with pytest.raises(ProblemError) as raised:
        expected_format(content_type)

    assert raised.value.status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_check_upload_size_accepts_a_file_at_the_limit():
    check_upload_size(b"x" * 10, max_bytes=10)


def test_check_upload_size_rejects_a_file_over_the_limit():
    with pytest.raises(ProblemError) as raised:
        check_upload_size(b"x" * 11, max_bytes=10)

    assert raised.value.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert "10 bytes" in raised.value.detail


def test_check_upload_size_rejects_an_empty_file():
    with pytest.raises(ProblemError) as raised:
        check_upload_size(b"", max_bytes=10)

    assert raised.value.status == HTTPStatus.BAD_REQUEST


def test_check_pixel_count_accepts_an_image_at_the_limit():
    check_pixel_count((100, 50), max_pixels=5000)


def test_check_pixel_count_rejects_an_image_over_the_limit():
    with pytest.raises(ProblemError) as raised:
        check_pixel_count((100, 51), max_pixels=5000)

    assert raised.value.status == HTTPStatus.BAD_REQUEST


@pytest.mark.parametrize(
    ("content_type", "image_format"),
    [("image/jpeg", "JPEG"), ("image/png", "PNG"), ("image/webp", "WEBP")],
)
def test_validate_upload_returns_the_decoded_image(content_type: str, image_format: str):
    image = validate_upload(encoded_image(image_format), content_type, MAX_BYTES, MAX_PIXELS)

    assert image.format == image_format
    assert image.size == (64, 48)
    assert image.getpixel((10, 10)) is not None


def test_validate_upload_checks_the_content_type_before_the_size():
    with pytest.raises(ProblemError) as raised:
        validate_upload(b"x" * 100, "image/gif", max_bytes=10, max_pixels=MAX_PIXELS)

    assert raised.value.status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_validate_upload_rejects_content_that_does_not_match_the_content_type():
    with pytest.raises(ProblemError) as raised:
        validate_upload(encoded_image("PNG"), "image/jpeg", MAX_BYTES, MAX_PIXELS)

    assert raised.value.status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_validate_upload_rejects_an_allowed_type_holding_another_format():
    with pytest.raises(ProblemError) as raised:
        validate_upload(encoded_image("GIF"), "image/png", MAX_BYTES, MAX_PIXELS)

    assert raised.value.status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


@pytest.mark.parametrize(
    "data",
    [
        b"definitely not an image",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 40,
        encoded_image("PNG")[:60],
        encoded_image("JPEG")[:200],
    ],
)
def test_decode_image_rejects_corrupt_bytes(data: bytes):
    wanted_format = "JPEG" if data.startswith(b"\xff\xd8") else "PNG"

    with pytest.raises(ProblemError) as raised:
        decode_image(data, wanted_format, MAX_PIXELS)

    assert raised.value.status == HTTPStatus.BAD_REQUEST
    assert raised.value.detail == "The file could not be read as an image."


def test_decode_image_rejects_a_png_with_a_broken_checksum():
    data = bytearray(encoded_image("PNG"))
    data[-20] ^= 0xFF

    with pytest.raises(ProblemError) as raised:
        decode_image(bytes(data), "PNG", MAX_PIXELS)

    assert raised.value.status == HTTPStatus.BAD_REQUEST


def test_decode_image_rejects_huge_dimensions_before_decoding():
    bomb = png_header_only(20_000, 20_000)

    with pytest.raises(ProblemError) as raised:
        decode_image(bomb, "PNG", MAX_PIXELS)

    assert len(bomb) < 100
    assert raised.value.status == HTTPStatus.BAD_REQUEST


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_decode_image_rejects_dimensions_just_over_the_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 5000)

    with pytest.raises(ProblemError) as raised:
        decode_image(encoded_image("PNG", (100, 51)), "PNG", max_pixels=5000)

    assert raised.value.status == HTTPStatus.BAD_REQUEST
    assert "5000 pixels" in raised.value.detail


@pytest.mark.filterwarnings("error::PIL.Image.DecompressionBombWarning")
def test_decode_image_rejects_the_pillow_warning_when_it_is_raised(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 5000)

    with pytest.raises(ProblemError) as raised:
        decode_image(encoded_image("PNG", (100, 51)), "PNG", max_pixels=5000)

    assert raised.value.status == HTTPStatus.BAD_REQUEST
    assert raised.value.detail == "The file could not be read as an image."


def test_decode_image_applies_its_own_limit_when_pillow_allows_more():
    with pytest.raises(ProblemError) as raised:
        decode_image(encoded_image("PNG", (100, 51)), "PNG", max_pixels=5000)

    assert raised.value.status == HTTPStatus.BAD_REQUEST
    assert "5000 pixels" in raised.value.detail


def test_decode_image_error_never_contains_the_file_content():
    secret = b"SECRET-CONTENT-12345"

    with pytest.raises(ProblemError) as raised:
        decode_image(secret, "PNG", MAX_PIXELS)

    assert "SECRET" not in raised.value.detail
    assert "SECRET" not in str(raised.value)
