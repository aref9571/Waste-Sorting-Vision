from http import HTTPStatus
from io import BytesIO

from PIL import Image

from wastevision.web.problems import ProblemError

FORMAT_BY_CONTENT_TYPE: dict[str, str] = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
DECODE_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    SyntaxError,
    ValueError,
    EOFError,
    Image.DecompressionBombError,
    Image.DecompressionBombWarning,
)


def expected_format(content_type: str | None) -> str:
    media_type = (content_type or "").split(";")[0].strip().lower()
    if media_type not in FORMAT_BY_CONTENT_TYPE:
        raise ProblemError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Upload a JPEG, PNG or WebP image.")
    return FORMAT_BY_CONTENT_TYPE[media_type]


def check_upload_size(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise ProblemError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"The image must not be larger than {max_bytes} bytes.",
        )
    if not data:
        raise ProblemError(HTTPStatus.BAD_REQUEST, "The uploaded file is empty.")


def check_format(actual_format: str | None, wanted_format: str) -> None:
    if actual_format != wanted_format:
        raise ProblemError(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            "The file content does not match its content type.",
        )


def check_pixel_count(size: tuple[int, int], max_pixels: int) -> None:
    width, height = size
    if width * height > max_pixels:
        raise ProblemError(
            HTTPStatus.BAD_REQUEST,
            f"The image must not have more than {max_pixels} pixels.",
        )


def decode_image(data: bytes, wanted_format: str, max_pixels: int) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as header:
            check_format(header.format, wanted_format)
            check_pixel_count(header.size, max_pixels)
            header.verify()
        image = Image.open(BytesIO(data))
        image.load()
    except DECODE_ERRORS:
        raise ProblemError(
            HTTPStatus.BAD_REQUEST, "The file could not be read as an image."
        ) from None
    return image


def validate_upload(
    data: bytes,
    content_type: str | None,
    max_bytes: int,
    max_pixels: int,
) -> Image.Image:
    wanted_format = expected_format(content_type)
    check_upload_size(data, max_bytes)
    return decode_image(data, wanted_format, max_pixels)
