from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000


class Augmentation(StrEnum):
    STANDARD = "standard"
    STRONG = "strong"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WASTEVISION_")

    num_workers: int = Field(default=0, ge=0)
    models_dir: Path = Path("models")
    max_upload_bytes: int = Field(default=DEFAULT_MAX_UPLOAD_BYTES, gt=0)
    max_image_pixels: int = Field(default=DEFAULT_MAX_IMAGE_PIXELS, gt=0)
    rate_limit_per_minute: int = Field(default=0, ge=0)
