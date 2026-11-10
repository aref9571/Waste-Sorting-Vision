import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from PIL import Image

from wastevision import __version__
from wastevision.config import Settings
from wastevision.inference.predictor import Predictor
from wastevision.web.api import router
from wastevision.web.problems import add_problem_handlers
from wastevision.web.rate_limit import RateLimiter

PredictorLoader = Callable[[Path], Predictor]

logger = logging.getLogger(__name__)


def load_predictor_or_none(load_predictor: PredictorLoader, models_dir: Path) -> Predictor | None:
    try:
        return load_predictor(models_dir)
    except Exception:
        logger.exception("The model could not be loaded from %s", models_dir)
        return None


def create_rate_limiter(settings: Settings) -> RateLimiter | None:
    if settings.rate_limit_per_minute == 0:
        return None
    return RateLimiter(settings.rate_limit_per_minute)


def create_app(
    settings: Settings | None = None,
    load_predictor: PredictorLoader = Predictor.from_directory,
) -> FastAPI:
    settings = settings or Settings()
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = load_predictor_or_none(load_predictor, settings.models_dir)
        yield
        app.state.predictor = None

    app = FastAPI(title="Waste Sorting Vision", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.predictor = None
    app.state.rate_limiter = create_rate_limiter(settings)
    add_problem_handlers(app)
    app.include_router(router)
    return app
