from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from wastevision.inference.predictor import Prediction
from wastevision.labels import sorting_hint_for
from wastevision.web.api import decode_upload, enforce_rate_limit, loaded_predictor
from wastevision.web.problems import ProblemError

WEB_DIRECTORY = Path(__file__).resolve().parent
TEMPLATES_DIRECTORY = WEB_DIRECTORY / "templates"
STATIC_DIRECTORY = WEB_DIRECTORY / "static"
BYTES_PER_MEGABYTE = 1024 * 1024

templates = Jinja2Templates(directory=TEMPLATES_DIRECTORY)
router = APIRouter()


@dataclass(frozen=True)
class RankedClass:
    label: str
    percent: int
    probability: float


@dataclass(frozen=True)
class PageResult:
    category: str
    label: str
    percent: int
    probability: float
    unsure: bool
    hint: str
    top: tuple[RankedClass, ...]


def to_percent(probability: float) -> int:
    return round(probability * 100)


def build_result(prediction: Prediction) -> PageResult:
    return PageResult(
        category=prediction.category,
        label=prediction.label,
        percent=to_percent(prediction.confidence),
        probability=prediction.confidence,
        unsure=prediction.unsure,
        hint=sorting_hint_for(prediction.label),
        top=tuple(
            RankedClass(label, to_percent(probability), probability)
            for label, probability in prediction.top
        ),
    )


def render_page(
    request: Request,
    status: int = HTTPStatus.OK,
    headers: dict[str, str] | None = None,
    **context: Any,
) -> HTMLResponse:
    max_megabytes = request.app.state.settings.max_upload_bytes / BYTES_PER_MEGABYTE
    context.setdefault("result", None)
    context.setdefault("error", None)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"max_megabytes": f"{max_megabytes:g}", **context},
        status_code=status,
        headers=headers,
    )


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return render_page(request)


@router.post("/", response_class=HTMLResponse)
def classify_page(
    request: Request, image: Annotated[UploadFile | None, File()] = None
) -> HTMLResponse:
    try:
        enforce_rate_limit(request)
        predictor = loaded_predictor(request)
        if image is None or not image.filename:
            raise ProblemError(HTTPStatus.BAD_REQUEST, "Choose an image before you press Classify.")
        picture = decode_upload(request, image)
    except ProblemError as error:
        return render_page(request, error.status, error.headers, error=error.detail)
    return render_page(request, result=build_result(predictor.predict(picture)))
