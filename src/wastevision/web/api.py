from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

from wastevision.inference.predictor import Prediction, Predictor
from wastevision.web.problems import ProblemError
from wastevision.web.validation import validate_upload

router = APIRouter()


class TopPrediction(BaseModel):
    label: str
    probability: float


class ClassificationResponse(BaseModel):
    category: str
    label: str
    confidence: float
    unsure: bool
    top3: list[TopPrediction]

    @classmethod
    def from_prediction(cls, prediction: Prediction) -> "ClassificationResponse":
        return cls(
            category=prediction.category,
            label=prediction.label,
            confidence=prediction.confidence,
            unsure=prediction.unsure,
            top3=[
                TopPrediction(label=label, probability=probability)
                for label, probability in prediction.top
            ],
        )


class HealthResponse(BaseModel):
    status: str


def enforce_rate_limit(request: Request) -> None:
    limiter = request.app.state.rate_limiter
    if limiter is None:
        return
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise ProblemError(
            HTTPStatus.TOO_MANY_REQUESTS,
            "Too many requests. Try again later.",
            headers={"Retry-After": str(limiter.seconds_until_reset())},
        )


def loaded_predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise ProblemError(HTTPStatus.SERVICE_UNAVAILABLE, "The model is not loaded.")
    return predictor


def decode_upload(request: Request, upload: UploadFile) -> Image.Image:
    settings = request.app.state.settings
    data = upload.file.read(settings.max_upload_bytes + 1)
    return validate_upload(
        data, upload.content_type, settings.max_upload_bytes, settings.max_image_pixels
    )


@router.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
def classify(
    request: Request,
    image: Annotated[UploadFile, File()],
    predictor: Annotated[Predictor, Depends(loaded_predictor)],
) -> ClassificationResponse:
    picture = decode_upload(request, image)
    return ClassificationResponse.from_prediction(predictor.predict(picture))


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="live")


@router.get("/health/ready", response_model=HealthResponse)
def ready(request: Request) -> HealthResponse:
    loaded_predictor(request)
    return HealthResponse(status="ready")
