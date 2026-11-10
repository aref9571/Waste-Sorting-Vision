import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

PROBLEM_MEDIA_TYPE = "application/problem+json"

logger = logging.getLogger(__name__)


class ProblemError(Exception):
    def __init__(self, status: int, detail: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.headers = headers


def problem_response(
    request: Request,
    status: int,
    detail: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": HTTPStatus(status).phrase,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM_MEDIA_TYPE, headers=headers)


async def handle_problem(request: Request, error: ProblemError) -> JSONResponse:
    return problem_response(request, error.status, error.detail, error.headers)


async def handle_http_exception(request: Request, error: HTTPException) -> JSONResponse:
    return problem_response(request, error.status_code, str(error.detail), error.headers)


async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    fields = sorted({".".join(str(part) for part in item["loc"]) for item in error.errors()})
    detail = f"Missing or invalid request fields: {', '.join(fields)}."
    return problem_response(request, HTTPStatus.UNPROCESSABLE_ENTITY, detail)


async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.exception("Unexpected error while handling %s", request.url.path)
    return problem_response(
        request, HTTPStatus.INTERNAL_SERVER_ERROR, "An internal error occurred."
    )


def add_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProblemError, handle_problem)
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
