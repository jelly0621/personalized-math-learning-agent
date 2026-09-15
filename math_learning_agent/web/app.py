"""FastAPI application exposing the existing MathLearningAgent workflows."""

import base64
import binascii
import os
from pathlib import Path
import secrets
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from math_learning_agent.web.schemas import (
    EvaluationResponse,
    GenerateQuestionRequest,
    GeneratedQuestionResponse,
    ProfileItem,
    PhotoImportRequest,
    PhotoImportResponse,
    TrainingFeedbackRequest,
    TrainingFeedbackResponse,
    TrainingPlanRequest,
    TrainingPlanResponse,
    WrongProblemRequest,
    WrongProblemResponse,
)
from math_learning_agent.models import PhotoExtractionResult
from math_learning_agent.web.services import WebServices


STATIC_DIRECTORY = Path(__file__).parent / "static"
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_PHOTO_COUNT = 5
ALLOWED_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_PHOTO_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _access_credentials() -> tuple[str, str] | None:
    username = os.getenv("APP_ACCESS_USERNAME", "")
    password = os.getenv("APP_ACCESS_PASSWORD", "")
    if bool(username) != bool(password):
        raise RuntimeError(
            "APP_ACCESS_USERNAME and APP_ACCESS_PASSWORD must either both be "
            "set or both be unset."
        )
    return (username, password) if username else None


def _has_valid_basic_auth(
    authorization: str | None,
    credentials: tuple[str, str],
) -> bool:
    if not authorization:
        return False
    scheme, separator, encoded = authorization.partition(" ")
    if not separator or scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    expected_username, expected_password = credentials
    username_matches = secrets.compare_digest(username, expected_username)
    password_matches = secrets.compare_digest(password, expected_password)
    return username_matches & password_matches


def _detected_image_mime(content: bytes) -> str | None:
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def create_app(services: Any | None = None) -> FastAPI:
    access_credentials = _access_credentials()
    application = FastAPI(title="MathLearningAgent", version="0.8.0")
    application.state.services = services or WebServices()
    application.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @application.middleware("http")
    async def require_basic_auth(request: Request, call_next):
        if (
            access_credentials is not None
            and request.url.path != "/api/health"
            and not _has_valid_basic_auth(
                request.headers.get("Authorization"), access_credentials
            )
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
                headers={"WWW-Authenticate": 'Basic realm="MathLearningAgent"'},
            )
        return await call_next(request)

    @application.exception_handler(LookupError)
    async def handle_not_found(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(RuntimeError)
    async def handle_runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
        message = str(exc)
        detail = (
            "LLM configuration is unavailable."
            if "LLM configuration" in message
            else "The request could not be completed."
        )
        return JSONResponse(status_code=503, content={"detail": detail})

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "The request could not be completed safely."},
        )

    def service(request: Request) -> Any:
        return request.app.state.services

    @application.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @application.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "app": "MathLearningAgent"}

    @application.post("/api/wrong-problems", response_model=WrongProblemResponse)
    async def create_wrong_problem(
        payload: WrongProblemRequest, request: Request
    ) -> object:
        return service(request).create_wrong_problem(payload)

    @application.post("/api/photo/extract", response_model=PhotoExtractionResult)
    async def extract_photo_problems(
        request: Request,
        images: list[UploadFile] = File(...),
    ) -> object:
        if not 1 <= len(images) <= MAX_PHOTO_COUNT:
            raise HTTPException(
                status_code=400,
                detail="Upload between 1 and 5 images.",
            )

        validated_images: list[tuple[bytes, str]] = []
        for image in images:
            extension = Path(image.filename or "").suffix.lower()
            if extension not in ALLOWED_PHOTO_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail="Only JPG, JPEG, PNG, and WebP images are supported.",
                )
            if image.content_type not in ALLOWED_PHOTO_MIME_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail="The uploaded file has an unsupported image MIME type.",
                )
            content = await image.read(MAX_PHOTO_BYTES + 1)
            if len(content) > MAX_PHOTO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Each image must be 10 MB or smaller.",
                )
            detected_mime = _detected_image_mime(content)
            if detected_mime is None or detected_mime != image.content_type:
                raise HTTPException(
                    status_code=415,
                    detail="The file content does not match a supported image type.",
                )
            validated_images.append((content, detected_mime))

        return service(request).extract_photo_problems(validated_images)

    @application.post("/api/photo/import", response_model=PhotoImportResponse)
    async def import_photo_problems(
        payload: PhotoImportRequest,
        request: Request,
    ) -> object:
        return service(request).import_photo_problems(payload)

    @application.get("/api/profile", response_model=list[ProfileItem])
    async def profile(request: Request) -> object:
        return service(request).get_profile()

    @application.post("/api/training-plan", response_model=TrainingPlanResponse)
    async def training_plan(
        payload: TrainingPlanRequest, request: Request
    ) -> object:
        return service(request).create_training_plan(payload)

    @application.post(
        "/api/questions/generate", response_model=GeneratedQuestionResponse
    )
    async def generate_question(
        payload: GenerateQuestionRequest, request: Request
    ) -> object:
        raw_result = service(request).generate_question(payload)
        if hasattr(raw_result, "model_dump"):
            raw_result = raw_result.model_dump()
        allowed_fields = {
            "question_id",
            "question_text",
            "knowledge_points",
            "difficulty",
            "generation_attempts",
            "generation_status",
        }
        return {
            key: value
            for key, value in raw_result.items()
            if key in allowed_fields
        }

    @application.post(
        "/api/training-feedback", response_model=TrainingFeedbackResponse
    )
    async def training_feedback(
        payload: TrainingFeedbackRequest, request: Request
    ) -> object:
        return service(request).submit_training_feedback(payload)

    @application.get("/api/history/wrong-problems")
    async def wrong_problem_history(
        request: Request, limit: int = Query(default=10, ge=1, le=100)
    ) -> object:
        return service(request).get_wrong_problem_history(limit)

    @application.get("/api/history/generated-questions")
    async def generated_question_history(
        request: Request, limit: int = Query(default=10, ge=1, le=100)
    ) -> object:
        return service(request).get_generated_question_history(limit)

    @application.get("/api/history/training-attempts")
    async def training_attempt_history(
        request: Request, limit: int = Query(default=10, ge=1, le=100)
    ) -> object:
        return service(request).get_training_attempt_history(limit)

    @application.get("/api/evaluation/latest", response_model=EvaluationResponse)
    async def latest_evaluation(request: Request) -> object:
        return service(request).get_latest_evaluation()

    return application


app = create_app()
