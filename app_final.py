import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response

from core.config import APP_NAME, Settings, configure_logging
from core.detector import Detector, load_or_train
from core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    SentencesResponse,
    SentenceVerdict,
    VersionResponse,
    Verdict,
)
from core.text import split_sentences

logger = logging.getLogger("extremism_detector")
router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_detector(request: Request) -> Detector:
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return detector


def get_threshold(
    settings: Settings = Depends(get_settings),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0, description="Decision threshold, overrides THRESHOLD"),
) -> float:
    return settings.threshold if threshold is None else threshold


def to_verdict(probability: float, threshold: float) -> Verdict:
    return Verdict(probability=round(probability, 4), is_extremist=probability >= threshold)


@router.get("/", tags=["service"])
def root(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"name": APP_NAME, "version": settings.app_version, "docs": "/docs", "health": "/health"}


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request, response: Response) -> HealthResponse:
    is_loaded = request.app.state.detector is not None
    if not is_loaded:
        response.status_code = 503
    return HealthResponse(status="ok" if is_loaded else "unavailable", model_loaded=is_loaded)


@router.get("/version", response_model=VersionResponse, tags=["service"])
def version(settings: Settings = Depends(get_settings)) -> VersionResponse:
    return VersionResponse(name=APP_NAME, version=settings.app_version)


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(
    payload: AnalyzeRequest,
    detector: Detector = Depends(get_detector),
    threshold: float = Depends(get_threshold),
) -> AnalyzeResponse:
    [probability] = detector.predict([payload.text])
    verdict = to_verdict(probability, threshold)
    logger.info("Analyzed text of %d chars: probability=%.4f extremist=%s", len(payload.text), verdict.probability, verdict.is_extremist)
    return AnalyzeResponse(**verdict.model_dump(), threshold=threshold)


@router.post("/analyze/batch", response_model=BatchResponse, tags=["analysis"])
def analyze_batch(
    payload: BatchRequest,
    detector: Detector = Depends(get_detector),
    threshold: float = Depends(get_threshold),
) -> BatchResponse:
    results = [to_verdict(probability, threshold) for probability in detector.predict(payload.texts)]
    extremist_count = sum(result.is_extremist for result in results)
    logger.info("Analyzed batch of %d texts: %d extremist", len(results), extremist_count)
    return BatchResponse(threshold=threshold, total=len(results), extremist_count=extremist_count, results=results)


@router.post("/analyze/sentences", response_model=SentencesResponse, tags=["analysis"])
def analyze_sentences(
    payload: AnalyzeRequest,
    detector: Detector = Depends(get_detector),
    threshold: float = Depends(get_threshold),
) -> SentencesResponse:
    sentences = split_sentences(payload.text)
    overall_probability, *probabilities = detector.predict([payload.text, *sentences])
    overall = to_verdict(overall_probability, threshold)
    verdicts = [
        SentenceVerdict(sentence=sentence, **to_verdict(probability, threshold).model_dump())
        for sentence, probability in zip(sentences, probabilities)
    ]
    extremist_count = sum(verdict.is_extremist for verdict in verdicts)
    logger.info("Analyzed %d sentences: %d extremist", len(verdicts), extremist_count)
    return SentencesResponse(
        **overall.model_dump(),
        threshold=threshold,
        extremist_count=extremist_count,
        sentences=verdicts,
    )


@router.get("/model", response_model=ModelInfoResponse, tags=["model"])
def model_info(
    detector: Detector = Depends(get_detector),
    settings: Settings = Depends(get_settings),
) -> ModelInfoResponse:
    return ModelInfoResponse(**asdict(detector.info), threshold=settings.threshold)


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting %s v%s", APP_NAME, settings.app_version)
        try:
            app.state.detector = load_or_train(settings.model_dir, settings.training_data_path)
        except Exception:
            logger.exception("Failed to prepare the model from %s", settings.model_dir)
            raise
        logger.info("Service is ready, threshold=%.2f", settings.threshold)
        yield
        logger.info("Shutting down")

    app = FastAPI(
        title=APP_NAME,
        version=settings.app_version,
        description="Detects extremist content in Russian texts with a small Keras model.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.detector = None
    app.include_router(router)
    return app


app_settings = Settings()
configure_logging(app_settings.log_level)
app = create_app(app_settings)


if __name__ == "__main__":
    uvicorn.run(app, host=app_settings.host, port=app_settings.port)
