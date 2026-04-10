from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
import uvicorn

from avito.constants import (
    SHOULD_SPLIT_API_DEFAULT_HOST,
    SHOULD_SPLIT_API_DEFAULT_PORT,
    SHOULD_SPLIT_API_HEALTH_PATH,
    SHOULD_SPLIT_API_INFER_PATH,
    SHOULD_SPLIT_API_OK_STATUS,
)
from avito.should_split.api.schemas import AvitoApiRequest, AvitoApiResponse
from avito.should_split.api.service import build_api_response
from avito.should_split.core.pipeline import ShouldSplitPipeline, build_default_pipeline
from common.logger import AVITO_SHOULD_SPLIT_LOGGER as logger
from common.logger import MISTRAL_LOGGER


def _resolve_pipeline(pipeline: ShouldSplitPipeline | None) -> ShouldSplitPipeline:
    if pipeline is not None:
        return pipeline
    return build_default_pipeline()


def create_app(pipeline: ShouldSplitPipeline | None = None) -> FastAPI:
    """Создает FastAPI приложение для shouldSplit API."""
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        original_mistral_disabled = MISTRAL_LOGGER.disabled
        original_mistral_level = MISTRAL_LOGGER.level
        MISTRAL_LOGGER.disabled = True
        MISTRAL_LOGGER.setLevel(logging.CRITICAL)
        app.state.pipeline = _resolve_pipeline(pipeline)
        logger.info("[api] Mistral logger muted for server runtime")
        logger.info("[api] FastAPI server is ready")
        try:
            yield
        finally:
            MISTRAL_LOGGER.disabled = original_mistral_disabled
            MISTRAL_LOGGER.setLevel(original_mistral_level)

    app = FastAPI(title="Avito ShouldSplit API", version="1.0.0", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get(SHOULD_SPLIT_API_HEALTH_PATH)
    def health() -> dict[str, str]:
        return {"status": SHOULD_SPLIT_API_OK_STATUS}

    @app.post(SHOULD_SPLIT_API_INFER_PATH, response_model=AvitoApiResponse)
    def infer(payload: AvitoApiRequest) -> AvitoApiResponse:
        try:
            pipeline_instance = getattr(app.state, "pipeline", None)
            if pipeline_instance is None:
                pipeline_instance = _resolve_pipeline(pipeline)
                app.state.pipeline = pipeline_instance
            return build_api_response(payload=payload, pipeline=pipeline_instance)
        except Exception as error:
            logger.exception("[api] inference failed")
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def run_server(host: str, port: int) -> None:
    uvicorn.run(
        "avito.should_split.api.server:create_app",
        host=host,
        port=port,
        factory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shouldSplit FastAPI server")
    parser.add_argument("--host", type=str, default=SHOULD_SPLIT_API_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=SHOULD_SPLIT_API_DEFAULT_PORT)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()



