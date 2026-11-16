import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.services.pipeline_service import PipelineService
from app.services.rewrite_service import RewriteService
from app.services.speech_service import AppError, SpeechService

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if not hasattr(app.state, "pipeline"):  # tests inject their own
        app.state.pipeline = PipelineService(SpeechService(s), RewriteService(s))
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.exception_handler(AppError)
async def app_error(_: Request, exc: AppError):
    headers = {"Retry-After": "5"} if exc.status == 429 else None
    return JSONResponse({"error": exc.code}, status_code=exc.status, headers=headers)
