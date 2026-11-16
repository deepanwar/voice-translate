from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.schemas.api import ProcessResponse, RewriteRequest, RewriteResponse, Tone
from app.services.speech_service import AppError

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready", dependencies=[Depends(require_api_key)])
async def ready(request: Request):
    return {"asr": True, "llm": await request.app.state.pipeline.rewriter.ready()}


@router.post("/v1/rewrite", response_model=RewriteResponse,
             dependencies=[Depends(require_api_key)])
async def rewrite(body: RewriteRequest, request: Request):
    return await request.app.state.pipeline.rewrite(body.transcript, body.tone)


async def read_capped(request: Request, cap: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cap:
        raise AppError(413, "BODY_TOO_LARGE")
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > cap:
            raise AppError(413, "BODY_TOO_LARGE")
    return bytes(buf)


@router.post("/v1/process", response_model=ProcessResponse,
             dependencies=[Depends(require_api_key)])
async def process(request: Request, tone: Tone = Tone.natural,
                  settings: Settings = Depends(get_settings)):
    if request.headers.get("content-type", "").split(";")[0].strip() not in (
        "audio/mp4", "audio/m4a", "audio/x-m4a",
    ):
        raise AppError(415, "UNSUPPORTED_AUDIO")
    audio = await read_capped(request, settings.max_body_bytes)
    if not audio:
        raise AppError(400, "EMPTY_BODY")
    return await request.app.state.pipeline.process(audio, tone)
