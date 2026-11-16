import asyncio
import logging
import time
import uuid

from fastapi.concurrency import run_in_threadpool

from app.schemas.api import ProcessResponse, RewriteResponse, Tone
from app.services.rewrite_service import RewriteService
from app.services.speech_service import AppError, SpeechService

log = logging.getLogger("pipeline")


class PipelineService:
    def __init__(self, speech: SpeechService, rewriter: RewriteService):
        self.speech = speech
        self.rewriter = rewriter
        self._lock = asyncio.Lock()  # one inference session at a time

    def _acquire_or_busy(self):
        if self._lock.locked():
            raise AppError(429, "SERVER_BUSY")

    async def process(self, audio: bytes, tone: Tone) -> ProcessResponse:
        self._acquire_or_busy()
        async with self._lock:
            rid, t0 = str(uuid.uuid4()), time.perf_counter()
            transcript = await run_in_threadpool(self.speech.transcribe, audio)
            t_asr = time.perf_counter()
            english = await self.rewriter.rewrite(transcript, tone)
            t_end = time.perf_counter()
            log.info("process id=%s asr_ms=%d llm_ms=%d", rid,
                     (t_asr - t0) * 1000, (t_end - t_asr) * 1000)
            return ProcessResponse(
                request_id=rid, transcript=transcript, english=english,
                tone=tone, processing_ms=int((t_end - t0) * 1000),
            )

    async def rewrite(self, transcript: str, tone: Tone) -> RewriteResponse:
        self._acquire_or_busy()
        async with self._lock:
            t0 = time.perf_counter()
            english = await self.rewriter.rewrite(transcript, tone)
            return RewriteResponse(
                english=english, tone=tone,
                processing_ms=int((time.perf_counter() - t0) * 1000),
            )
