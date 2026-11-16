import io
import logging

from faster_whisper import WhisperModel

from app.core.config import Settings


log = logging.getLogger("asr")


class AppError(Exception):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code


class SpeechService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.model = WhisperModel(
            settings.asr_model,
            device=settings.asr_device,
            compute_type=settings.asr_compute,
            cpu_threads=settings.asr_threads,
        )

    def transcribe(self, audio: bytes) -> str:
        """Blocking; call from a worker thread. Decodes from memory, no disk."""
        try:
            segments, info = self.model.transcribe(
                io.BytesIO(audio),
                language=self.s.asr_language,
                task="transcribe",
                beam_size=self.s.asr_beam_size,
                vad_filter=True,
            )
        except Exception as e:
            log.error("transcribe failed: %s", type(e).__name__)
            raise AppError(415, "UNSUPPORTED_AUDIO")
        if info.duration > self.s.max_audio_seconds:
            raise AppError(413, "AUDIO_TOO_LONG")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            raise AppError(422, "ASR_NO_SPEECH")
        return text[: self.s.max_transcript_chars]
