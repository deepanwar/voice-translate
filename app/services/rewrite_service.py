import re

import httpx

from app.core.config import Settings
from app.prompts.tone_prompts import build_messages
from app.schemas.api import Tone
from app.services.speech_service import AppError


class RewriteService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.s = settings
        self.client = client or httpx.AsyncClient(timeout=settings.ollama_timeout_s)

    async def rewrite(self, transcript: str, tone: Tone) -> str:
        try:
            r = await self.client.post(
                f"{self.s.ollama_url}/api/chat",
                json={
                    "model": self.s.ollama_model,
                    "stream": False,
                    "messages": build_messages(transcript, tone),
                    "options": {"temperature": 0.2},
                },
            )
            r.raise_for_status()
            text = r.json()["message"]["content"]
        except httpx.TimeoutException:
            raise AppError(504, "REQUEST_TIMEOUT")
        except (httpx.HTTPError, KeyError, ValueError):
            raise AppError(502, "MODEL_ERROR")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if not text:
            raise AppError(502, "MODEL_ERROR")
        return text

    async def ready(self) -> bool:
        try:
            r = await self.client.get(f"{self.s.ollama_url}/api/tags", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
