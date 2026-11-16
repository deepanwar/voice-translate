from enum import Enum

from pydantic import BaseModel, Field


class Tone(str, Enum):
    natural = "natural"
    professional = "professional"
    friendly = "friendly"
    polite = "polite"
    direct = "direct"


class RewriteRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=4000)
    tone: Tone = Tone.natural


class RewriteResponse(BaseModel):
    english: str
    tone: Tone
    processing_ms: int


class ProcessResponse(RewriteResponse):
    request_id: str
    transcript: str
