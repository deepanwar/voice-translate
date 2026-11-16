from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    voice_api_key: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_timeout_s: float = 60.0

    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_compute: str = "int8"
    asr_threads: int = 8
    asr_language: str | None = "hi"
    asr_beam_size: int = 3

    max_body_bytes: int = 10 * 1024 * 1024
    max_audio_seconds: float = 30.0
    max_transcript_chars: int = 4000


@lru_cache
def get_settings() -> Settings:
    return Settings()
