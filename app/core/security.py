import secrets

from fastapi import Depends, Header, HTTPException

from app.core.config import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    # Fail closed: no configured key means nothing authenticates.
    if not settings.voice_api_key:
        raise HTTPException(503, detail={"code": "SERVER_NOT_CONFIGURED"})
    if not x_api_key or not secrets.compare_digest(
        x_api_key.encode(), settings.voice_api_key.encode()
    ):
        raise HTTPException(401, detail={"code": "INVALID_API_KEY"})
