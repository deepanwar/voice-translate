import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.schemas.api import Tone
from app.services.pipeline_service import PipelineService
from app.services.rewrite_service import RewriteService
from app.services.speech_service import AppError

KEY = "test-key"
H = {"X-API-Key": KEY}


class FakeSpeech:
    def __init__(self):
        self.calls = 0
        self.fail: AppError | None = None

    def transcribe(self, audio: bytes) -> str:
        self.calls += 1
        if self.fail:
            raise self.fail
        return "Mujhe kal tak update de dena"


def make_rewriter(handler=None):
    def default(request: httpx.Request):
        return httpx.Response(200, json={"message": {"content": "<think>x</think>Please update me by tomorrow."}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler or default))
    return RewriteService(Settings(voice_api_key=KEY), client)


@pytest.fixture
def ctx():
    speech = FakeSpeech()
    app.state.pipeline = PipelineService(speech, make_rewriter())
    app.dependency_overrides[get_settings] = lambda: Settings(voice_api_key=KEY, max_body_bytes=1000)
    with TestClient(app) as c:
        yield c, speech
    app.dependency_overrides.clear()
    del app.state.pipeline


def test_health_open(ctx):
    assert ctx[0].get("/health").json() == {"status": "ok"}


def test_rewrite_requires_key_and_skips_inference(ctx):
    c, speech = ctx
    body = {"transcript": "hi", "tone": "polite"}
    assert c.post("/v1/rewrite", json=body).status_code == 401
    assert c.post("/v1/rewrite", json=body, headers={"X-API-Key": "bad"}).status_code == 401


def test_rewrite_ok_strips_think_and_skips_asr(ctx):
    c, speech = ctx
    r = c.post("/v1/rewrite", json={"transcript": "Mujhe kal tak update de dena", "tone": "professional"}, headers=H)
    assert r.status_code == 200
    assert r.json()["english"] == "Please update me by tomorrow."
    assert speech.calls == 0


def test_process_ok(ctx):
    c, speech = ctx
    r = c.post("/v1/process?tone=natural", content=b"abc", headers={**H, "Content-Type": "audio/mp4"})
    assert r.status_code == 200
    j = r.json()
    assert j["transcript"] and j["english"] and j["request_id"]
    assert speech.calls == 1


def test_process_oversize_413(ctx):
    c, _ = ctx
    r = c.post("/v1/process", content=b"x" * 2000, headers={**H, "Content-Type": "audio/mp4"})
    assert r.status_code == 413


def test_process_wrong_content_type(ctx):
    c, _ = ctx
    r = c.post("/v1/process", content=b"x", headers={**H, "Content-Type": "text/plain"})
    assert r.status_code == 415


def test_process_no_speech(ctx):
    c, speech = ctx
    speech.fail = AppError(422, "ASR_NO_SPEECH")
    r = c.post("/v1/process", content=b"x", headers={**H, "Content-Type": "audio/mp4"})
    assert r.status_code == 422 and r.json()["error"] == "ASR_NO_SPEECH"


def test_ollama_down_maps_502():
    def boom(request):
        raise httpx.ConnectError("down")

    svc = make_rewriter(boom)
    with pytest.raises(AppError) as e:
        asyncio.run(svc.rewrite("hi", Tone.natural))
    assert e.value.status == 502


def test_busy_429():
    async def go():
        gate = asyncio.Event()

        async def slow(request):
            await gate.wait()
            return httpx.Response(200, json={"message": {"content": "ok"}})

        p = PipelineService(FakeSpeech(), make_rewriter(slow))
        first = asyncio.create_task(p.rewrite("a", Tone.natural))
        await asyncio.sleep(0.05)
        with pytest.raises(AppError) as e:
            await p.rewrite("b", Tone.natural)
        gate.set()
        await first
        return e.value.status

    assert asyncio.run(go()) == 429
