from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.runtime import AppState, QueueFullError
from api.schemas import (
    CloneRequest,
    CustomVoiceRequest,
    TrainVoiceRequest,
    TranscribeRequest,
    VoiceDesignRequest,
)
from api.service import ApiService


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS API", version="0.1.0")
    api_prefix = "/api"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = ApiService(state)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, exc: RequestValidationError):
        message = "; ".join(item["msg"] for item in exc.errors()) or "Invalid request body"
        return JSONResponse({"ok": False, "error": message}, status_code=422)

    @app.exception_handler(KeyError)
    async def key_error(_: Request, exc: KeyError):
        return JSONResponse({"ok": False, "error": f"Missing field: {exc.args[0]}"}, status_code=400)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.exception_handler(QueueFullError)
    async def queue_full_error(_: Request, exc: QueueFullError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)

    @app.exception_handler(FileNotFoundError)
    async def file_not_found(_: Request, exc: FileNotFoundError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

    @app.exception_handler(PermissionError)
    async def permission_error(_: Request, exc: PermissionError):
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=403)

    @app.exception_handler(Exception)
    async def generic_error(_: Request, exc: Exception):
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.get(f"{api_prefix}/healthz")
    def healthz():
        return service.healthz()

    @app.get(f"{api_prefix}/voices")
    def voices():
        return service.get_voices()

    @app.post(f"{api_prefix}/transcribe")
    async def transcribe(request: TranscribeRequest = Depends(TranscribeRequest.as_form)):
        return service.transcribe(await request.to_payload())

    @app.post(f"{api_prefix}/voiceDesign")
    def voice_design(request: VoiceDesignRequest):
        result = service.voice_design(request.model_dump(exclude_none=True))
        return service.build_audio_response(result)

    @app.post(f"{api_prefix}/clone")
    async def clone(request: CloneRequest = Depends(CloneRequest.as_form)):
        result = service.clone(await request.to_payload())
        return service.build_audio_response(result)

    @app.post(f"{api_prefix}/customVoice")
    def custom_voice(request: CustomVoiceRequest):
        result = service.custom_voice(request.model_dump(exclude_none=True))
        return service.build_audio_response(result)

    @app.post(f"{api_prefix}/trainVoice")
    async def train_voice(http_request: Request, request: TrainVoiceRequest = Depends(TrainVoiceRequest.as_form)):
        result = service.train_voice(await request.to_payload())
        return service.stream_train_status(result["taskId"], http_request)

    @app.delete(f"{api_prefix}/voices/{{voice_id}}")
    def delete_voice(voice_id: str):
        return service.delete_voice(voice_id)

    return app
