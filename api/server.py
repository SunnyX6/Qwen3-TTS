from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from qwen_tts.inference.voice_registry import VoiceRegistry

from api.exceptions import register_exception_handlers
from api.schemas import (
    CloneRequest,
    CustomVoiceRequest,
    TrainVoiceRequest,
    TranscribeRequest,
    VoiceDesignRequest,
)
from runtime.catalog import (
    BASE_MODEL_IDS,
    CUSTOM_VOICE_MODEL_IDS,
    VOICE_DESIGN_MODEL_IDS,
    require_model_ref,
    require_supported_model_id,
)
from runtime.executor import Executor
from runtime.task import (
    PreparedAudioResponse,
    get_builtin_speakers_for_base_model,
    list_available_voices,
    resolve_public_base_model_id,
    resolve_train_model_pair,
)


def _deserialize_audio_response(payload: Dict[str, Any]) -> PreparedAudioResponse:
    json_payload = payload.get("json_payload")
    return PreparedAudioResponse(
        response_format=str(payload["response_format"]),
        audio_bytes=payload.get("audio_bytes"),
        json_payload=dict(json_payload) if isinstance(json_payload, dict) else None,
    )


def _build_audio_response(result: PreparedAudioResponse) -> Response:
    if result.response_format == "wav":
        body = result.audio_bytes or b""
        result.audio_bytes = None
        result.json_payload = None
        return Response(content=body, media_type="audio/wav")
    payload = dict(result.json_payload or {})
    result.audio_bytes = None
    result.json_payload = None
    return JSONResponse(payload)


def _resolve_transcribe_status_code(payload: Dict[str, Any]) -> int:
    total = int(payload.get("total") or 0)
    failed = int(payload.get("failedCount") or 0)
    if failed <= 0:
        return 200
    if total > 0 and failed >= total:
        return 422
    return 207


def _require_model(
    config: Any,
    model_id: str,
    *,
    supported_model_ids: tuple[str, ...],
    field_name: str = "modelId",
) -> tuple[str, str]:
    normalized_model_id = require_supported_model_id(
        model_id,
        supported_model_ids=supported_model_ids,
        field_name=field_name,
    )
    resolved_model_ref = require_model_ref(
        normalized_model_id,
        models_dir=config.models_dir,
        field_name=field_name,
    )
    return normalized_model_id, resolved_model_ref


def _build_train_status_payload(task_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "requestId": str(meta.get("requestId") or task_id),
        "taskId": task_id,
        "status": meta.get("status"),
        "speaker": meta.get("speaker"),
        "voiceId": meta.get("voiceId"),
        "baseModelId": meta.get("trainModelId") or meta.get("baseModelId"),
        "jobId": meta.get("jobId"),
        "queuePosition": meta.get("queuePosition"),
        "error": meta.get("error"),
    }


def create_server(config: Any) -> FastAPI:
    executor = Executor(config)
    config = executor.config
    voice_registry = VoiceRegistry(config.data_dir / "voices")
    app = FastAPI(title="Qwen3-TTS API", version="0.1.0")
    app.state.executor = executor
    app.state.voice_registry = voice_registry
    api_prefix = "/api"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("shutdown")
    def shutdown_executor():
        executor.close()

    register_exception_handlers(app)

    @app.get(f"{api_prefix}/healthz")
    def healthz():
        return {
            "ok": True,
            "status": "healthy",
            "selectedDevice": config.device,
            "deviceMode": config.device_mode,
            "deviceName": config.device_name,
            "processPid": os.getpid(),
            "queueStatus": executor.snapshot(),
            "runtimePolicy": "executor-queue+single-process-task-thread",
            "dataDir": str(config.data_dir.resolve()),
            "modelsDir": str(config.models_dir.resolve()),
            "asrModelsDir": str((config.models_dir / "asr").resolve()),
            "customVoiceModelId": config.custom_voice_model_id,
        }

    @app.get(f"{api_prefix}/voices")
    def voices(modelId: str | None = None):
        requested_model_id = config.custom_voice_model_id if modelId is None else modelId
        _, resolved_model_ref = _require_model(
            config,
            requested_model_id,
            supported_model_ids=CUSTOM_VOICE_MODEL_IDS,
            field_name="modelId",
        )
        return {
            "ok": True,
            "voices": list_available_voices(
                voice_registry=voice_registry,
                models_dir=config.models_dir,
                runtime_model_id=resolved_model_ref,
            ),
        }

    @app.post(f"{api_prefix}/translate")
    async def translate(request: TranscribeRequest = Depends(TranscribeRequest.as_form)):
        payload = await request.to_payload()
        audio_payloads = payload.get("audios")
        if not isinstance(audio_payloads, list) or not audio_payloads:
            raise ValueError("`audios` must be a non-empty list")
        result = await asyncio.to_thread(
            executor.run,
            "transcribe",
            payload,
            meta={
                "audioCount": len(audio_payloads),
                "language": str(payload.get("language", "Auto") or "Auto"),
            },
        )
        response_payload = dict(result or {})
        return JSONResponse(response_payload, status_code=_resolve_transcribe_status_code(response_payload))

    @app.post(f"{api_prefix}/voiceDesign")
    def voice_design(request: VoiceDesignRequest):
        payload = request.model_dump(exclude_none=True)
        model_id, _ = _require_model(
            config,
            payload["modelId"],
            supported_model_ids=VOICE_DESIGN_MODEL_IDS,
            field_name="modelId",
        )
        result = executor.run(
            "voiceDesign",
            payload,
            meta={"modelId": model_id, "seed": int(payload.get("seed", 0))},
        )
        return _build_audio_response(_deserialize_audio_response(dict(result or {})))

    @app.post(f"{api_prefix}/clone")
    async def clone(request: CloneRequest = Depends(CloneRequest.as_form)):
        payload = await request.to_payload()
        model_id, _ = _require_model(
            config,
            payload["modelId"],
            supported_model_ids=BASE_MODEL_IDS,
            field_name="modelId",
        )
        result = await asyncio.to_thread(
            executor.run,
            "clone",
            payload,
            meta={"modelId": model_id, "seed": int(payload.get("seed", 0))},
        )
        return _build_audio_response(_deserialize_audio_response(dict(result or {})))

    @app.post(f"{api_prefix}/customVoice")
    def custom_voice(request: CustomVoiceRequest):
        payload = request.model_dump(exclude_none=True)
        requested_speaker = (payload.get("speaker") or "").strip()
        if not requested_speaker:
            raise ValueError("`speaker` is required")
        runtime_model_id, _ = _require_model(
            config,
            str(payload.get("modelId") or config.custom_voice_model_id).strip(),
            supported_model_ids=CUSTOM_VOICE_MODEL_IDS,
            field_name="modelId",
        )
        result = executor.run(
            "customVoice",
            payload,
            meta={
                "modelId": runtime_model_id,
                "speaker": requested_speaker,
                "seed": int(payload.get("seed", 0)),
            },
        )
        return _build_audio_response(_deserialize_audio_response(dict(result or {})))

    @app.post(f"{api_prefix}/trainVoice")
    async def train_voice(
        request: TrainVoiceRequest = Depends(TrainVoiceRequest.as_form),
        requestId: str = Query(..., min_length=1),
    ):
        payload = await request.to_payload()
        required = ["modelId", "tokenizerModelId", "speakerName", "samples", "refAudioBytes"]
        for key in required:
            if key not in payload:
                raise ValueError(f"Missing required field: {key}")
        if not payload["refAudioBytes"]:
            raise ValueError("`refAudio` file is empty")
        if not isinstance(payload["samples"], list) or not payload["samples"]:
            raise ValueError("`samples` must be a non-empty list")
        for sample in payload["samples"]:
            if "audioBytes" not in sample or "text" not in sample:
                raise ValueError("Each sample must include `audioBytes` and `text`")
            if not sample["audioBytes"]:
                raise ValueError("Sample audio file is empty")

        _, resolved_runtime_model_id = resolve_train_model_pair(
            payload["modelId"],
            config.models_dir,
        )
        resolved_train_model_id = resolve_public_base_model_id(
            payload["modelId"],
            config.models_dir,
        )
        builtin_speakers = get_builtin_speakers_for_base_model(
            config.models_dir,
            resolved_runtime_model_id,
        )
        voice_registry.assert_speaker_available(
            payload["speakerName"],
            speak_model_id=resolved_runtime_model_id,
            builtin_speakers=builtin_speakers,
        )

        task_id = str(requestId).strip()
        if not task_id:
            raise ValueError("`requestId` is required")
        queued_meta = {
            "taskId": task_id,
            "requestId": task_id,
            "status": "queued",
            "speaker": payload["speakerName"],
            "trainModelId": resolved_train_model_id,
            "speakModelId": resolved_runtime_model_id,
            "createdAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
        }
        result = await asyncio.to_thread(
            executor.run_train,
            task_id=task_id,
            payload={**payload, "taskId": task_id, "requestId": task_id},
            meta={"taskId": task_id, "requestId": task_id, "speaker": payload["speakerName"]},
            initial_meta=queued_meta,
        )
        return JSONResponse(_build_train_status_payload(task_id, dict(result or {})))

    @app.delete(f"{api_prefix}/trainVoice/{{request_id}}")
    def cancel_train_voice(request_id: str):
        task_id = str(request_id).strip()
        if not task_id:
            raise ValueError("`requestId` is required")
        result = executor.cancel_train(task_id)
        payload = _build_train_status_payload(task_id, dict(result or {}))
        return JSONResponse(payload)

    @app.delete(f"{api_prefix}/voices/{{voice_id}}")
    def delete_voice(voice_id: str):
        existing = voice_registry.find_by_voice_id(voice_id)
        if existing is None:
            raise FileNotFoundError(f"Unknown voiceId: {voice_id}")
        meta = {}
        meta_path = Path(existing.path) / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        removed = voice_registry.delete(voice_id)
        return {
            "ok": True,
            "voiceId": removed.voice_id,
            "speaker": removed.speaker,
            "baseModelId": str(meta.get("speakModelId") or removed.speak_model_id).strip(),
        }

    return app
