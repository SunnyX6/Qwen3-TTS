from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from api.asr import (
    ASR_AUDIO_SAMPLE_RATE,
    DEFAULT_ASR_MODEL_SIZE,
    display_transcribe_language,
    normalize_asr_model_size,
    normalize_transcribe_language,
    normalize_transcribe_provider,
)
from api.common import (
    DEFAULT_GENERATION_MAX_NEW_TOKENS,
    DEFAULT_GENERATION_REPETITION_PENALTY,
    DEFAULT_GENERATION_SEED,
    DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_GENERATION_TOP_P,
    load_demo_audio_bytes,
    make_id,
    read_json,
    set_generation_seed,
    wav_bytes_from_array,
)
from api.runtime import (
    AppState,
    QueueFullError,
    get_builtin_speakers_for_base_model,
    list_available_voices,
    load_task_meta,
    resolve_public_base_model_id,
    resolve_train_model_pair,
    run_train_task,
    update_task_meta,
)
from qwen_tts import Qwen3TTSModel


@dataclass
class AudioResult:
    wav: Any
    sample_rate: int
    response_format: str
    extra: Dict[str, Any]


class ApiService:
    def __init__(self, state: AppState):
        self.state = state

    @staticmethod
    def _resolve_requested_language(payload: Dict[str, Any]) -> str:
        language = payload.get("language", "Auto")
        language_value = str(language).strip()
        return language_value or "Auto"

    def _build_train_status_payload(self, task_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "taskId": task_id,
            "status": meta.get("status"),
            "speaker": meta.get("speaker"),
            "voiceId": meta.get("voiceId"),
            "baseModelId": meta.get("baseModelId"),
            "jobId": meta.get("jobId"),
            "queuePosition": meta.get("queuePosition"),
            "error": meta.get("error"),
        }

    def _extract_generation_kwargs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        max_new_tokens = payload.get("max_new_tokens")
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        repetition_penalty = payload.get("repetition_penalty")
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(
                DEFAULT_GENERATION_MAX_NEW_TOKENS
                if max_new_tokens is None
                else max_new_tokens
            ),
            "temperature": float(
                DEFAULT_GENERATION_TEMPERATURE if temperature is None else temperature
            ),
            "top_p": float(DEFAULT_GENERATION_TOP_P if top_p is None else top_p),
            "repetition_penalty": float(
                DEFAULT_GENERATION_REPETITION_PENALTY
                if repetition_penalty is None
                else repetition_penalty
            ),
        }
        return kwargs

    @staticmethod
    def _resolve_generation_seed(payload: Dict[str, Any]) -> int:
        seed = payload.get("seed")
        return DEFAULT_GENERATION_SEED if seed is None else int(seed)

    @staticmethod
    def _post_process_audio(wav, sample_rate: int):
        return wav

    def _seeded_model_call(self, seed: int, fn):
        set_generation_seed(seed)
        return fn()

    def _build_audio_result(
        self,
        wav,
        sample_rate: int,
        response_format: str,
        extra: Dict[str, Any],
    ) -> AudioResult:
        return AudioResult(
            wav=self._post_process_audio(wav, sample_rate),
            sample_rate=sample_rate,
            response_format=response_format,
            extra=extra,
        )

    def build_audio_response(self, result: AudioResult) -> Response:
        audio_bytes = wav_bytes_from_array(result.wav, result.sample_rate)
        if result.response_format == "wav":
            return Response(content=audio_bytes, media_type="audio/wav")

        payload = {
            "ok": True,
            "sampleRate": result.sample_rate,
            "audioBase64": base64.b64encode(audio_bytes).decode("utf-8"),
        }
        payload.update(result.extra)
        return JSONResponse(payload)

    def healthz(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "status": "healthy",
            "selectedDevice": self.state.config.device,
            "deviceMode": self.state.config.device_mode,
            "deviceName": self.state.config.device_name,
            "processPid": os.getpid(),
            "queueStatus": self.state.scheduler.snapshot(),
            "dataDir": str(self.state.config.data_dir.resolve()),
            "modelsDir": str(self.state.config.models_dir.resolve()),
            "asrModelsDir": str((self.state.config.models_dir / "asr").resolve()),
            "customVoiceModelId": self.state.config.custom_voice_model_id,
        }

    def get_voices(self) -> Dict[str, Any]:
        return {"ok": True, "voices": list_available_voices(self.state)}

    def transcribe(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        audio_payloads = payload.get("audios")
        if not isinstance(audio_payloads, list) or not audio_payloads:
            raise ValueError("`audios` must be a non-empty list")

        requested_language_code = normalize_transcribe_language(payload.get("language", "Auto"))
        requested_language = display_transcribe_language(requested_language_code) or "Auto"
        requested_provider = normalize_transcribe_provider(payload.get("provider", "auto"))
        raw_model_size = str(payload.get("modelSize") or DEFAULT_ASR_MODEL_SIZE).strip()
        requested_model_size = (
            raw_model_size
            if requested_provider == "funasr"
            else normalize_asr_model_size(raw_model_size)
        )
        if requested_provider == "funasr" and requested_language_code not in {"zh", "yue"}:
            raise ValueError("`provider=funasr` requires `language` to be Chinese or Cantonese")

        def _run_batch() -> list[Dict[str, Any]]:
            results: list[Dict[str, Any]] = []
            for index, audio_payload in enumerate(audio_payloads):
                filename = str(audio_payload.get("audioFilename") or f"audio_{index}.wav").strip()
                try:
                    audio_bytes = audio_payload.get("audioBytes")
                    if not audio_bytes:
                        raise ValueError("Uploaded audio file is empty")
                    audio, _ = load_demo_audio_bytes(
                        audio_bytes,
                        target_sample_rate=ASR_AUDIO_SAMPLE_RATE,
                    )
                    transcribe_result = self.state.asr.transcribe(
                        audio,
                        language=requested_language_code,
                        provider=requested_provider,
                        model_size=requested_model_size,
                    )
                    results.append(
                        {
                            "index": index,
                            "fileName": filename,
                            "ok": True,
                            "text": transcribe_result.get("text") or "",
                            "languageDetected": transcribe_result.get("languageDetected"),
                            "languageCode": transcribe_result.get("languageCode"),
                            "providerUsed": transcribe_result.get("providerUsed"),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "index": index,
                            "fileName": filename,
                            "ok": False,
                            "text": "",
                            "languageDetected": None,
                            "languageCode": None,
                            "providerUsed": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            return results

        results = self._run_gpu_job_sync(
            kind="transcribe",
            meta={
                "provider": requested_provider,
                "language": requested_language,
                "audioCount": len(audio_payloads),
            },
            fn=_run_batch,
        )
        success_count = sum(1 for item in results if item.get("ok"))
        failed_count = len(results) - success_count
        return {
            "ok": failed_count == 0,
            "providerRequested": requested_provider,
            "languageRequested": requested_language,
            "modelSize": requested_model_size,
            "total": len(results),
            "successCount": success_count,
            "failedCount": failed_count,
            "results": results,
        }

    @staticmethod
    def _is_terminal_train_status(status: Optional[str]) -> bool:
        normalized = (status or "").strip().lower()
        return normalized in {
            "completed",
            "failed",
            "error",
            "rejected",
            "canceled",
            "cancelled",
        }

    @staticmethod
    def _encode_sse_payload(payload: Dict[str, Any], *, event: str, event_id: int) -> str:
        raw_data = json.dumps(payload, ensure_ascii=False)
        return f"id: {event_id}\nevent: {event}\ndata: {raw_data}\n\n"

    def stream_train_status(self, task_id: str, request: Request) -> StreamingResponse:
        initial_meta = load_task_meta(self.state, task_id)
        if not initial_meta:
            raise FileNotFoundError(f"Unknown taskId: {task_id}")

        initial_snapshot = self.state.tasks.snapshot(task_id)

        async def event_stream():
            if initial_snapshot is None:
                initial_version = 0
                initial_payload = self._build_train_status_payload(task_id, initial_meta)
            else:
                initial_version, snapshot_meta = initial_snapshot
                initial_payload = self._build_train_status_payload(task_id, snapshot_meta)

            yield self._encode_sse_payload(
                initial_payload,
                event="status",
                event_id=initial_version,
            )
            if self._is_terminal_train_status(initial_payload.get("status")):
                return

            current_version = initial_version
            while True:
                if await request.is_disconnected():
                    return

                next_snapshot = await asyncio.to_thread(
                    self.state.tasks.wait_for_update,
                    task_id,
                    current_version,
                    15.0,
                )
                if next_snapshot is None:
                    yield ": keep-alive\n\n"
                    continue

                current_version, next_meta = next_snapshot
                next_payload = self._build_train_status_payload(task_id, next_meta)
                yield self._encode_sse_payload(
                    next_payload,
                    event="status",
                    event_id=current_version,
                )
                if self._is_terminal_train_status(next_payload.get("status")):
                    return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def voice_design(self, payload: Dict[str, Any]) -> AudioResult:
        model_id = payload["modelId"]
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)
        requested_language = self._resolve_requested_language(payload)
        wavs, sample_rate = self._run_gpu_job_sync(
            kind="voiceDesign",
            meta={"modelId": model_id, "seed": generation_seed},
            fn=lambda: self._seeded_model_call(
                generation_seed,
                lambda: self.state.models.get(model_id).generate_voice_design(
                    text=payload["text"],
                    language=requested_language,
                    instruct=payload["instruct"],
                    **generation_kwargs,
                ),
            ),
        )
        return self._build_audio_result(
            wav=wavs[0],
            sample_rate=sample_rate,
            response_format=payload.get("responseFormat", "base64"),
            extra={"modelId": model_id},
        )

    def clone(self, payload: Dict[str, Any]) -> AudioResult:
        model_id = payload["modelId"]
        ref_audio = load_demo_audio_bytes(payload["refAudioBytes"])
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)
        requested_language = self._resolve_requested_language(payload)
        wavs, sample_rate = self._run_gpu_job_sync(
            kind="clone",
            meta={"modelId": model_id, "seed": generation_seed},
            fn=lambda: self._seeded_model_call(
                generation_seed,
                lambda: self.state.models.get(model_id).generate_voice_clone(
                    text=payload["text"].strip(),
                    language=requested_language,
                    ref_audio=ref_audio,
                    ref_text=(payload.get("refText") or "").strip() or None,
                    x_vector_only_mode=bool(payload.get("xVectorOnlyMode", False)),
                    **generation_kwargs,
                ),
            ),
        )
        return self._build_audio_result(
            wav=wavs[0],
            sample_rate=sample_rate,
            response_format=payload.get("responseFormat", "base64"),
            extra={"modelId": model_id},
        )

    def custom_voice(self, payload: Dict[str, Any]) -> AudioResult:
        requested_speaker = (payload.get("speaker") or "").strip()
        if not requested_speaker:
            raise ValueError("`speaker` is required")
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)
        requested_language = self._resolve_requested_language(payload)
        requested_dialect = payload.get("dialect")

        def _run_custom_voice():
            def _generate():
                model = self.state.models.get(self.state.config.custom_voice_model_id)
                speaker = self._resolve_custom_voice(model, requested_speaker)
                wavs, sample_rate = model.generate_custom_voice(
                    text=payload["text"],
                    language=requested_language,
                    dialect=requested_dialect,
                    speaker=speaker,
                    instruct=payload.get("instruct"),
                    **generation_kwargs,
                )
                return wavs, sample_rate, speaker

            return self._seeded_model_call(generation_seed, _generate)

        wavs, sample_rate, speaker = self._run_gpu_job_sync(
            kind="customVoice",
            meta={
                "modelId": self.state.config.custom_voice_model_id,
                "speaker": requested_speaker,
                "seed": generation_seed,
            },
            fn=_run_custom_voice,
        )
        return self._build_audio_result(
            wav=wavs[0],
            sample_rate=sample_rate,
            response_format=payload.get("responseFormat", "base64"),
            extra={"modelId": self.state.config.custom_voice_model_id, "speaker": speaker},
        )

    def train_voice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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
            self.state.config.models_dir,
            expected_runtime_model_id=self.state.config.custom_voice_model_id,
        )
        resolved_train_model_id = resolve_public_base_model_id(payload["modelId"], self.state.config.models_dir)
        builtin_speakers = get_builtin_speakers_for_base_model(self.state, resolved_runtime_model_id)
        self.state.voice_registry.assert_speaker_available(
            payload["speakerName"],
            base_model_id=resolved_runtime_model_id,
            builtin_speakers=builtin_speakers,
        )

        task_id = make_id("train")
        queued_meta = update_task_meta(
            None,
            {
                "taskId": task_id,
                "status": "queued",
                "speaker": payload["speakerName"],
                "baseModelId": resolved_train_model_id,
                "modelId": resolved_runtime_model_id,
                "createdAt": datetime.now().isoformat(),
            },
        )
        self.state.tasks.set(task_id, queued_meta)

        try:
            handle, position = self.state.scheduler.submit(
                kind="trainVoice",
                meta={"taskId": task_id, "speaker": payload["speakerName"]},
                fn=lambda: run_train_task(self.state, task_id, payload),
            )
        except QueueFullError:
            rejected = update_task_meta(
                self.state.tasks.get(task_id),
                {
                    "status": "rejected",
                    "error": f"GPU queue is full (max={self.state.config.max_gpu_queue_size})",
                    "updatedAt": datetime.now().isoformat(),
                },
            )
            self.state.tasks.set(task_id, rejected)
            raise

        updated = update_task_meta(
            self.state.tasks.get(task_id),
            {
                "jobId": handle.job_id,
                "queuePosition": position,
                "updatedAt": datetime.now().isoformat(),
            },
        )
        self.state.tasks.set(task_id, updated)
        return {
            "ok": True,
            "taskId": task_id,
            "status": "queued",
            "queuePosition": position,
        }

    def delete_voice(self, voice_id: str) -> Dict[str, Any]:
        existing = self.state.voice_registry.find_by_voice_id(voice_id)
        if existing is None:
            raise FileNotFoundError(f"Unknown voiceId: {voice_id}")
        meta = read_json(Path(existing.path) / "meta.json", default={}) or {}
        public_base_model_id = str(
            meta.get("baseModelId") or resolve_public_base_model_id(existing.base_model_id, self.state.config.models_dir)
        ).strip()
        removed = self.state.voice_registry.delete(voice_id)
        self.state.models.clear()
        return {
            "ok": True,
            "voiceId": removed.voice_id,
            "speaker": removed.speaker,
            "baseModelId": public_base_model_id,
        }

    def _run_gpu_job_sync(self, kind: str, fn, meta: Optional[Dict[str, Any]] = None) -> Any:
        def _wrapped():
            self._prepare_gpu_job_models(kind)
            return fn()

        handle, _ = self.state.scheduler.submit(kind=kind, fn=_wrapped, meta=meta)
        return self.state.scheduler.wait(handle)

    def _prepare_gpu_job_models(self, kind: str) -> None:
        if kind == "transcribe":
            self.state.models.clear()
            return
        self.state.asr.clear()

    @staticmethod
    def _resolve_custom_voice(model: Qwen3TTSModel, requested_speaker: Optional[str]) -> str:
        supported = model.get_supported_speakers() or []
        if requested_speaker:
            return requested_speaker
        if len(supported) == 1:
            return supported[0]
        raise ValueError("`speaker` is required for models that expose multiple speakers")
