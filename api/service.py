from __future__ import annotations

import base64
import mimetypes
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi.responses import FileResponse, JSONResponse, Response

from api.common import (
    DEFAULT_GENERATION_MAX_NEW_TOKENS,
    DEFAULT_GENERATION_REPETITION_PENALTY,
    DEFAULT_GENERATION_SEED,
    DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_GENERATION_TOP_P,
    ensure_dir,
    load_demo_audio_bytes,
    make_id,
    normalize_file_route,
    read_json,
    resolve_relative_url,
    set_generation_seed,
    tail_text,
    wav_bytes_from_array,
    write_json,
)
from api.runtime import (
    AppState,
    QueueFullError,
    list_voices,
    load_task_meta,
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
        }

    def get_voices(self) -> Dict[str, Any]:
        return {"ok": True, "voices": list_voices(self.state)}

    def get_train_status(self, task_id: str) -> Dict[str, Any]:
        meta = load_task_meta(self.state, task_id)
        if not meta:
            raise FileNotFoundError(f"Unknown taskId: {task_id}")

        result = {
            "ok": True,
            "taskId": task_id,
            "status": meta.get("status"),
            "speakerName": meta.get("speakerName"),
            "jobId": meta.get("jobId"),
            "queuePosition": meta.get("queuePosition"),
            "draftModelId": meta.get("draftModelId"),
            "error": meta.get("error"),
        }
        preview_audio_path = meta.get("previewAudioPath")
        if preview_audio_path:
            result["previewAudioUrl"] = resolve_relative_url(Path(preview_audio_path), self.state.config.data_dir)
        log_path = meta.get("logPath")
        if log_path:
            result["logUrl"] = resolve_relative_url(Path(log_path), self.state.config.data_dir)
            result["logTail"] = tail_text(Path(log_path))
        return result

    def get_data_file_response(self, relative_path: str) -> FileResponse:
        clean_relative = normalize_file_route(relative_path)
        file_path = (self.state.config.data_dir / clean_relative).resolve()
        data_root = self.state.config.data_dir.resolve()
        if not str(file_path).startswith(str(data_root)):
            raise PermissionError("Forbidden path")
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError("File not found")
        media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        return FileResponse(file_path, media_type=media_type)

    def voice_design(self, payload: Dict[str, Any]) -> AudioResult:
        model_id = payload["modelId"]
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)
        wavs, sample_rate = self._run_gpu_job_sync(
            kind="voiceDesign",
            meta={"modelId": model_id, "seed": generation_seed},
            fn=lambda: self._seeded_model_call(
                generation_seed,
                lambda: self.state.models.get(model_id).generate_voice_design(
                    text=payload["text"],
                    language=payload.get("language", "Auto"),
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
        wavs, sample_rate = self._run_gpu_job_sync(
            kind="clone",
            meta={"modelId": model_id, "seed": generation_seed},
            fn=lambda: self._seeded_model_call(
                generation_seed,
                lambda: self.state.models.get(model_id).generate_voice_clone(
                    text=payload["text"].strip(),
                    language=payload.get("language", "Auto"),
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
        model_id = payload["modelId"]
        requested_voice = payload.get("voice")
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)

        def _run_custom_voice():
            def _generate():
                model = self.state.models.get(model_id)
                voice = self._resolve_custom_voice(model, requested_voice)
                wavs, sample_rate = model.generate_custom_voice(
                    text=payload["text"],
                    language=payload.get("language", "Auto"),
                    speaker=voice,
                    instruct=payload.get("instruct"),
                    **generation_kwargs,
                )
                return wavs, sample_rate, voice

            return self._seeded_model_call(generation_seed, _generate)

        wavs, sample_rate, voice = self._run_gpu_job_sync(
            kind="customVoice",
            meta={"modelId": model_id, "voice": requested_voice, "seed": generation_seed},
            fn=_run_custom_voice,
        )
        return self._build_audio_result(
            wav=wavs[0],
            sample_rate=sample_rate,
            response_format=payload.get("responseFormat", "base64"),
            extra={"modelId": model_id, "voice": voice},
        )

    def train_voice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        required = ["modelId", "tokenizerModelId", "speakerName", "samples", "refAudioBytes", "previewText"]
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

        task_id = make_id("train")
        draft_dir = self.state.draft_dir(task_id)
        ensure_dir(draft_dir)

        queued_meta = update_task_meta(
            draft_dir,
            {
                "taskId": task_id,
                "status": "queued",
                "speakerName": payload["speakerName"],
                "modelId": payload["modelId"],
                "createdAt": datetime.now().isoformat(),
            },
        )
        self.state.tasks.set(task_id, queued_meta)

        try:
            handle, position = self.state.scheduler.submit(
                kind="trainVoice",
                meta={"taskId": task_id, "speakerName": payload["speakerName"]},
                fn=lambda: run_train_task(self.state, task_id, payload),
            )
        except QueueFullError:
            rejected = update_task_meta(
                draft_dir,
                {
                    "status": "rejected",
                    "error": f"GPU queue is full (max={self.state.config.max_gpu_queue_size})",
                    "updatedAt": datetime.now().isoformat(),
                },
            )
            self.state.tasks.set(task_id, rejected)
            raise

        updated = update_task_meta(
            draft_dir,
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

    def deploy_voice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = payload["taskId"]
        speaker_name = (payload.get("speakerName") or "").strip()
        if not speaker_name:
            raise ValueError("`speakerName` is required")

        meta = load_task_meta(self.state, task_id)
        if not meta:
            raise ValueError(f"Unknown taskId: {task_id}")
        if meta.get("status") != "preview_ready":
            raise ValueError(f"Task {task_id} is not ready to deploy")

        draft_dir = self.state.draft_dir(task_id)
        draft_model_id = meta.get("draftModelId")
        if not draft_model_id:
            raise ValueError("Draft model path missing")

        draft_model_path = Path(draft_model_id)
        config_path = draft_model_path / "config.json"
        config_dict = read_json(config_path, default=None)
        if not isinstance(config_dict, dict):
            raise ValueError("Draft model config missing or invalid")
        talker_config = config_dict.get("talker_config")
        if not isinstance(talker_config, dict):
            talker_config = {}
        talker_config["spk_id"] = {speaker_name: 3000}
        talker_config["spk_is_dialect"] = {speaker_name: False}
        config_dict["talker_config"] = talker_config
        write_json(config_path, config_dict)

        voice_id = make_id("voice")
        voice_dir = self.state.voice_dir(voice_id)
        ensure_dir(voice_dir)

        preview_dst = voice_dir / "preview"
        model_dst = voice_dir / "model"
        shutil.copytree(draft_dir / "preview", preview_dst, dirs_exist_ok=True)
        shutil.copytree(draft_model_path, model_dst, dirs_exist_ok=True)

        saved_meta = {
            "voiceId": voice_id,
            "voiceName": speaker_name,
            "voice": speaker_name,
            "speakerName": speaker_name,
            "modelId": str(model_dst.resolve()),
            "previewAudioUrl": resolve_relative_url(preview_dst / "preview.wav", self.state.config.data_dir),
            "sourceTaskId": task_id,
            "createdAt": datetime.now().isoformat(),
        }
        write_json(voice_dir / "meta.json", saved_meta)

        saved_task_meta = update_task_meta(
            draft_dir,
            {
                "status": "saved",
                "savedVoiceId": voice_id,
                "savedModelId": str(model_dst.resolve()),
                "savedSpeakerName": speaker_name,
                "updatedAt": datetime.now().isoformat(),
            },
        )
        self.state.tasks.set(task_id, saved_task_meta)
        return {"ok": True, **saved_meta}

    def _run_gpu_job_sync(self, kind: str, fn, meta: Optional[Dict[str, Any]] = None) -> Any:
        handle, _ = self.state.scheduler.submit(kind=kind, fn=fn, meta=meta)
        return self.state.scheduler.wait(handle)

    @staticmethod
    def _resolve_custom_voice(model: Qwen3TTSModel, requested_voice: Optional[str]) -> str:
        supported = model.get_supported_speakers() or []
        if requested_voice:
            return requested_voice
        if len(supported) == 1:
            return supported[0]
        raise ValueError("`voice` is required for models that expose multiple voices")
