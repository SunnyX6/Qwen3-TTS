from __future__ import annotations

import base64
import mimetypes
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi.responses import FileResponse, JSONResponse, Response

from api.common import ensure_dir, make_id, normalize_file_route, resolve_relative_url, tail_text, wav_bytes_from_array, write_json
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
            "queueStatus": self.state.scheduler.snapshot(),
            "dataDir": str(self.state.config.data_dir.resolve()),
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
            "trainingAudioDir": meta.get("trainingAudioDir"),
            "trainingAudioManagedExternally": meta.get("trainingAudioManagedExternally"),
            "manifestPath": meta.get("manifestPath"),
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
        wavs, sample_rate = self._run_gpu_job_sync(
            kind="voiceDesign",
            meta={"modelId": model_id},
            fn=lambda: self.state.models.get(model_id).generate_voice_design(
                text=payload["text"],
                language=payload.get("language", "Auto"),
                instruct=payload["instruct"],
            ),
        )
        return AudioResult(
            wav=wavs[0],
            sample_rate=sample_rate,
            response_format=payload.get("responseFormat", "base64"),
            extra={"modelId": model_id},
        )

    def clone(self, payload: Dict[str, Any]) -> AudioResult:
        model_id = payload["modelId"]
        wavs, sample_rate = self._run_gpu_job_sync(
            kind="clone",
            meta={"modelId": model_id},
            fn=lambda: self.state.models.get(model_id).generate_voice_clone(
                text=payload["text"],
                language=payload.get("language", "Auto"),
                ref_audio=payload["refAudio"],
                ref_text=payload.get("refText"),
                x_vector_only_mode=bool(payload.get("xVectorOnlyMode", False)),
            ),
        )
        return AudioResult(
            wav=wavs[0],
            sample_rate=sample_rate,
            response_format=payload.get("responseFormat", "base64"),
            extra={"modelId": model_id},
        )

    def custom_voice(self, payload: Dict[str, Any]) -> AudioResult:
        model_id = payload["modelId"]
        requested_voice = payload.get("voice")

        def _run_custom_voice():
            model = self.state.models.get(model_id)
            voice = self._resolve_custom_voice(model, requested_voice)
            wavs, sample_rate = model.generate_custom_voice(
                text=payload["text"],
                language=payload.get("language", "Auto"),
                speaker=voice,
                instruct=payload.get("instruct"),
            )
            return wavs, sample_rate, voice

        wavs, sample_rate, voice = self._run_gpu_job_sync(
            kind="customVoice",
            meta={"modelId": model_id, "voice": requested_voice},
            fn=_run_custom_voice,
        )
        return AudioResult(
            wav=wavs[0],
            sample_rate=sample_rate,
            response_format=payload.get("responseFormat", "base64"),
            extra={"modelId": model_id, "voice": voice},
        )

    def train_voice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        required = ["modelId", "tokenizerModelId", "speakerName", "samples", "refAudio", "previewText"]
        for key in required:
            if key not in payload:
                raise ValueError(f"Missing required field: {key}")
        if not isinstance(payload["samples"], list) or not payload["samples"]:
            raise ValueError("`samples` must be a non-empty list")
        for sample in payload["samples"]:
            if "audio" not in sample or "text" not in sample:
                raise ValueError("Each sample must include `audio` and `text`")

        task_id = make_id("train")
        draft_dir = self.state.draft_dir(task_id)
        ensure_dir(draft_dir)

        from api.common import resolve_training_audio_dir

        training_audio_dir, training_audio_external = resolve_training_audio_dir(draft_dir / "dataset", payload)
        queued_meta = update_task_meta(
            draft_dir,
            {
                "taskId": task_id,
                "status": "queued",
                "speakerName": payload["speakerName"],
                "modelId": payload["modelId"],
                "trainingAudioDir": str(training_audio_dir.resolve()),
                "trainingAudioManagedExternally": training_audio_external,
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
            "trainingAudioDir": str(training_audio_dir.resolve()),
            "trainingAudioManagedExternally": training_audio_external,
        }

    def save_voice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = payload["taskId"]
        voice_name = payload.get("voiceName")
        meta = load_task_meta(self.state, task_id)
        if not meta:
            raise ValueError(f"Unknown taskId: {task_id}")
        if meta.get("status") != "preview_ready":
            raise ValueError(f"Task {task_id} is not ready to save")

        draft_dir = self.state.draft_dir(task_id)
        draft_model_id = meta.get("draftModelId")
        if not draft_model_id:
            raise ValueError("Draft model path missing")

        voice_id = make_id("voice")
        voice_dir = self.state.voice_dir(voice_id)
        ensure_dir(voice_dir)

        preview_dst = voice_dir / "preview"
        model_dst = voice_dir / "model"
        shutil.copytree(draft_dir / "preview", preview_dst, dirs_exist_ok=True)
        shutil.copytree(Path(draft_model_id), model_dst, dirs_exist_ok=True)

        training_audio_dir = meta.get("trainingAudioDir")
        training_audio_external = bool(meta.get("trainingAudioManagedExternally"))
        manifest_path = meta.get("manifestPath")
        if training_audio_external:
            saved_training_audio_dir = training_audio_dir
            saved_manifest_path = manifest_path
        else:
            dataset_dst = voice_dir / "dataset"
            shutil.copytree(draft_dir / "dataset", dataset_dst, dirs_exist_ok=True)
            saved_training_audio_dir = str(dataset_dst.resolve())
            manifest_dst = dataset_dst / "manifest.json"
            saved_manifest_path = str(manifest_dst.resolve()) if manifest_dst.exists() else manifest_path

        saved_meta = {
            "voiceId": voice_id,
            "voiceName": voice_name or meta.get("speakerName"),
            "voice": meta.get("speakerName"),
            "modelId": str(model_dst.resolve()),
            "previewAudioUrl": resolve_relative_url(preview_dst / "preview.wav", self.state.config.data_dir),
            "sourceTaskId": task_id,
            "trainingAudioDir": saved_training_audio_dir,
            "trainingAudioManagedExternally": training_audio_external,
            "manifestPath": saved_manifest_path,
            "createdAt": datetime.now().isoformat(),
        }
        write_json(voice_dir / "meta.json", saved_meta)

        saved_task_meta = update_task_meta(
            draft_dir,
            {
                "status": "saved",
                "savedVoiceId": voice_id,
                "savedModelId": str(model_dst.resolve()),
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
