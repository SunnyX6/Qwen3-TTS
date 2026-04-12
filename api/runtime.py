from __future__ import annotations

import gc
import queue
import threading
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from api.common import (
    ensure_dir,
    load_demo_audio_bytes,
    read_json,
    resolve_model_ref,
    write_stdout_line,
    write_json,
)
from api.asr import AsrManager
from api.config import ServerConfig
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.voice_registry import VoiceRegistry

TRAINING_AUDIO_SAMPLE_RATE = 24000


def parse_dtype(value: str) -> torch.dtype:
    value = (value or "").strip().lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


class ModelManager:
    def __init__(
        self,
        device: str,
        dtype: str,
        flash_attn: bool,
        models_dir: Path,
        voice_registry: VoiceRegistry,
    ):
        self.device = device
        self.dtype = dtype
        self.flash_attn = flash_attn
        self.models_dir = models_dir
        self.voice_registry = voice_registry
        self._cache: Dict[str, Qwen3TTSModel] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str) -> Qwen3TTSModel:
        resolved_model_id = resolve_model_ref(model_id, self.models_dir)
        with self._lock:
            cached = self._cache.get(resolved_model_id)
            if cached is not None:
                return cached

            self._clear_locked()
            attn_impl = "flash_attention_2" if self.flash_attn else None
            model = Qwen3TTSModel.from_pretrained(
                resolved_model_id,
                device_map=self.device,
                dtype=parse_dtype(self.dtype),
                attn_implementation=attn_impl,
                models_dir=self.models_dir,
            )
            model.bind_voice_registry(self.voice_registry, base_model_id=resolved_model_id)
            self._cache = {resolved_model_id: model}
            return model

    def clear(self) -> None:
        with self._lock:
            self._clear_locked()

    def _clear_locked(self) -> None:
        if not self._cache:
            return
        self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TrainTaskStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, int] = {}
        self._conditions: Dict[str, threading.Condition] = {}

    def set(self, task_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._tasks[task_id] = dict(payload)
            self._versions[task_id] = self._versions.get(task_id, 0) + 1
            condition = self._conditions.get(task_id)
            if condition is None:
                condition = threading.Condition(self._lock)
                self._conditions[task_id] = condition
            condition.notify_all()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            value = self._tasks.get(task_id)
            return dict(value) if value else None

    def snapshot(self, task_id: str) -> Optional[tuple[int, Dict[str, Any]]]:
        with self._lock:
            value = self._tasks.get(task_id)
            if value is None:
                return None
            return self._versions.get(task_id, 0), dict(value)

    def wait_for_update(
        self,
        task_id: str,
        after_version: int,
        timeout: Optional[float] = None,
    ) -> Optional[tuple[int, Dict[str, Any]]]:
        with self._lock:
            condition = self._conditions.get(task_id)
            if condition is None:
                return None

            def has_update() -> bool:
                return self._versions.get(task_id, 0) > after_version

            if not has_update():
                condition.wait(timeout=timeout)
            if not has_update():
                return None

            value = self._tasks.get(task_id)
            if value is None:
                return None
            return self._versions.get(task_id, 0), dict(value)


@dataclass
class JobHandle:
    job_id: str
    kind: str
    created_at: str
    event: threading.Event
    result: Any = None
    error: Optional[BaseException] = None


class QueueFullError(RuntimeError):
    pass


class GpuJobScheduler:
    def __init__(self, max_queue_size: int):
        self.max_queue_size = max_queue_size
        self._queue: "queue.Queue[tuple[JobHandle, Any]]" = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._active_job: Optional[Dict[str, Any]] = None
        self._queued_jobs: List[Dict[str, Any]] = []
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="gpu-job-worker")
        self._worker.start()

    def submit(self, kind: str, fn, meta: Optional[Dict[str, Any]] = None) -> tuple[JobHandle, int]:
        job_meta = dict(meta or {})
        handle = JobHandle(
            job_id=_make_job_id(kind),
            kind=kind,
            created_at=datetime.now().isoformat(),
            event=threading.Event(),
        )
        with self._lock:
            if len(self._queued_jobs) >= self.max_queue_size:
                raise QueueFullError(f"GPU queue is full (max={self.max_queue_size})")
            queued_item = {
                "jobId": handle.job_id,
                "kind": kind,
                "createdAt": handle.created_at,
                "meta": job_meta,
            }
            self._queued_jobs.append(queued_item)
            position = len(self._queued_jobs)
            self._queue.put((handle, fn))
        return handle, position

    def wait(self, handle: JobHandle, timeout: Optional[float] = None) -> Any:
        if not handle.event.wait(timeout):
            raise TimeoutError(f"Timed out while waiting for job {handle.job_id}")
        if handle.error is not None:
            raise handle.error
        return handle.result

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            queued_jobs = list(self._queued_jobs)
            active_job = dict(self._active_job) if self._active_job else None
        return {
            "activeJob": active_job,
            "queuedCount": len(queued_jobs),
            "queuedJobs": queued_jobs,
        }

    def _worker_loop(self) -> None:
        while True:
            handle, fn = self._queue.get()
            with self._lock:
                self._queued_jobs = [job for job in self._queued_jobs if job["jobId"] != handle.job_id]
                self._active_job = {
                    "jobId": handle.job_id,
                    "kind": handle.kind,
                    "createdAt": handle.created_at,
                    "startedAt": datetime.now().isoformat(),
                }
            try:
                handle.result = fn()
            except BaseException as exc:
                handle.error = exc
            finally:
                handle.event.set()
                with self._lock:
                    self._active_job = None
                self._queue.task_done()


class AppState:
    def __init__(self, config: ServerConfig):
        self.config = config
        ensure_dir(config.data_dir)
        ensure_dir(config.models_dir)
        self.voice_registry = VoiceRegistry(ensure_dir(config.data_dir / "voices"))
        self.models = ModelManager(
            config.device,
            config.dtype,
            config.flash_attn,
            config.models_dir,
            self.voice_registry,
        )
        self.asr = AsrManager(
            device=config.device,
            device_mode=config.device_mode,
            dtype=config.dtype,
            models_dir=config.models_dir,
        )
        self.tasks = TrainTaskStore()
        self.scheduler = GpuJobScheduler(config.max_gpu_queue_size)

    def voice_dir(self, voice_id: str) -> Path:
        return self.config.data_dir / "voices" / voice_id

    def train_tmp_root_dir(self) -> Path:
        return ensure_dir(self.config.data_dir / "train" / "tmp")


def _load_model_config_payload(model_ref: str) -> dict[str, Any]:
    config_path = Path(model_ref) / "config.json"
    payload = read_json(config_path, default=None)
    return payload if isinstance(payload, dict) else {}


def _replace_qwen_variant(model_ref: str, target_suffix: str) -> str:
    normalized = str(model_ref).strip()
    for source_suffix in ("-Base", "-CustomVoice"):
        if normalized.endswith(source_suffix):
            return normalized[: -len(source_suffix)] + target_suffix
    return normalized


def resolve_train_model_pair(
    model_id: str,
    models_dir: Path,
    *,
    expected_runtime_model_id: Optional[str] = None,
) -> tuple[str, str]:
    requested_model_ref = resolve_model_ref(model_id, models_dir)
    requested_config = _load_model_config_payload(requested_model_ref)
    requested_tts_model_type = str(requested_config.get("tts_model_type", "")).strip().lower()

    if requested_tts_model_type == "base":
        train_model_ref = requested_model_ref
        runtime_model_ref = resolve_model_ref(_replace_qwen_variant(requested_model_ref, "-CustomVoice"), models_dir)
    elif requested_tts_model_type == "custom_voice":
        runtime_model_ref = requested_model_ref
        train_model_ref = resolve_model_ref(_replace_qwen_variant(requested_model_ref, "-Base"), models_dir)
    elif requested_model_ref.endswith("-Base"):
        train_model_ref = requested_model_ref
        runtime_model_ref = resolve_model_ref(_replace_qwen_variant(requested_model_ref, "-CustomVoice"), models_dir)
    elif requested_model_ref.endswith("-CustomVoice"):
        runtime_model_ref = requested_model_ref
        train_model_ref = resolve_model_ref(_replace_qwen_variant(requested_model_ref, "-Base"), models_dir)
    else:
        raise ValueError("Training model must be a Qwen3-TTS Base or CustomVoice model from the same family")

    train_config = _load_model_config_payload(train_model_ref)
    runtime_config = _load_model_config_payload(runtime_model_ref)
    if train_config and str(train_config.get("tts_model_type", "")).strip().lower() != "base":
        raise ValueError(f"Training source model must resolve to a Base model: {train_model_ref}")
    if runtime_config and str(runtime_config.get("tts_model_type", "")).strip().lower() != "custom_voice":
        raise ValueError(f"Runtime backbone must resolve to a CustomVoice model: {runtime_model_ref}")

    if expected_runtime_model_id is not None:
        resolved_expected_runtime_model_id = resolve_model_ref(expected_runtime_model_id, models_dir)
        if runtime_model_ref != resolved_expected_runtime_model_id:
            raise ValueError(
                "Training request does not match the configured shared CustomVoice backbone: "
                f"expected {resolved_expected_runtime_model_id}, got {runtime_model_ref}"
            )
    return train_model_ref, runtime_model_ref


def resolve_public_base_model_id(model_id: str, models_dir: Path) -> str:
    resolved_model_id = resolve_model_ref(model_id, models_dir)
    config_payload = _load_model_config_payload(resolved_model_id)
    tts_model_type = str(config_payload.get("tts_model_type", "")).strip().lower()
    if tts_model_type == "custom_voice" or resolved_model_id.endswith("-CustomVoice"):
        paired_model_ref = _replace_qwen_variant(resolved_model_id, "-Base")
        try:
            return resolve_model_ref(paired_model_ref, models_dir)
        except Exception:
            return paired_model_ref
    return resolved_model_id


def prepare_training_records(payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    logs: List[str] = []
    ref_audio, ref_sample_rate = load_demo_audio_bytes(
        payload["refAudioBytes"],
        target_sample_rate=TRAINING_AUDIO_SAMPLE_RATE,
    )
    if ref_sample_rate != TRAINING_AUDIO_SAMPLE_RATE:
        raise ValueError(f"Reference audio must resolve to {TRAINING_AUDIO_SAMPLE_RATE}Hz")
    logs.append(
        "Prepared refAudio in memory "
        f"({payload.get('refAudioFilename') or 'refAudio'} -> {ref_sample_rate}Hz)"
    )

    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(payload["samples"], start=1):
        sample_audio, sample_rate = load_demo_audio_bytes(
            sample["audioBytes"],
            target_sample_rate=TRAINING_AUDIO_SAMPLE_RATE,
        )
        if sample_rate != TRAINING_AUDIO_SAMPLE_RATE:
            raise ValueError(f"Sample audio must resolve to {TRAINING_AUDIO_SAMPLE_RATE}Hz")
        records.append(
            {
                "audio": sample_audio,
                "text": sample["text"],
                "ref_audio": (ref_audio, ref_sample_rate),
            }
        )
        logs.append(
            "Prepared sample audio in memory "
            f"(#{index} {sample.get('audioFilename') or 'sample'} -> {sample_rate}Hz)"
        )
    return records, logs


def update_task_meta(current_meta: Optional[Dict[str, Any]], updates: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(current_meta or {})
    meta.update(updates)
    return meta


def write_train_log(task_id: str, speaker_name: str, message: str) -> None:
    write_stdout_line(f"[train][{task_id}][{speaker_name}] {message}")


def register_trained_voice(
    state: AppState,
    *,
    task_id: str,
    package_dir: Path,
    speaker_name: str,
    train_model_id: str,
    runtime_model_id: str,
    tokenizer_type: str = "12hz",
    tts_model_type: str = "custom_voice",
) -> dict[str, Any]:
    builtin_speakers = get_builtin_speakers_for_base_model(state, runtime_model_id)
    record = state.voice_registry.register(
        package_dir=package_dir,
        speaker=speaker_name,
        base_model_id=runtime_model_id,
        source_task_id=task_id,
        tokenizer_type=tokenizer_type,
        tts_model_type=tts_model_type,
        builtin_speakers=builtin_speakers,
    )
    meta_path = Path(record.path) / "meta.json"
    meta = read_json(meta_path, default={}) or {}
    meta["baseModelId"] = train_model_id
    write_json(meta_path, meta)
    return {
        "voiceId": record.voice_id,
        "speaker": record.speaker,
        "voice": record.speaker,
        "speakerName": record.speaker,
        "baseModelId": train_model_id,
        "createdAt": record.created_at,
        "sourceTaskId": record.source_task_id,
    }


def run_train_task(state: AppState, task_id: str, payload: Dict[str, Any]) -> None:
    speaker_name = payload["speakerName"]
    log = lambda message: write_train_log(task_id, speaker_name, message)

    try:
        from qwen_tts.training import SpeakerPackageTrainConfig, encode_training_records, train_speaker_package

        resolved_train_model_id, resolved_runtime_model_id = resolve_train_model_pair(
            payload["modelId"],
            state.config.models_dir,
            expected_runtime_model_id=state.config.custom_voice_model_id,
        )
        resolved_tokenizer_model_id = resolve_model_ref(payload["tokenizerModelId"], state.config.models_dir)
        state.asr.clear()
        state.models.clear()
        running_meta = update_task_meta(
            state.tasks.get(task_id),
            {
                "taskId": task_id,
                "status": "running",
                "speakerName": speaker_name,
                "baseModelId": resolved_train_model_id,
                "modelId": resolved_runtime_model_id,
                "createdAt": datetime.now().isoformat(),
                "language": payload.get("language", "Auto"),
            },
        )
        state.tasks.set(task_id, running_meta)
        log(f"Training source model: {resolved_train_model_id}")
        log(f"Runtime CustomVoice backbone: {resolved_runtime_model_id}")
        log(f"Tokenizer model: {resolved_tokenizer_model_id}")
        train_records, preparation_logs = prepare_training_records(payload)
        for message in preparation_logs:
            log(message)

        encoded_records = encode_training_records(
            records=train_records,
            tokenizer_model_path=resolved_tokenizer_model_id,
            device=state.config.device,
            models_dir=state.config.models_dir,
            audio_sample_rate=TRAINING_AUDIO_SAMPLE_RATE,
            log_fn=log,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"qwen3_tts_train_{task_id}_",
            dir=state.train_tmp_root_dir(),
        ) as temp_dir:
            train_output_dir = ensure_dir(Path(temp_dir) / "export")
            train_result = train_speaker_package(
                SpeakerPackageTrainConfig(
                    train_model_id=resolved_train_model_id,
                    runtime_model_id=resolved_runtime_model_id,
                    tokenizer_model_id=resolved_tokenizer_model_id,
                    train_jsonl=None,
                    train_records=encoded_records,
                    output_dir=train_output_dir,
                    speaker_name=speaker_name,
                    device=state.config.device,
                    dtype=state.config.dtype,
                    flash_attn=state.config.flash_attn,
                    batch_size=int(payload.get("batchSize", 8)),
                    lr=float(payload.get("lr", 2e-6)),
                    num_epochs=int(payload.get("numEpochs", 3)),
                    models_dir=state.config.models_dir,
                ),
                log_fn=log,
            )

            registered_voice = register_trained_voice(
                state,
                task_id=task_id,
                package_dir=train_result.package_dir,
                speaker_name=speaker_name,
                train_model_id=train_result.train_model_id,
                runtime_model_id=train_result.runtime_model_id,
                tokenizer_type=train_result.tokenizer_type,
                tts_model_type=train_result.tts_model_type,
            )

        registered_meta = update_task_meta(
            state.tasks.get(task_id),
            {
                "status": "completed",
                **registered_voice,
                "updatedAt": datetime.now().isoformat(),
            },
        )
        state.tasks.set(task_id, registered_meta)
    except Exception as exc:
        log(f"[ERROR] {type(exc).__name__}: {exc}")
        failed_meta = update_task_meta(
            state.tasks.get(task_id),
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updatedAt": datetime.now().isoformat(),
            },
        )
        state.tasks.set(task_id, failed_meta)
    finally:
        state.asr.clear()
        state.models.clear()


def load_task_meta(state: AppState, task_id: str) -> Optional[Dict[str, Any]]:
    return state.tasks.get(task_id)


def _normalize_voice_key(payload: Dict[str, Any]) -> str:
    return str(payload.get("voice") or payload.get("speaker") or "").strip().lower()


def _format_builtin_speaker_display_name(speaker: str) -> str:
    normalized = str(speaker).strip()
    if not normalized:
        return normalized
    return "_".join(part[:1].upper() + part[1:] for part in normalized.split("_"))


def _extract_builtin_speakers_from_config_payload(config_payload: dict[str, Any]) -> List[str]:
    talker_config = config_payload.get("talker_config")
    if not isinstance(talker_config, dict):
        return []

    speaker_map = talker_config.get("spk_id")
    if not isinstance(speaker_map, dict):
        return []

    return sorted(
        {str(name).strip() for name in speaker_map.keys() if str(name).strip()},
        key=str.lower,
    )


def _list_builtin_voices(state: AppState) -> List[Dict[str, Any]]:
    base_model_id = resolve_model_ref(state.config.custom_voice_model_id, state.config.models_dir)
    public_base_model_id = resolve_public_base_model_id(base_model_id, state.config.models_dir)
    speakers = get_builtin_speakers_for_base_model(state, base_model_id)
    out: List[Dict[str, Any]] = []
    for speaker in speakers:
        display_name = _format_builtin_speaker_display_name(speaker)
        out.append(
            {
                "voiceId": None,
                "speaker": display_name,
                "voice": display_name,
                "speakerName": display_name,
                "baseModelId": public_base_model_id,
                "enabled": True,
                "source": "builtin",
                "deletable": False,
            }
        )
    return out


def get_builtin_speakers_for_base_model(state: AppState, base_model_id: str) -> List[str]:
    speakers = _extract_builtin_speakers_from_config_payload(_load_model_config_payload(base_model_id))
    if not speakers:
        try:
            speakers = state.models.get(base_model_id).get_builtin_speakers()
        except Exception:
            speakers = []
    return speakers


def list_voices(state: AppState) -> List[Dict[str, Any]]:
    base_model_id = resolve_model_ref(state.config.custom_voice_model_id, state.config.models_dir)
    out: List[Dict[str, Any]] = []
    for record in state.voice_registry.list(base_model_id=base_model_id):
        meta = read_json(Path(record.path) / "meta.json", default=None)
        if meta:
            meta["baseModelId"] = str(
                meta.get("baseModelId") or resolve_public_base_model_id(record.base_model_id, state.config.models_dir)
            ).strip()
            meta.setdefault("voice", meta.get("speaker"))
            meta.setdefault("speakerName", meta.get("speaker"))
            meta["source"] = "custom"
            meta["deletable"] = True
            out.append(meta)
    return out


def list_available_voices(state: AppState) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for payload in _list_builtin_voices(state):
        key = _normalize_voice_key(payload)
        if key:
            merged[key] = payload
    for payload in list_voices(state):
        key = _normalize_voice_key(payload)
        if key:
            merged[key] = payload

    custom_voices = [item for item in merged.values() if item.get("source") != "builtin"]
    builtin_voices = [item for item in merged.values() if item.get("source") == "builtin"]

    custom_voices.sort(
        key=lambda item: (
            str(item.get("createdAt") or ""),
            str(item.get("voice") or item.get("speaker") or "").lower(),
        ),
        reverse=True,
    )
    builtin_voices.sort(key=lambda item: str(item.get("voice") or item.get("speaker") or "").lower())
    return custom_voices + builtin_voices


def _make_job_id(kind: str) -> str:
    from api.common import make_id

    return make_id(kind)
