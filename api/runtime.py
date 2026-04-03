from __future__ import annotations

import gc
import json
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from api.common import (
    ROOT_DIR,
    append_log,
    ensure_dir,
    guess_audio_extension,
    materialize_audio_input,
    read_json,
    resolve_relative_url,
    resolve_training_audio_dir,
    write_json,
)
from api.config import ServerConfig
from qwen_tts import Qwen3TTSModel


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
    def __init__(self, device: str, dtype: str, flash_attn: bool):
        self.device = device
        self.dtype = dtype
        self.flash_attn = flash_attn
        self._cache: Dict[str, Qwen3TTSModel] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str) -> Qwen3TTSModel:
        with self._lock:
            cached = self._cache.get(model_id)
            if cached is not None:
                return cached

            self._clear_locked()
            attn_impl = "flash_attention_2" if self.flash_attn else None
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=parse_dtype(self.dtype),
                attn_implementation=attn_impl,
            )
            self._cache = {model_id: model}
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

    def set(self, task_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._tasks[task_id] = dict(payload)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            value = self._tasks.get(task_id)
            return dict(value) if value else None


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
        self.models = ModelManager(config.device, config.dtype, config.flash_attn)
        self.tasks = TrainTaskStore()
        self.scheduler = GpuJobScheduler(config.max_gpu_queue_size)
        ensure_dir(config.data_dir)
        ensure_dir(config.data_dir / "voiceLibrary" / "drafts")
        ensure_dir(config.data_dir / "voiceLibrary" / "voices")

    def draft_dir(self, task_id: str) -> Path:
        return self.config.data_dir / "voiceLibrary" / "drafts" / task_id

    def voice_dir(self, voice_id: str) -> Path:
        return self.config.data_dir / "voiceLibrary" / "voices" / voice_id


def latest_checkpoint_dir(training_dir: Path) -> Optional[Path]:
    checkpoints = [path for path in training_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-epoch-")]
    if not checkpoints:
        return None

    def epoch_num(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except Exception:
            return -1

    checkpoints.sort(key=epoch_num)
    return checkpoints[-1]


def build_train_raw_jsonl(samples: List[Dict[str, Any]], ref_audio_path: Path, out_jsonl: Path) -> None:
    lines: List[str] = []
    for sample in samples:
        lines.append(
            json.dumps(
                {
                    "audio": str(Path(sample["localAudioPath"]).resolve()),
                    "text": sample["text"],
                    "ref_audio": str(ref_audio_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
    ensure_dir(out_jsonl.parent)
    with out_jsonl.open("w", encoding="utf-8") as file_obj:
        for line in lines:
            file_obj.write(line + "\n")


def run_command(cmd: List[str], cwd: Path, log_path: Path) -> None:
    append_log(log_path, f"$ {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def generate_preview(
    model_manager: ModelManager,
    model_id: str,
    voice: str,
    text: str,
    language: str,
    instruct: Optional[str],
    output_path: Path,
) -> tuple[Path, int]:
    import soundfile as sf

    model = model_manager.get(model_id)
    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        language=language or "Auto",
        speaker=voice,
        instruct=instruct or None,
    )
    ensure_dir(output_path.parent)
    sf.write(str(output_path), wavs[0], sample_rate)
    return output_path, sample_rate


def prepare_dataset_files(dataset_dir: Path, payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Path, Path]:
    dataset_dir = ensure_dir(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    ref_input = payload["refAudio"]
    ref_ext = guess_audio_extension(ref_input) if isinstance(ref_input, str) else ".wav"
    ref_path = dataset_dir / f"ref{ref_ext}"
    materialize_audio_input(ref_input, ref_path)

    samples_out: List[Dict[str, Any]] = []
    for index, sample in enumerate(payload["samples"], start=1):
        audio_input = sample["audio"]
        extension = guess_audio_extension(audio_input) if isinstance(audio_input, str) else ".wav"
        sample_path = dataset_dir / f"sample_{index:04d}{extension}"
        materialize_audio_input(audio_input, sample_path)
        samples_out.append(
            {
                "audio": audio_input,
                "text": sample["text"],
                "localAudioPath": str(sample_path),
            }
        )

    manifest = {
        "speakerName": payload["speakerName"],
        "refAudio": str(ref_path),
        "samples": [{"file": Path(item["localAudioPath"]).name, "text": item["text"]} for item in samples_out],
    }
    write_json(manifest_path, manifest)
    return samples_out, ref_path, manifest_path


def update_task_meta(draft_dir: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    meta_path = draft_dir / "meta.json"
    meta = read_json(meta_path, default={}) or {}
    meta.update(updates)
    write_json(meta_path, meta)
    return meta


def run_train_task(state: AppState, task_id: str, payload: Dict[str, Any]) -> None:
    draft_dir = state.draft_dir(task_id)
    training_dir = ensure_dir(draft_dir / "training")
    preview_dir = ensure_dir(draft_dir / "preview")
    log_path = training_dir / "train.log"
    dataset_dir, dataset_external = resolve_training_audio_dir(draft_dir / "dataset", payload)

    try:
        state.models.clear()
        running_meta = update_task_meta(
            draft_dir,
            {
                "taskId": task_id,
                "status": "running",
                "speakerName": payload["speakerName"],
                "modelId": payload["modelId"],
                "createdAt": datetime.now().isoformat(),
                "previewText": payload["previewText"],
                "previewInstruct": payload.get("previewInstruct"),
                "language": payload.get("language", "Auto"),
                "trainingAudioDir": str(dataset_dir.resolve()),
                "trainingAudioManagedExternally": dataset_external,
            },
        )
        state.tasks.set(task_id, running_meta)

        samples_out, ref_audio_path, manifest_path = prepare_dataset_files(dataset_dir, payload)
        raw_jsonl = training_dir / "train_raw.jsonl"
        build_train_raw_jsonl(samples_out, ref_audio_path, raw_jsonl)

        coded_jsonl = training_dir / "train_with_codes.jsonl"
        finetune_cwd = ROOT_DIR / "finetuning"

        run_command(
            [
                sys.executable,
                "prepare_data.py",
                "--device",
                state.config.device,
                "--tokenizer_model_path",
                payload["tokenizerModelId"],
                "--input_jsonl",
                str(raw_jsonl.resolve()),
                "--output_jsonl",
                str(coded_jsonl.resolve()),
            ],
            cwd=finetune_cwd,
            log_path=log_path,
        )

        run_command(
            [
                sys.executable,
                "sft_12hz.py",
                "--init_model_path",
                payload["modelId"],
                "--output_model_path",
                str(training_dir.resolve()),
                "--train_jsonl",
                str(coded_jsonl.resolve()),
                "--batch_size",
                str(payload.get("batchSize", 8)),
                "--lr",
                str(payload.get("lr", 2e-6)),
                "--num_epochs",
                str(payload.get("numEpochs", 3)),
                "--speaker_name",
                payload["speakerName"],
            ],
            cwd=finetune_cwd,
            log_path=log_path,
        )

        checkpoint_dir = latest_checkpoint_dir(training_dir)
        if checkpoint_dir is None:
            raise RuntimeError("No checkpoint generated")

        preview_path = preview_dir / "preview.wav"
        generate_preview(
            model_manager=state.models,
            model_id=str(checkpoint_dir.resolve()),
            voice=payload["speakerName"],
            text=payload["previewText"],
            language=payload.get("language", "Auto"),
            instruct=payload.get("previewInstruct"),
            output_path=preview_path,
        )

        ready_meta = update_task_meta(
            draft_dir,
            {
                "status": "preview_ready",
                "manifestPath": str(manifest_path.resolve()),
                "draftModelId": str(checkpoint_dir.resolve()),
                "previewAudioPath": str(preview_path.resolve()),
                "logPath": str(log_path.resolve()),
                "updatedAt": datetime.now().isoformat(),
            },
        )
        state.tasks.set(task_id, ready_meta)
    except Exception as exc:
        append_log(log_path, f"[ERROR] {type(exc).__name__}: {exc}")
        failed_meta = update_task_meta(
            draft_dir,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "logPath": str(log_path.resolve()),
                "updatedAt": datetime.now().isoformat(),
            },
        )
        state.tasks.set(task_id, failed_meta)
    finally:
        state.models.clear()


def load_task_meta(state: AppState, task_id: str) -> Optional[Dict[str, Any]]:
    draft_dir = state.draft_dir(task_id)
    meta = read_json(draft_dir / "meta.json", default=None)
    if meta:
        return meta
    return state.tasks.get(task_id)


def list_voices(state: AppState) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    voices_dir = state.config.data_dir / "voiceLibrary" / "voices"
    if not voices_dir.exists():
        return out
    for voice_dir in sorted(voices_dir.iterdir()):
        meta = read_json(voice_dir / "meta.json", default=None)
        if meta:
            out.append(meta)
    return out


def _make_job_id(kind: str) -> str:
    from api.common import make_id

    return make_id(kind)
