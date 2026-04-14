from __future__ import annotations

import gc
import os
import signal
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from qwen_tts.inference.voice_registry import VoiceRegistry

from runtime.catalog import require_model_ref, resolve_model_ref
from runtime.task import AsrManager

_RESOURCE_TRACKER_LOCK = threading.RLock()
_RESOURCE_TRACKER_PIDS: set[int] = set()
_RESOURCE_TRACKER_RECORDER_INSTALLED = False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def release_torch_memory(device: Optional[str] = None) -> None:
    try:
        import torch
    except Exception:
        gc.collect()
        return

    normalized_device = str(device or "").strip().lower()
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                pass

    mps = getattr(torch, "mps", None)
    if mps is not None:
        should_flush_mps = normalized_device.startswith("mps")
        if not should_flush_mps:
            backend = getattr(torch.backends, "mps", None)
            if backend is not None:
                try:
                    should_flush_mps = bool(
                        getattr(backend, "is_built", lambda: False)()
                        and getattr(backend, "is_available", lambda: False)()
                    )
                except Exception:
                    should_flush_mps = False
        if should_flush_mps:
            synchronize = getattr(mps, "synchronize", None)
            if callable(synchronize):
                try:
                    synchronize()
                except Exception:
                    pass
            empty_cache = getattr(mps, "empty_cache", None)
            if callable(empty_cache):
                try:
                    empty_cache()
                except Exception:
                    pass

    gc.collect()


def install_resource_tracker_pid_recorder() -> None:
    global _RESOURCE_TRACKER_RECORDER_INSTALLED
    try:
        from multiprocessing import resource_tracker
    except Exception:
        return

    with _RESOURCE_TRACKER_LOCK:
        if _RESOURCE_TRACKER_RECORDER_INSTALLED:
            return
        tracker = resource_tracker._resource_tracker
        ensure_running = tracker.ensure_running

        def record_ensure_running(*args, **kwargs):
            result = ensure_running(*args, **kwargs)
            pid = getattr(tracker, "_pid", None)
            if pid is None:
                return result
            with _RESOURCE_TRACKER_LOCK:
                _RESOURCE_TRACKER_PIDS.add(int(pid))
            return result

        tracker.ensure_running = record_ensure_running
        _RESOURCE_TRACKER_RECORDER_INSTALLED = True


def stop_resource_tracker() -> None:
    try:
        from multiprocessing import resource_tracker
    except Exception:
        return

    tracker = resource_tracker._resource_tracker
    with _RESOURCE_TRACKER_LOCK:
        current_pid = getattr(tracker, "_pid", None)
        fd = getattr(tracker, "_fd", None)
        if current_pid is not None:
            _RESOURCE_TRACKER_PIDS.add(int(current_pid))
        pids = set(_RESOURCE_TRACKER_PIDS)
        _RESOURCE_TRACKER_PIDS.clear()

    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception:
            continue
        try:
            # Do not block request teardown waiting for the helper to exit.
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        except Exception:
            pass

    with _RESOURCE_TRACKER_LOCK:
        if getattr(tracker, "_pid", None) in pids:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            tracker._fd = None
            tracker._pid = None


def parse_dtype(value: str):
    import torch

    normalized = (value or "").strip().lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
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
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str):
        from qwen_tts import Qwen3TTSModel

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
        if self._cache:
            self._cache.clear()


class TaskStore:
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


class AppState:
    def __init__(self, config: Any):
        install_resource_tracker_pid_recorder()
        self.config = config
        _ensure_dir(config.data_dir)
        _ensure_dir(config.models_dir)
        require_model_ref(
            config.custom_voice_model_id,
            models_dir=config.models_dir,
            field_name="custom_voice_model_id",
        )
        self.voice_registry = VoiceRegistry(_ensure_dir(config.data_dir / "voices"))
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
        self.tasks = TaskStore()

    def train_tmp_root_dir(self) -> Path:
        return _ensure_dir(self.config.data_dir / "train" / "tmp")

    def close(self) -> None:
        try:
            self.asr.clear()
        finally:
            try:
                self.models.clear()
            finally:
                release_torch_memory(self.config.device)
                stop_resource_tracker()
