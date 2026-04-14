from __future__ import annotations

import builtins
import queue
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from qwen_tts.device import (
    get_cpu_confirmation_reason,
    get_flash_attn_validation_errors,
    resolve_device,
)

from runtime.catalog import require_model_ref


class QueueFullError(RuntimeError):
    pass


class ConflictError(RuntimeError):
    pass


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_job_id(kind: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _raise_runtime_error(error: Dict[str, Any]) -> None:
    error_type = str(error.get("type") or "RuntimeError")
    message = str(error.get("message") or "RuntimeError")
    if error_type == "QueueFullError":
        raise QueueFullError(message)
    if error_type == "ConflictError":
        raise ConflictError(message)
    exc_cls = getattr(builtins, error_type, None)
    if isinstance(exc_cls, type) and issubclass(exc_cls, BaseException):
        raise exc_cls(message)
    raise RuntimeError(f"{error_type}: {message}")


def _queued_snapshot_entry(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jobId": job["jobId"],
        "kind": job["kind"],
        "createdAt": job["createdAt"],
        "meta": dict(job.get("meta") or {}),
    }


def _make_snapshot(active_job: Optional[Dict[str, Any]], pending_jobs: deque[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "activeJob": dict(active_job) if active_job is not None else None,
        "queuedCount": len(pending_jobs),
        "queuedJobs": [_queued_snapshot_entry(job) for job in pending_jobs],
    }


def _confirm_cpu_startup(reason: Optional[str] = None) -> bool:
    if reason:
        print(reason, flush=True)
    print("Running Qwen3-TTS on CPU will be very slow.", flush=True)
    print("Do you want to continue on CPU? [y/N]", flush=True)
    try:
        answer = input("> ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _prepare_config(config: Any) -> Any:
    resolved = resolve_device(str(config.device), set_cuda_device=True)
    flash_errors = get_flash_attn_validation_errors(
        enabled=bool(config.flash_attn),
        device_mode=resolved.device_mode,
    )
    if flash_errors:
        raise RuntimeError("; ".join(flash_errors))
    reason = get_cpu_confirmation_reason(str(config.device), resolved)
    if reason and not _confirm_cpu_startup(reason):
        raise RuntimeError("CPU startup was not confirmed")

    config.device = resolved.device
    config.device_mode = resolved.device_mode
    config.device_name = resolved.device_name
    config.keep_warm = False
    config.data_dir = Path(config.data_dir).resolve()
    config.models_dir = Path(config.models_dir).resolve()
    _ensure_dir(config.data_dir)
    _ensure_dir(config.models_dir)
    require_model_ref(
        str(config.custom_voice_model_id),
        models_dir=config.models_dir,
        field_name="custom_voice_model_id",
    )
    return config


class Executor:
    def __init__(self, config: Any):
        self.config = _prepare_config(config)
        self._tasks_lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._task_versions: Dict[str, int] = {}
        self._task_conditions: Dict[str, threading.Condition] = {}
        self._commands: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._state_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="qwen3-tts-executor",
            daemon=True,
        )
        self._thread.start()

    def _start_task_thread(self, job: Dict[str, Any]) -> Dict[str, Any]:
        active_snapshot = {
            "jobId": job["jobId"],
            "kind": job["kind"],
            "createdAt": job["createdAt"],
            "startedAt": datetime.now().isoformat(),
            "meta": dict(job.get("meta") or {}),
        }
        active = {
            **job,
            "thread": None,
            "final_message": None,
            "activeSnapshot": active_snapshot,
        }
        if active["kind"] == "trainVoice":
            task_id = str(active.get("taskId") or "").strip()
            if task_id:
                current_meta = self.get_task_meta(task_id) or dict(active.get("initialMeta") or {})
                running_meta = dict(current_meta)
                running_meta.update(
                    {
                        "taskId": task_id,
                        "requestId": task_id,
                        "status": "running",
                        "jobId": active["jobId"],
                        "updatedAt": datetime.now().isoformat(),
                    }
                )
                self.set_task_meta(task_id, running_meta)
        thread = threading.Thread(
            target=self._run_task_thread,
            args=(active,),
            name=f"qwen3-tts-task-{job['kind']}",
            daemon=True,
        )
        active["thread"] = thread
        thread.start()
        return active

    def _run_task_thread(self, active: Dict[str, Any]) -> None:
        final_message = self._execute_task(active)
        self._events.put(
            {
                "type": "task_done",
                "jobId": active["jobId"],
                "message": final_message,
            }
        )

    def _execute_task(self, active: Dict[str, Any]) -> Dict[str, Any]:
        state = None
        try:
            from runtime.state import AppState
            from runtime.task import TaskRunner, run_train_task

            state = AppState(self.config)
            kind = str(active["kind"])
            payload = dict(active.get("payload") or {})

            if kind == "trainVoice":
                task_id = str(payload["taskId"])
                initial_meta = dict(active.get("initialMeta") or {})
                if initial_meta:
                    initial_meta["jobId"] = active["jobId"]
                    state.tasks.set(task_id, initial_meta)
                run_train_task(
                    state,
                    task_id,
                    payload,
                    cancel_event=active.get("cancel_event"),
                )
                result = state.tasks.get(task_id) or {"taskId": task_id, "requestId": task_id, "status": "completed"}
                return {"ok": True, "result": result}

            task = TaskRunner(state)
            if kind == "transcribe":
                result: Any = task.transcribe(payload)
            elif kind == "voiceDesign":
                result = TaskRunner.serialize_audio_response(task.voice_design(payload))
            elif kind == "clone":
                result = TaskRunner.serialize_audio_response(task.clone(payload))
            elif kind == "customVoice":
                result = TaskRunner.serialize_audio_response(task.custom_voice(payload))
            else:
                raise ValueError(f"Unsupported runtime job kind: {kind}")
            return {"ok": True, "result": result}
        except BaseException as exc:
            return {
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        finally:
            if state is not None:
                try:
                    state.close()
                except Exception:
                    pass

    def set_task_meta(self, task_id: str, payload: Dict[str, Any]) -> None:
        with self._tasks_lock:
            self._tasks[task_id] = dict(payload)
            self._task_versions[task_id] = self._task_versions.get(task_id, 0) + 1
            condition = self._task_conditions.get(task_id)
            if condition is None:
                condition = threading.Condition(self._tasks_lock)
                self._task_conditions[task_id] = condition
            condition.notify_all()

    def get_task_meta(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._tasks_lock:
            value = self._tasks.get(task_id)
            return dict(value) if value else None

    def task_snapshot(self, task_id: str) -> Optional[tuple[int, Dict[str, Any]]]:
        with self._tasks_lock:
            value = self._tasks.get(task_id)
            if value is None:
                return None
            return self._task_versions.get(task_id, 0), dict(value)

    def wait_for_task_update(
        self,
        task_id: str,
        after_version: int,
        timeout: Optional[float] = None,
    ) -> Optional[tuple[int, Dict[str, Any]]]:
        with self._tasks_lock:
            condition = self._task_conditions.get(task_id)
            if condition is None:
                return None

            def has_update() -> bool:
                return self._task_versions.get(task_id, 0) > after_version

            if not has_update():
                condition.wait(timeout=timeout)
            if not has_update():
                return None
            value = self._tasks.get(task_id)
            if value is None:
                return None
            return self._task_versions.get(task_id, 0), dict(value)

    def snapshot(self) -> Dict[str, Any]:
        result = self._request({"type": "snapshot"})
        return dict(result or {})

    def run(self, kind: str, payload: Dict[str, Any], *, meta: Optional[Dict[str, Any]] = None) -> Any:
        return self._request(
            {
                "type": "run_job",
                "kind": kind,
                "payload": dict(payload),
                "meta": dict(meta or {}),
            }
        )

    def run_train(
        self,
        *,
        task_id: str,
        payload: Dict[str, Any],
        meta: Dict[str, Any],
        initial_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self._request(
            {
                "type": "run_job",
                "kind": "trainVoice",
                "task_id": task_id,
                "payload": dict(payload),
                "meta": dict(meta),
                "initial_meta": dict(initial_meta),
            }
        )
        return dict(result or {})

    def cancel_train(self, task_id: str) -> Dict[str, Any]:
        result = self._request(
            {
                "type": "cancel_train_job",
                "task_id": str(task_id).strip(),
            }
        )
        return dict(result or {})

    def close(self) -> None:
        self.request_shutdown()
        self._thread.join(timeout=5.0)
        with self._state_lock:
            if not self._thread.is_alive():
                self._closed = True

    def request_shutdown(self) -> None:
        with self._state_lock:
            if self._closed or self._closing:
                return
            self._closing = True
        self._commands.put({"type": "shutdown"})

    @staticmethod
    def _reply(command: Dict[str, Any], payload: Dict[str, Any]) -> None:
        reply_queue = command.get("reply_queue")
        if reply_queue is not None:
            reply_queue.put(payload)

    def _reply_with_error(self, command: Dict[str, Any], error_type: str, message: str) -> None:
        self._reply(
            command,
            {
                "ok": False,
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
        )

    def _request(self, payload: Dict[str, Any], *, timeout: Optional[float] = None) -> Any:
        with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("Executor is shutting down")
        if not self._thread.is_alive():
            raise RuntimeError("Executor is not running")
        reply_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        message = dict(payload)
        message["reply_queue"] = reply_queue
        self._commands.put(message)
        try:
            if timeout is None:
                response = reply_queue.get()
            else:
                response = reply_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out while waiting for executor response") from exc
        if not bool(response.get("ok")):
            _raise_runtime_error(dict(response.get("error") or {}))
        return response.get("result")

    def _drain_task_events(self, active: Dict[str, Any]) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if event.get("type") != "task_done":
                continue
            if event.get("jobId") != active["jobId"]:
                continue
            active["final_message"] = dict(event.get("message") or {})

    def _merge_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        current_meta = self.get_task_meta(task_id) or {}
        merged_meta = dict(current_meta)
        merged_meta.update(result)
        self.set_task_meta(task_id, merged_meta)

    def _finalize_active_job(self, active: Dict[str, Any]) -> None:
        final_message = active.get("final_message")
        if final_message is None:
            final_message = {
                "ok": False,
                "error": {
                    "type": "RuntimeError",
                    "message": f"Runtime task for `{active['kind']}` exited without returning a result",
                },
            }

        task_id = str(active.get("taskId") or "").strip()
        if task_id:
            if final_message.get("ok"):
                result = final_message.get("result")
                if isinstance(result, dict):
                    self._merge_task_result(task_id, result)
            else:
                error = dict(final_message.get("error") or {})
                failed_meta = dict(self.get_task_meta(task_id) or active.get("initialMeta") or {})
                failed_meta.update(
                    {
                        "taskId": task_id,
                        "requestId": task_id,
                        "status": "failed",
                        "error": f"{error.get('type') or 'RuntimeError'}: {error.get('message') or 'Runtime failed'}",
                        "updatedAt": datetime.now().isoformat(),
                    }
                )
                self.set_task_meta(task_id, failed_meta)

        if str(active.get("kind") or "") == "trainVoice":
            cancel_waiters = list(active.get("cancel_waiters") or [])
            if cancel_waiters:
                terminal_meta = self.get_task_meta(task_id) if task_id else None
                if terminal_meta is not None:
                    for waiter in cancel_waiters:
                        self._reply(waiter, {"ok": True, "result": terminal_meta})
                else:
                    for waiter in cancel_waiters:
                        self._reply(waiter, final_message)

        if active["commandType"] == "run_job":
            self._reply(active, final_message)
            return

    def _fail_queued_job(self, job: Dict[str, Any], message: str) -> None:
        error = {
            "type": "RuntimeError",
            "message": message,
        }
        task_id = str(job.get("taskId") or "").strip()
        if task_id:
            failed_meta = dict(job.get("initialMeta") or {})
            failed_meta.update(
                {
                    "taskId": task_id,
                    "requestId": task_id,
                    "status": "failed",
                    "error": f"{error['type']}: {error['message']}",
                    "updatedAt": datetime.now().isoformat(),
                }
            )
            self.set_task_meta(task_id, failed_meta)
        if job["commandType"] == "run_job":
            self._reply(job, {"ok": False, "error": error})

    def _refresh_pending_train_queue_positions(self, pending_jobs: deque[Dict[str, Any]]) -> None:
        position = 0
        for job in pending_jobs:
            if str(job.get("kind") or "") != "trainVoice":
                continue
            task_id = str(job.get("taskId") or "").strip()
            if not task_id:
                continue
            position += 1
            initial_meta = dict(job.get("initialMeta") or {})
            initial_meta.update(
                {
                    "taskId": task_id,
                    "requestId": task_id,
                    "jobId": job["jobId"],
                    "queuePosition": position,
                    "updatedAt": datetime.now().isoformat(),
                }
            )
            job["initialMeta"] = initial_meta
            self._merge_task_result(task_id, initial_meta)

    def _cancel_pending_train_job(
        self,
        pending_jobs: deque[Dict[str, Any]],
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        for index, job in enumerate(pending_jobs):
            if str(job.get("kind") or "") != "trainVoice":
                continue
            if str(job.get("taskId") or "").strip() != task_id:
                continue
            del pending_jobs[index]
            canceled_meta = dict(job.get("initialMeta") or {})
            canceled_meta.update(
                {
                    "taskId": task_id,
                    "requestId": task_id,
                    "status": "canceled",
                    "jobId": job["jobId"],
                    "error": None,
                    "updatedAt": datetime.now().isoformat(),
                }
            )
            self.set_task_meta(task_id, canceled_meta)
            self._reply(job, {"ok": True, "result": canceled_meta})
            self._refresh_pending_train_queue_positions(pending_jobs)
            return canceled_meta
        return None

    def _shutdown_active_job(self, active: Dict[str, Any], message: str) -> None:
        cancel_event = active.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        active["final_message"] = {
            "ok": False,
            "error": {
                "type": "RuntimeError",
                "message": message,
            },
        }
        self._finalize_active_job(active)

    def _run_loop(self) -> None:
        pending_jobs: deque[Dict[str, Any]] = deque()
        active: Optional[Dict[str, Any]] = None
        shutting_down = False
        running = True

        while running:
            if active is not None:
                self._drain_task_events(active)
                thread = active["thread"]
                if active.get("final_message") is not None:
                    self._finalize_active_job(active)
                    active = None
                elif not thread.is_alive():
                    self._finalize_active_job(active)
                    active = None

            if not shutting_down and active is None and pending_jobs:
                active = self._start_task_thread(pending_jobs.popleft())
                self._refresh_pending_train_queue_positions(pending_jobs)
                continue

            timeout = 0.1 if (active is not None or shutting_down) else None
            try:
                command = self._commands.get(timeout=timeout)
            except queue.Empty:
                command = None

            if command is None:
                if shutting_down and active is None:
                    running = False
                continue

            command_type = str(command.get("type") or "").strip()

            if command_type == "shutdown":
                shutting_down = True
                while pending_jobs:
                    self._fail_queued_job(pending_jobs.popleft(), "Executor is shutting down")
                if active is not None:
                    self._shutdown_active_job(active, "Executor is shutting down")
                    active = None
                continue

            if shutting_down:
                self._reply_with_error(command, "RuntimeError", "Executor is shutting down")
                continue

            if command_type == "snapshot":
                self._reply(
                    command,
                    {
                        "ok": True,
                        "result": _make_snapshot(
                            active["activeSnapshot"] if active is not None else None,
                            pending_jobs,
                        ),
                    },
                )
                continue

            if command_type == "cancel_train_job":
                task_id = str(command.get("task_id") or "").strip()
                if not task_id:
                    self._reply_with_error(command, "ValueError", "`requestId` is required")
                    continue
                canceled_meta = self._cancel_pending_train_job(pending_jobs, task_id)
                if canceled_meta is not None:
                    self._reply(command, {"ok": True, "result": canceled_meta})
                    continue
                if active is not None and str(active.get("kind") or "") == "trainVoice":
                    active_task_id = str(active.get("taskId") or "").strip()
                    if active_task_id == task_id:
                        current_meta = self.get_task_meta(task_id) or dict(active.get("initialMeta") or {})
                        current_status = str(current_meta.get("status") or "").strip().lower()
                        if current_status in {"canceled", "cancelled"}:
                            self._reply(command, {"ok": True, "result": current_meta})
                            continue
                        active.setdefault("cancel_waiters", []).append(command)
                        if current_status == "canceling":
                            continue
                        cancel_event = active.get("cancel_event")
                        if cancel_event is not None:
                            cancel_event.set()
                        canceling_meta = dict(current_meta)
                        canceling_meta.update(
                            {
                                "taskId": task_id,
                                "requestId": task_id,
                                "status": "canceling",
                                "jobId": active["jobId"],
                                "error": None,
                                "updatedAt": datetime.now().isoformat(),
                            }
                        )
                        self.set_task_meta(task_id, canceling_meta)
                        continue
                existing_meta = self.get_task_meta(task_id)
                if existing_meta is None:
                    self._reply_with_error(command, "FileNotFoundError", f"Unknown train requestId: {task_id}")
                    continue
                existing_status = str(existing_meta.get("status") or "").strip().lower()
                if existing_status in {"canceling", "canceled", "cancelled"}:
                    self._reply(command, {"ok": True, "result": existing_meta})
                    continue
                self._reply_with_error(
                    command,
                    "ConflictError",
                    f"Train requestId `{task_id}` is already `{existing_status or 'completed'}` and cannot be canceled",
                )
                continue

            if command_type != "run_job":
                self._reply_with_error(command, "ValueError", f"Unsupported executor command: {command_type}")
                continue

            if len(pending_jobs) >= int(self.config.max_gpu_queue_size):
                self._reply_with_error(
                    command,
                    "QueueFullError",
                    f"GPU queue is full (max={self.config.max_gpu_queue_size})",
                )
                continue

            kind = str(command["kind"])
            payload = dict(command.get("payload") or {})
            task_id = str(command.get("task_id") or payload.get("taskId") or "").strip()
            if kind == "trainVoice":
                if not task_id:
                    self._reply_with_error(command, "ValueError", "`requestId` is required")
                    continue
                if self.get_task_meta(task_id) is not None:
                    self._reply_with_error(command, "ConflictError", f"Train requestId already exists: {task_id}")
                    continue

            queued_job = {
                "commandType": command_type,
                "kind": kind,
                "payload": payload,
                "meta": dict(command.get("meta") or {}),
                "createdAt": datetime.now().isoformat(),
                "jobId": _make_job_id(kind),
                "reply_queue": command.get("reply_queue"),
                "taskId": task_id or None,
                "initialMeta": dict(command.get("initial_meta") or {}),
                "cancel_event": threading.Event() if kind == "trainVoice" else None,
                "cancel_waiters": [] if kind == "trainVoice" else None,
            }
            pending_jobs.append(queued_job)

            if kind == "trainVoice":
                queued_meta = dict(queued_job["initialMeta"])
                queued_meta.update(
                    {
                        "taskId": task_id,
                        "requestId": task_id,
                        "status": "queued",
                        "jobId": queued_job["jobId"],
                        "error": None,
                        "updatedAt": datetime.now().isoformat(),
                    }
                )
                queued_job["initialMeta"] = queued_meta
                self.set_task_meta(task_id, queued_meta)
                self._refresh_pending_train_queue_positions(pending_jobs)

        with self._state_lock:
            self._closed = True
