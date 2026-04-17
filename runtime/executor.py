from __future__ import annotations

import builtins
import multiprocessing as mp
import queue
import signal
import threading
import uuid
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

_CANCEL_TERMINATE_TIMEOUT_SECONDS = 2.0


class QueueFullError(RuntimeError):
    pass


class ConflictError(RuntimeError):
    pass


class RequestCanceledError(RuntimeError):
    def __init__(self, message: str, *, request_id: str | None = None, kind: str | None = None):
        super().__init__(message)
        self.request_id = request_id
        self.kind = kind


class RequestNotFoundError(FileNotFoundError):
    pass


class ChildResultUnavailableError(RuntimeError):
    pass


class ChildExitedDuringCancelError(RuntimeError):
    pass


class ChildProcessCanceledError(BaseException):
    pass


def _make_job_id(kind: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _request_key(kind: str, request_id: str) -> tuple[str, str]:
    return str(kind).strip(), str(request_id).strip()


def _handle_sigterm(signum, frame):
    raise ChildProcessCanceledError("Canceled by request")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _confirm_cpu_startup(reason: str | None = None) -> bool:
    if reason:
        print(reason, flush=True)
    print("Running Qwen3-TTS on CPU will be very slow.", flush=True)
    print("Do you want to continue on CPU? [y/N]", flush=True)
    try:
        answer = input("> ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _build_runtime_config(config_payload: Dict[str, Any]) -> Any:
    from qwen_tts.device import (
        get_cpu_confirmation_reason,
        get_flash_attn_validation_errors,
        resolve_device,
    )

    config = SimpleNamespace(**dict(config_payload))
    config.data_dir = Path(config.data_dir).resolve()
    config.models_dir = Path(config.models_dir).resolve()

    resolved = resolve_device(str(config.device), set_cuda_device=True)
    flash_errors = get_flash_attn_validation_errors(
        enabled=bool(config.flash_attn),
        device_mode=resolved.device_mode,
    )
    if flash_errors:
        raise RuntimeError("; ".join(flash_errors))
    reason = get_cpu_confirmation_reason(str(config.device), resolved)
    cpu_confirmed = bool(getattr(config, "cpu_confirmed", False))
    if reason and not cpu_confirmed and not _confirm_cpu_startup(reason):
        raise RuntimeError("CPU startup was not confirmed")

    config.device = resolved.device
    config.device_mode = resolved.device_mode
    config.device_name = resolved.device_name
    config.cpu_confirmed = cpu_confirmed or resolved.device == "cpu"
    config.keep_warm = False
    _ensure_dir(config.data_dir)
    _ensure_dir(config.models_dir)
    return config


def run_child_task(
    config_payload: Dict[str, Any],
    kind: str,
    payload: Dict[str, Any],
    response_conn: Any,
) -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    state = None
    try:
        from runtime.state import AppState
        from runtime.task import TaskRunner, run_train_task

        config = _build_runtime_config(config_payload)
        state = AppState(config)
        task = TaskRunner(state)
        runtime_payload = dict(payload)
        request_id = str(runtime_payload.get("requestId") or "").strip()

        if kind == "trainVoice":
            if not request_id:
                raise ValueError("`requestId` is required")
            state.tasks.set(
                request_id,
                {
                    "taskId": request_id,
                    "requestId": request_id,
                    "status": "queued",
                    "speaker": runtime_payload.get("speakerName"),
                    "trainModelId": runtime_payload.get("modelId"),
                    "createdAt": datetime.now().isoformat(),
                    "updatedAt": datetime.now().isoformat(),
                },
            )
            run_train_task(state, request_id, runtime_payload)
            result = state.tasks.get(request_id) or {
                "taskId": request_id,
                "requestId": request_id,
                "status": "completed",
            }
        elif kind == "translate":
            result = task.transcribe(runtime_payload)
        elif kind == "voiceDesign":
            result = TaskRunner.serialize_audio_response(task.voice_design(runtime_payload))
        elif kind == "clone":
            result = TaskRunner.serialize_audio_response(task.clone(runtime_payload))
        elif kind == "customVoice":
            result = TaskRunner.serialize_audio_response(task.custom_voice(runtime_payload))
        else:
            raise ValueError(f"Unsupported runtime job kind: {kind}")

        response_conn.send({"ok": True, "result": result})
    except ChildProcessCanceledError as exc:
        response_conn.send(
            {
                "ok": False,
                "error": {
                    "type": "RequestCanceledError",
                    "message": str(exc),
                },
            }
        )
    except BaseException as exc:
        response_conn.send(
            {
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
    finally:
        try:
            response_conn.close()
        except Exception:
            pass
        if state is not None:
            try:
                state.close()
            except Exception:
                pass


def _raise_runtime_error(error: Dict[str, Any]) -> None:
    error_type = str(error.get("type") or "RuntimeError")
    message = str(error.get("message") or "RuntimeError")
    request_id = str(error.get("requestId") or "").strip() or None
    kind = str(error.get("kind") or "").strip() or None
    if error_type == "QueueFullError":
        raise QueueFullError(message)
    if error_type == "ConflictError":
        raise ConflictError(message)
    if error_type in {"FileNotFoundError", "RequestNotFoundError"}:
        raise RequestNotFoundError(message)
    if error_type == "RequestCanceledError":
        raise RequestCanceledError(message, request_id=request_id, kind=kind)
    exc_cls = getattr(builtins, error_type, None)
    if isinstance(exc_cls, type) and issubclass(exc_cls, BaseException):
        raise exc_cls(message)
    raise RuntimeError(f"{error_type}: {message}")


def _queued_snapshot_entry(request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jobId": request["jobId"],
        "kind": request["kind"],
        "requestId": request["requestId"],
        "createdAt": request["createdAt"],
        "status": request["status"],
        "meta": dict(request.get("meta") or {}),
    }


class Executor:
    def __init__(self, config: Any):
        self.config = config
        self._config_payload = {
            "device": str(config.device),
            "device_mode": str(getattr(config, "device_mode", "") or ""),
            "device_name": str(getattr(config, "device_name", "") or ""),
            "cpu_confirmed": str(config.device).strip().lower() == "cpu",
            "dtype": str(config.dtype),
            "flash_attn": bool(config.flash_attn),
            "keep_warm": False,
            "data_dir": str(Path(config.data_dir).resolve()),
            "models_dir": str(Path(config.models_dir).resolve()),
            "max_gpu_queue_size": int(config.max_gpu_queue_size),
        }
        self._commands: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._state_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._mp_ctx = mp.get_context("spawn")
        self._thread = threading.Thread(
            target=self._run_loop,
            name="qwen3-tts-executor",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> Dict[str, Any]:
        result = self._request({"type": "snapshot"})
        return dict(result or {})

    def run_request(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        request_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._request(
            {
                "type": "enqueue_request",
                "kind": str(kind),
                "request_id": str(request_id).strip(),
                "payload": dict(payload),
                "meta": dict(meta or {}),
            }
        )

    def cancel_request(self, request_id: str, *, kind: str) -> Dict[str, Any]:
        result = self._request(
            {
                "type": "cancel_request",
                "request_id": str(request_id).strip(),
                "kind": str(kind).strip(),
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

    def _reply_with_error(
        self,
        command: Dict[str, Any],
        error_type: str,
        message: str,
        *,
        request_id: str | None = None,
        kind: str | None = None,
    ) -> None:
        error_payload: Dict[str, Any] = {
            "type": error_type,
            "message": message,
        }
        if request_id:
            error_payload["requestId"] = request_id
        if kind:
            error_payload["kind"] = kind
        self._reply(command, {"ok": False, "error": error_payload})

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
            response = reply_queue.get() if timeout is None else reply_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out while waiting for executor response") from exc
        if not bool(response.get("ok")):
            _raise_runtime_error(dict(response.get("error") or {}))
        return response.get("result")

    def _build_snapshot(self, active: Optional[Dict[str, Any]], pending: deque[Dict[str, Any]]) -> Dict[str, Any]:
        active_snapshot = None
        if active is not None:
            active_snapshot = {
                "jobId": active["jobId"],
                "kind": active["kind"],
                "requestId": active["requestId"],
                "createdAt": active["createdAt"],
                "startedAt": active.get("startedAt"),
                "status": active.get("status"),
                "pid": active.get("pid"),
                "meta": dict(active.get("meta") or {}),
            }
        return {
            "activeJob": active_snapshot,
            "queuedCount": len(pending),
            "queuedJobs": [_queued_snapshot_entry(item) for item in pending],
        }

    def _release_request_memory(self, request: Dict[str, Any]) -> None:
        request["payload"] = None
        request["reply_queue"] = None
        request["cancel_waiters"] = []
        request["result_conn"] = None
        request["process"] = None

    def _start_process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        parent_conn, child_conn = self._mp_ctx.Pipe(duplex=False)
        process = self._mp_ctx.Process(
            target=run_child_task,
            args=(self._config_payload, request["kind"], dict(request["payload"] or {}), child_conn),
            name=f"qwen3-tts-{request['kind']}-{request['requestId']}",
        )
        process.start()
        child_conn.close()
        request["process"] = process
        request["result_conn"] = parent_conn
        request["pid"] = process.pid
        request["startedAt"] = datetime.now().isoformat()
        request["status"] = "running"
        request["cancel_deadline"] = None
        request["final_message"] = None
        return request

    def _close_active_handles(self, active: Dict[str, Any]) -> None:
        result_conn = active.get("result_conn")
        if result_conn is not None:
            try:
                result_conn.close()
            except Exception:
                pass
            active["result_conn"] = None
        process = active.get("process")
        if process is not None:
            try:
                process.join(timeout=0)
            except Exception:
                pass

    def _mark_canceled(self, request: Dict[str, Any]) -> None:
        request["status"] = "canceled"
        request["finishedAt"] = datetime.now().isoformat()

    def _cancel_queued_request(self, pending: deque[Dict[str, Any]], request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pending.remove(request)
        except ValueError:
            pass
        self._mark_canceled(request)
        self._reply_with_error(
            request,
            "RequestCanceledError",
            "Canceled by request",
            request_id=request["requestId"],
            kind=request["kind"],
        )
        self._release_request_memory(request)
        return {"requestId": request["requestId"], "kind": request["kind"]}

    def _begin_cancel_active(self, active: Dict[str, Any]) -> None:
        if active.get("cancel_requested"):
            return
        active["cancel_requested"] = True
        active["status"] = "canceling"
        active["cancel_deadline"] = datetime.now() + timedelta(seconds=_CANCEL_TERMINATE_TIMEOUT_SECONDS)
        process = active.get("process")
        if process is not None and process.is_alive():
            process.terminate()

    def _drain_active_process(self, active: Dict[str, Any]) -> None:
        result_conn = active.get("result_conn")
        if result_conn is not None:
            try:
                if result_conn.poll():
                    active["final_message"] = dict(result_conn.recv() or {})
            except EOFError:
                pass
            except OSError:
                pass

        process = active.get("process")
        if process is None:
            return
        if active.get("cancel_requested") and process.is_alive():
            deadline = active.get("cancel_deadline")
            if isinstance(deadline, datetime) and datetime.now() >= deadline:
                process.kill()
                active["cancel_deadline"] = None
        if not process.is_alive() and active.get("final_message") is None:
            if active.get("cancel_requested"):
                active["final_message"] = {
                    "ok": False,
                    "error": {
                        "type": "RequestCanceledError",
                        "message": "Canceled by request",
                        "requestId": active["requestId"],
                        "kind": active["kind"],
                    },
                }
            else:
                active["final_message"] = {
                    "ok": False,
                    "error": {
                        "type": "ChildResultUnavailableError",
                        "message": f"Runtime task for `{active['kind']}` exited without returning a result",
                        "requestId": active["requestId"],
                        "kind": active["kind"],
                    },
                }

    def _finalize_active_request(self, active: Dict[str, Any]) -> None:
        final_message = dict(active.get("final_message") or {})
        self._close_active_handles(active)

        if active.get("cancel_requested"):
            if final_message.get("ok"):
                active["status"] = "completed"
                active["finishedAt"] = datetime.now().isoformat()
                self._reply(active, final_message)
                for waiter in list(active.get("cancel_waiters") or []):
                    self._reply_with_error(
                        waiter,
                        "ConflictError",
                        f"Request kind={active['kind']}, requestId={active['requestId']} finished before it could be canceled",
                        request_id=active["requestId"],
                        kind=active["kind"],
                    )
            else:
                error = dict(final_message.get("error") or {})
                if str(error.get("type") or "") == "RequestCanceledError":
                    self._mark_canceled(active)
                    self._reply_with_error(
                        active,
                        "RequestCanceledError",
                        str(error.get("message") or "Canceled by request"),
                        request_id=active["requestId"],
                        kind=active["kind"],
                    )
                    for waiter in list(active.get("cancel_waiters") or []):
                        self._reply(
                            waiter,
                            {
                                "ok": True,
                                "result": {"requestId": active["requestId"], "kind": active["kind"]},
                            },
                        )
                else:
                    active["status"] = "failed"
                    active["finishedAt"] = datetime.now().isoformat()
                    self._reply(active, final_message)
                    for waiter in list(active.get("cancel_waiters") or []):
                        self._reply_with_error(
                            waiter,
                            "ConflictError",
                            f"Request kind={active['kind']}, requestId={active['requestId']} failed before it could be canceled",
                            request_id=active["requestId"],
                            kind=active["kind"],
                        )
        else:
            active["finishedAt"] = datetime.now().isoformat()
            active["status"] = "completed" if final_message.get("ok") else "failed"
            self._reply(active, final_message)

        self._release_request_memory(active)

    def _fail_pending_request(self, request: Dict[str, Any], message: str) -> None:
        request["status"] = "failed"
        request["finishedAt"] = datetime.now().isoformat()
        self._reply_with_error(
            request,
            "RuntimeError",
            message,
            request_id=request["requestId"],
            kind=request["kind"],
        )
        self._release_request_memory(request)

    def _run_loop(self) -> None:
        pending_requests: deque[Dict[str, Any]] = deque()
        requests_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
        active: Optional[Dict[str, Any]] = None
        shutting_down = False
        running = True

        while running:
            if active is not None:
                self._drain_active_process(active)
                process = active.get("process")
                if process is None:
                    if active.get("final_message") is not None:
                        self._finalize_active_request(active)
                        active = None
                elif not process.is_alive():
                    self._finalize_active_request(active)
                    active = None

            if not shutting_down and active is None and pending_requests:
                active = self._start_process(pending_requests.popleft())
                continue

            timeout = 0.05 if (active is not None or shutting_down) else None
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
                while pending_requests:
                    self._fail_pending_request(pending_requests.popleft(), "Executor is shutting down")
                if active is not None:
                    self._begin_cancel_active(active)
                continue

            if shutting_down:
                self._reply_with_error(command, "RuntimeError", "Executor is shutting down")
                continue

            if command_type == "snapshot":
                self._reply(command, {"ok": True, "result": self._build_snapshot(active, pending_requests)})
                continue

            if command_type == "enqueue_request":
                request_id = str(command.get("request_id") or "").strip()
                kind = str(command.get("kind") or "").strip()
                if not request_id:
                    self._reply_with_error(command, "ValueError", "`requestId` is required")
                    continue
                if not kind:
                    self._reply_with_error(command, "ValueError", "`kind` is required")
                    continue
                key = _request_key(kind, request_id)
                if key in requests_by_key:
                    self._reply_with_error(
                        command,
                        "ConflictError",
                        f"Request already exists: kind={kind}, requestId={request_id}",
                        request_id=request_id,
                        kind=kind,
                    )
                    continue
                if len(pending_requests) >= int(self.config.max_gpu_queue_size):
                    self._reply_with_error(
                        command,
                        "QueueFullError",
                        f"GPU queue is full (max={self.config.max_gpu_queue_size})",
                        request_id=request_id,
                        kind=kind,
                    )
                    continue
                payload = dict(command.get("payload") or {})
                payload["requestId"] = request_id
                if kind == "trainVoice":
                    payload["taskId"] = request_id
                request = {
                    "commandType": command_type,
                    "kind": kind,
                    "requestId": request_id,
                    "payload": payload,
                    "meta": dict(command.get("meta") or {}),
                    "createdAt": datetime.now().isoformat(),
                    "jobId": _make_job_id(kind),
                    "reply_queue": command.get("reply_queue"),
                    "cancel_waiters": [],
                    "status": "queued",
                    "process": None,
                    "result_conn": None,
                    "pid": None,
                    "startedAt": None,
                    "finishedAt": None,
                    "final_message": None,
                    "cancel_requested": False,
                    "cancel_deadline": None,
                }
                requests_by_key[key] = request
                pending_requests.append(request)
                continue

            if command_type == "cancel_request":
                request_id = str(command.get("request_id") or "").strip()
                kind = str(command.get("kind") or "").strip()
                if not request_id:
                    self._reply_with_error(command, "ValueError", "`requestId` is required")
                    continue
                if not kind:
                    self._reply_with_error(command, "ValueError", "`kind` is required")
                    continue
                request = requests_by_key.get(_request_key(kind, request_id))
                if request is None:
                    self._reply_with_error(
                        command,
                        "RequestNotFoundError",
                        f"Unknown request: kind={kind}, requestId={request_id}",
                        request_id=request_id,
                        kind=kind,
                    )
                    continue
                status = str(request.get("status") or "").strip().lower()
                if status == "queued":
                    result = self._cancel_queued_request(pending_requests, request)
                    self._reply(command, {"ok": True, "result": result})
                    continue
                if active is not None and active is request and status in {"running", "canceling"}:
                    active.setdefault("cancel_waiters", []).append(command)
                    self._begin_cancel_active(active)
                    continue
                if status == "canceled":
                    self._reply(command, {"ok": True, "result": {"requestId": request_id, "kind": kind}})
                    continue
                if status in {"completed", "failed"}:
                    self._reply_with_error(
                        command,
                        "ConflictError",
                        f"Request kind={kind}, requestId={request_id} is already `{status}` and cannot be canceled",
                        request_id=request_id,
                        kind=kind,
                    )
                    continue
                self._reply_with_error(
                    command,
                    "ConflictError",
                    f"Request kind={kind}, requestId={request_id} is already `{status}` and cannot be canceled",
                    request_id=request_id,
                    kind=kind,
                )
                continue

            self._reply_with_error(command, "ValueError", f"Unsupported executor command: {command_type}")

        with self._state_lock:
            self._closed = True
