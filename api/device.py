from __future__ import annotations

import subprocess
import sys
from typing import Optional

import torch


def choose_best_cuda_index() -> int:
    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = int(torch.cuda.get_device_properties(index).total_memory)
        if free_bytes > best_free:
            best_index = index
            best_free = free_bytes
    return best_index


def is_mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        is_built = getattr(backend, "is_built", lambda: True)()
        is_available = getattr(backend, "is_available", lambda: False)()
        return bool(is_built and is_available)
    except Exception:
        return False


def detect_mps_device_name() -> str:
    fallback = "Apple Metal (MPS)"
    if sys.platform != "darwin":
        return fallback
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return fallback

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Chipset Model:"):
            return line.split(":", 1)[1].strip() or fallback
    return fallback


def confirm_cpu_startup(reason: Optional[str] = None) -> bool:
    if reason:
        print(reason, flush=True)
    print("Running Qwen3-TTS on CPU will be very slow.", flush=True)
    print("Do you want to continue on CPU? [y/N]", flush=True)
    try:
        answer = input("> ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def resolve_startup_device(requested: str) -> tuple[str, str, str]:
    requested = (requested or "auto").strip().lower()

    if requested in ("", "auto"):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            index = choose_best_cuda_index()
            torch.cuda.set_device(index)
            return f"cuda:{index}", "CUDA", torch.cuda.get_device_name(index)
        if is_mps_available():
            return "mps", "MPS", detect_mps_device_name()
        if not confirm_cpu_startup("No CUDA or MPS device detected."):
            raise SystemExit(1)
        return "cpu", "CPU", "CPU"

    if requested == "cpu":
        if not confirm_cpu_startup("CPU mode was explicitly requested."):
            raise SystemExit(1)
        return "cpu", "CPU", "CPU"

    if requested == "cuda":
        requested = "cuda:0"

    if requested.startswith("cuda:"):
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(f"Requested device `{requested}` but CUDA is not available")
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device: {requested}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device out of range: {requested}")
        torch.cuda.set_device(index)
        return f"cuda:{index}", "CUDA", torch.cuda.get_device_name(index)

    if requested == "mps":
        if not is_mps_available():
            raise RuntimeError("Requested device `mps` but MPS is not available")
        return "mps", "MPS", detect_mps_device_name()

    raise RuntimeError(f"Unsupported device: {requested}")
