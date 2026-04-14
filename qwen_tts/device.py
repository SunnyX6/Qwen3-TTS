from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional

FLASH_ATTN_DOC_URL = "https://github.com/Dao-AILab/flash-attention"


@dataclass(frozen=True)
class ResolvedDevice:
    device: str
    device_mode: str
    device_name: str


def _load_torch():
    import torch

    return torch


def normalize_requested_device(requested: str) -> str:
    normalized = (requested or "auto").strip().lower()
    if normalized == "cuda":
        return "cuda:0"
    return normalized


def choose_best_cuda_index(torch_module: Any) -> int:
    best_index = 0
    best_free = -1
    for index in range(torch_module.cuda.device_count()):
        try:
            with torch_module.cuda.device(index):
                free_bytes, _ = torch_module.cuda.mem_get_info()
        except Exception:
            free_bytes = int(torch_module.cuda.get_device_properties(index).total_memory)
        if free_bytes > best_free:
            best_index = index
            best_free = free_bytes
    return best_index


def is_mps_available(torch_module: Any) -> bool:
    backend = getattr(torch_module.backends, "mps", None)
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


def infer_device_mode(device: str) -> str:
    normalized = normalize_requested_device(device)
    if normalized.startswith("cuda:"):
        return "CUDA"
    if normalized == "mps":
        return "MPS"
    return "CPU"


def resolve_device(
    requested: str,
    *,
    torch_module: Optional[Any] = None,
    set_cuda_device: bool = False,
) -> ResolvedDevice:
    torch_module = torch_module or _load_torch()
    normalized = normalize_requested_device(requested)

    if normalized in ("", "auto"):
        if torch_module.cuda.is_available() and torch_module.cuda.device_count() > 0:
            index = choose_best_cuda_index(torch_module)
            if set_cuda_device:
                torch_module.cuda.set_device(index)
            return ResolvedDevice(
                device=f"cuda:{index}",
                device_mode="CUDA",
                device_name=torch_module.cuda.get_device_name(index),
            )
        if is_mps_available(torch_module):
            return ResolvedDevice(
                device="mps",
                device_mode="MPS",
                device_name=detect_mps_device_name(),
            )
        return ResolvedDevice(device="cpu", device_mode="CPU", device_name="CPU")

    if normalized == "cpu":
        return ResolvedDevice(device="cpu", device_mode="CPU", device_name="CPU")

    if normalized.startswith("cuda:"):
        if not torch_module.cuda.is_available() or torch_module.cuda.device_count() == 0:
            raise RuntimeError(f"Requested device `{normalized}` but CUDA is not available")
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device: {normalized}") from exc
        if index < 0 or index >= torch_module.cuda.device_count():
            raise RuntimeError(f"CUDA device out of range: {normalized}")
        if set_cuda_device:
            torch_module.cuda.set_device(index)
        return ResolvedDevice(
            device=f"cuda:{index}",
            device_mode="CUDA",
            device_name=torch_module.cuda.get_device_name(index),
        )

    if normalized == "mps":
        if not is_mps_available(torch_module):
            raise RuntimeError("Requested device `mps` but MPS is not available")
        return ResolvedDevice(
            device="mps",
            device_mode="MPS",
            device_name=detect_mps_device_name(),
        )

    raise RuntimeError(f"Unsupported device: {requested}")


def get_cpu_confirmation_reason(requested: str, resolved: ResolvedDevice) -> Optional[str]:
    if resolved.device != "cpu":
        return None
    normalized = normalize_requested_device(requested)
    if normalized == "cpu":
        return "CPU mode was explicitly requested."
    if normalized in ("", "auto"):
        return "No CUDA or MPS device detected."
    return None


def get_flash_attn_validation_errors(
    enabled: bool,
    *,
    device: Optional[str] = None,
    device_mode: Optional[str] = None,
) -> list[str]:
    if not enabled:
        return []

    resolved_mode = str(device_mode or infer_device_mode(device or "")).upper()
    if resolved_mode != "CUDA":
        return [
            "`--flash-attn` is enabled, but FlashAttention requires CUDA.",
            "Please switch to a CUDA device or start again with `--no-flash-attn`.",
        ]

    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        return [
            "`--flash-attn` is enabled, but the `flash_attn` package is not installed.",
            f"Please install `flash_attn` in the current environment. See: {FLASH_ATTN_DOC_URL}",
            "If you do not want to install it, start again with `--no-flash-attn`.",
        ]
    except Exception as exc:
        return [
            f"`flash_attn` is installed but failed to import: {type(exc).__name__}: {exc}",
            f"Please fix the `flash_attn` installation in the current environment. See: {FLASH_ATTN_DOC_URL}",
            "If you do not want to use it, start again with `--no-flash-attn`.",
        ]

    return []


def validate_flash_attn(
    enabled: bool,
    *,
    device: Optional[str] = None,
    device_mode: Optional[str] = None,
) -> None:
    errors = get_flash_attn_validation_errors(
        enabled,
        device=device,
        device_mode=device_mode,
    )
    if errors:
        raise RuntimeError("\n".join(errors))
