from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import ServerConfig, build_arg_parser

PYTORCH_INSTALL_URL = "https://pytorch.org/get-started/locally/"
FLASH_ATTN_DOC_URL = "https://github.com/Dao-AILab/flash-attention"
SERVER_CONFIG_ENV = "QWEN3_TTS_SERVER_CONFIG"


def _print_torch_setup_hint() -> None:
    print("Missing dependency `torch`.", file=sys.stderr)
    print(f"Install a PyTorch build that matches your machine from: {PYTORCH_INSTALL_URL}", file=sys.stderr)
    print("Then install project runtime dependencies with: pip install -e \".[runtime,api]\"", file=sys.stderr)


def _validate_flash_attn(enabled: bool, device_mode: str) -> bool:
    if not enabled:
        return True
    if device_mode != "CUDA":
        print("`--flash-attn` is enabled, but FlashAttention requires CUDA.", file=sys.stderr)
        print("Please switch to a CUDA device or start again with `--no-flash-attn`.", file=sys.stderr)
        return False
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        print("`--flash-attn` is enabled, but the `flash_attn` package is not installed.", file=sys.stderr)
        print(f"Please install `flash_attn` in the current environment. See: {FLASH_ATTN_DOC_URL}", file=sys.stderr)
        print("If you do not want to install it, start again with `--no-flash-attn`.", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"`flash_attn` is installed but failed to import: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Please fix the `flash_attn` installation in the current environment. See: {FLASH_ATTN_DOC_URL}", file=sys.stderr)
        print("If you do not want to use it, start again with `--no-flash-attn`.", file=sys.stderr)
        return False
    return True


def _serialize_config(config: ServerConfig) -> str:
    payload = {
        "host": config.host,
        "port": config.port,
        "device": config.device,
        "device_mode": config.device_mode,
        "device_name": config.device_name,
        "dtype": config.dtype,
        "flash_attn": config.flash_attn,
        "data_dir": str(config.data_dir),
        "models_dir": str(config.models_dir),
        "workers": config.workers,
        "max_gpu_queue_size": config.max_gpu_queue_size,
    }
    return json.dumps(payload)


def _load_config_from_env() -> ServerConfig:
    raw = os.environ.get(SERVER_CONFIG_ENV)
    if not raw:
        raise RuntimeError(f"Missing server config env var: {SERVER_CONFIG_ENV}")
    payload = json.loads(raw)
    return ServerConfig(
        host=str(payload["host"]),
        port=int(payload["port"]),
        device=str(payload["device"]),
        device_mode=str(payload["device_mode"]),
        device_name=str(payload["device_name"]),
        dtype=str(payload["dtype"]),
        flash_attn=bool(payload["flash_attn"]),
        data_dir=Path(payload["data_dir"]).resolve(),
        models_dir=Path(payload["models_dir"]).resolve(),
        workers=int(payload["workers"]),
        max_gpu_queue_size=int(payload["max_gpu_queue_size"]),
    )


def create_uvicorn_app():
    try:
        from api.runtime import AppState
        from api.app import create_app
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            _print_torch_setup_hint()
            raise RuntimeError("Missing dependency `torch` while starting API worker.") from exc
        raise

    config = _load_config_from_env()
    state = AppState(config)
    return create_app(state)


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        from api.device import resolve_startup_device
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            _print_torch_setup_hint()
            return 1
        raise

    try:
        device, device_mode, device_name = resolve_startup_device(args.device)
    except RuntimeError as exc:
        print(f"Device selection failed: {exc}", file=sys.stderr)
        return 1
    if not _validate_flash_attn(bool(args.flash_attn), device_mode):
        return 1
    if int(args.workers) < 1:
        print("`--workers` must be at least 1.", file=sys.stderr)
        return 1

    config = ServerConfig(
        host=args.host,
        port=args.port,
        device=device,
        device_mode=device_mode,
        device_name=device_name,
        dtype=args.dtype,
        flash_attn=bool(args.flash_attn),
        data_dir=Path(args.data_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        workers=int(args.workers),
        max_gpu_queue_size=int(args.max_gpu_queue_size),
    )

    try:
        import uvicorn
    except ImportError:
        print("Missing dependency `uvicorn`. Install with: pip install -e \".[api]\"", file=sys.stderr)
        return 1

    try:
        from api.runtime import AppState as _AppState  # noqa: F401
        from api.app import create_app as _create_app  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            _print_torch_setup_hint()
            return 1
        print(f"Missing API dependency: {exc}. Install with: pip install -e \".[api]\"", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Missing API dependency: {exc}. Install with: pip install -e \".[api]\"", file=sys.stderr)
        return 1

    print("Qwen3-TTS API starting...")
    print(f"Device mode: {config.device_mode}")
    print(f"Selected device: {config.device}")
    print(f"Device name: {config.device_name}")
    print(f"Data dir: {config.data_dir}")
    print(f"Models dir: {config.models_dir}")
    print(f"Workers: {config.workers}")
    print(f"Max GPU queue size: {config.max_gpu_queue_size}")
    if config.workers > 1:
        print("Note: each worker keeps its own model cache and GPU queue; GPU memory usage will increase.")
    print(f"Qwen3-TTS API listening on http://{config.host}:{config.port}/api")

    os.environ[SERVER_CONFIG_ENV] = _serialize_config(config)
    uvicorn.run(
        "api.main:create_uvicorn_app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        factory=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
