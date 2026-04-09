from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import ServerConfig, build_arg_parser


def main() -> int:
    args = build_arg_parser().parse_args()

    from api.device import resolve_startup_device

    try:
        device, device_mode, device_name = resolve_startup_device(args.device)
    except RuntimeError as exc:
        print(f"Device selection failed: {exc}", file=sys.stderr)
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
        max_gpu_queue_size=int(args.max_gpu_queue_size),
    )

    try:
        import uvicorn
    except ImportError:
        print("Missing dependency `uvicorn`. Install with: pip install -e \".[api]\"", file=sys.stderr)
        return 1

    try:
        from api.runtime import AppState
        from api.app import create_app
    except ImportError as exc:
        print(f"Missing API dependency: {exc}. Install with: pip install -e \".[api]\"", file=sys.stderr)
        return 1

    state = AppState(config)
    app = create_app(state)

    print("Qwen3-TTS API starting...")
    print(f"Device mode: {config.device_mode}")
    print(f"Selected device: {config.device}")
    print(f"Device name: {config.device_name}")
    print(f"Data dir: {config.data_dir}")
    print(f"Max GPU queue size: {config.max_gpu_queue_size}")
    print(f"Qwen3-TTS API listening on http://{config.host}:{config.port}/api")

    uvicorn.run(app, host=config.host, port=config.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
