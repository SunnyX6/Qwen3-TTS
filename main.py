from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import FrameType, SimpleNamespace

from runtime.catalog import CUSTOM_VOICE_MODEL_IDS

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
DEFAULT_TIMEOUT_GRACEFUL_SHUTDOWN = 5


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FastAPI server for Qwen3-TTS.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, cuda:N, mps, cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--flash-attn", dest="flash_attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--max-gpu-queue-size", type=int, default=2)
    parser.add_argument("--timeout-graceful-shutdown", type=int, default=DEFAULT_TIMEOUT_GRACEFUL_SHUTDOWN)
    parser.add_argument(
        "--custom-voice-model-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        choices=CUSTOM_VOICE_MODEL_IDS,
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    pid_file = Path.cwd() / "pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    config = SimpleNamespace(
        host=args.host,
        port=int(args.port),
        device=str(args.device),
        device_mode="",
        device_name="",
        dtype=str(args.dtype),
        flash_attn=bool(args.flash_attn),
        keep_warm=False,
        data_dir=Path(args.data_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        max_gpu_queue_size=int(args.max_gpu_queue_size),
        custom_voice_model_id=str(args.custom_voice_model_id),
    )

    import uvicorn
    from api.server import create_server

    class GracefulExecutorShutdownServer(uvicorn.Server):
        def __init__(self, config, *, on_exit_signal=None):
            super().__init__(config)
            self._on_exit_signal = on_exit_signal
            self._exit_signal_handled = False

        def handle_exit(self, sig: int, frame: FrameType | None) -> None:
            if not self._exit_signal_handled:
                self._exit_signal_handled = True
                if self._on_exit_signal is not None:
                    try:
                        self._on_exit_signal()
                    except Exception:
                        pass
            super().handle_exit(sig, frame)

    try:
        app = create_server(config)
        executor = getattr(app.state, "executor", None)

        def request_executor_shutdown() -> None:
            if executor is not None:
                executor.request_shutdown()

        server = GracefulExecutorShutdownServer(
            uvicorn.Config(
                app,
                host=config.host,
                port=config.port,
                timeout_graceful_shutdown=max(0, int(args.timeout_graceful_shutdown)),
            ),
            on_exit_signal=request_executor_shutdown,
        )
        server.run()
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            if pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_file.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
