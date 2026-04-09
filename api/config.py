from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from api.common import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR


@dataclass
class ServerConfig:
    host: str
    port: int
    device: str
    device_mode: str
    device_name: str
    dtype: str
    flash_attn: bool
    data_dir: Path
    models_dir: Path
    workers: int
    max_gpu_queue_size: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FastAPI server for Qwen3-TTS.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, cuda:N, mps, cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--flash-attn", dest="flash_attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-gpu-queue-size", type=int, default=3)
    return parser
