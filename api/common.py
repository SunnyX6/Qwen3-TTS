from __future__ import annotations

import io
import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
DEFAULT_GENERATION_SEED = 0
DEFAULT_GENERATION_MAX_NEW_TOKENS = 2048
DEFAULT_GENERATION_TEMPERATURE = 0.9
DEFAULT_GENERATION_TOP_P = 1.0
DEFAULT_GENERATION_REPETITION_PENALTY = 1.05
_STDOUT_LOCK = threading.Lock()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_id(prefix: str) -> str:
    return f"{prefix}_{now_ts()}_{uuid.uuid4().hex[:8]}"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_stdout_line(message: str) -> None:
    line = message.rstrip() + "\n"
    with _STDOUT_LOCK:
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
            sys.stdout.write(safe_line)
        sys.stdout.flush()


def guess_audio_extension_from_filename(filename: Optional[str], default: str = ".wav") -> str:
    suffix = Path(filename or "").suffix.lower().strip()
    if suffix and suffix.startswith(".") and suffix[1:].isalnum():
        return suffix
    return default


def materialize_uploaded_audio(audio_bytes: bytes, output_path: Path) -> Path:
    if not audio_bytes:
        raise ValueError("Uploaded audio file is empty")
    ensure_dir(output_path.parent)
    output_path.write_bytes(audio_bytes)
    return output_path


def wav_bytes_from_array(wav, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    return buffer.getvalue()


def set_generation_seed(seed: Optional[int]) -> int:
    import numpy as np
    import torch

    resolved_seed = DEFAULT_GENERATION_SEED if seed is None else int(seed)
    np.random.seed(resolved_seed % (2**32))
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    return resolved_seed


def normalize_demo_audio(wav, eps: float = 1e-12, clip: bool = True):
    import numpy as np

    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        peak = np.max(np.abs(y)) if y.size else 0.0
        if peak > 1.0 + 1e-6:
            y = y / (peak + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def load_demo_audio_bytes(audio_bytes: bytes, *, target_sample_rate: Optional[int] = None):
    import librosa
    import soundfile as sf

    if not audio_bytes:
        raise ValueError("Uploaded audio file is empty")

    with io.BytesIO(audio_bytes) as file_obj:
        try:
            wav, sample_rate = sf.read(file_obj, always_2d=False)
        except Exception:
            file_obj.seek(0)
            wav, sample_rate = librosa.load(file_obj, sr=None, mono=False)
    normalized_wav = normalize_demo_audio(wav)
    resolved_sample_rate = int(sample_rate)
    if target_sample_rate is not None and resolved_sample_rate != int(target_sample_rate):
        normalized_wav = librosa.resample(
            y=normalized_wav,
            orig_sr=resolved_sample_rate,
            target_sr=int(target_sample_rate),
        ).astype("float32")
        resolved_sample_rate = int(target_sample_rate)
    return normalize_demo_audio(normalized_wav), resolved_sample_rate


def resolve_model_ref(model_id: str, models_dir: Path) -> str:
    from qwen_tts.path_utils import resolve_pretrained_model_ref

    return resolve_pretrained_model_ref(model_id, models_dir=models_dir)
