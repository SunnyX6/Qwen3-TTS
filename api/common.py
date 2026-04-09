from __future__ import annotations

import base64
import io
import json
import mimetypes
import posixpath
import re
import shutil
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"


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


def append_log(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(message.rstrip() + "\n")


def tail_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def resolve_relative_url(path: Path, data_dir: Path) -> str:
    relative = path.resolve().relative_to(data_dir.resolve())
    return "/api/files/" + "/".join(relative.parts)


def normalize_file_route(relative_path: str) -> str:
    return posixpath.normpath(unquote(relative_path)).lstrip("/")


def _is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _guess_ext_from_data_uri(value: str) -> str:
    if not value.startswith("data:"):
        return ".wav"
    media = value.split(",", 1)[0]
    mime = media[5:].split(";", 1)[0]
    extension = mimetypes.guess_extension(mime) or ".wav"
    if extension == ".jpe":
        return ".jpg"
    return extension


def guess_audio_extension(value: str) -> str:
    if value.startswith("data:audio"):
        return _guess_ext_from_data_uri(value)
    if _is_url(value):
        suffix = Path(urlparse(value).path).suffix
        return suffix or ".wav"
    return Path(value).suffix or ".wav"


def _is_probably_base64(value: str) -> bool:
    if value.startswith("data:audio"):
        return True
    return ("/" not in value and "\\" not in value) and len(value) > 256


def _decode_base64_bytes(value: str) -> bytes:
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def materialize_audio_input(audio: str, output_path: Path) -> Path:
    ensure_dir(output_path.parent)

    if _is_url(audio):
        with urllib.request.urlopen(audio) as response:
            output_path.write_bytes(response.read())
        return output_path

    audio_path = Path(audio)
    if audio_path.exists():
        shutil.copy2(audio_path, output_path)
        return output_path

    if _is_probably_base64(audio):
        output_path.write_bytes(_decode_base64_bytes(audio))
        return output_path

    raise ValueError(f"Unsupported audio input: {audio[:80]}")


def wav_bytes_from_array(wav, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    return buffer.getvalue()


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


def load_demo_audio_input(audio: str):
    import librosa
    import soundfile as sf

    if _is_url(audio):
        with urllib.request.urlopen(audio) as response:
            payload = response.read()
        with io.BytesIO(payload) as file_obj:
            wav, sample_rate = sf.read(file_obj, always_2d=False)
        return normalize_demo_audio(wav), int(sample_rate)

    audio_path = Path(audio)
    if audio_path.exists():
        try:
            wav, sample_rate = sf.read(audio_path, always_2d=False)
        except Exception:
            wav, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        return normalize_demo_audio(wav), int(sample_rate)

    if _is_probably_base64(audio):
        with io.BytesIO(_decode_base64_bytes(audio)) as file_obj:
            wav, sample_rate = sf.read(file_obj, always_2d=False)
        return normalize_demo_audio(wav), int(sample_rate)

    raise ValueError(f"Unsupported audio input: {audio[:80]}")


def get_training_audio_dir_value(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("training-audio-dir", "trainingAudioDir", "training_audio_dir"):
        value = payload.get(key)
        if value is not None:
            value = str(value).strip()
            return value or None
    return None


def resolve_training_audio_dir(default_dir: Path, payload: Dict[str, Any]) -> tuple[Path, bool]:
    requested = get_training_audio_dir_value(payload)
    if not requested:
        return ensure_dir(default_dir), False

    dataset_dir = Path(requested).expanduser().resolve()
    if dataset_dir.exists() and not dataset_dir.is_dir():
        raise ValueError(f"`training-audio-dir` is not a directory: {dataset_dir}")
    return ensure_dir(dataset_dir), True


def resolve_model_ref(model_id: str, models_dir: Path) -> str:
    from qwen_tts.path_utils import resolve_pretrained_model_ref

    return resolve_pretrained_model_ref(model_id, models_dir=models_dir)
