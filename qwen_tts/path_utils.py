# coding=utf-8
import os
import re
from pathlib import Path
from typing import Optional, Union


def get_default_models_dir(models_dir: Optional[Union[str, Path]] = None) -> Path:
    if models_dir is not None and str(models_dir).strip():
        return Path(models_dir).expanduser().resolve()

    env_value = os.environ.get("QWEN_TTS_MODELS_DIR")
    if env_value and env_value.strip():
        return Path(env_value).expanduser().resolve()

    return (Path.cwd() / "models").resolve()


def _leaf_name(value: str) -> str:
    parts = [part for part in re.split(r"[\\/]+", value.strip()) if part]
    return parts[-1] if parts else value.strip()


def _looks_like_local_path(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    if value.startswith((".", "~")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    if "\\" in value:
        return True

    parts = [part for part in re.split(r"[\\/]+", value) if part]
    if not parts:
        return False
    if parts[0] in {"models", "data"}:
        return True
    return len(parts) > 2


def resolve_pretrained_model_ref(
    pretrained_model_name_or_path: str,
    models_dir: Optional[Union[str, Path]] = None,
) -> str:
    value = str(pretrained_model_name_or_path or "").strip()
    if not value:
        raise ValueError("Model path or model id is required")

    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return str(raw_path.resolve())

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    models_root = get_default_models_dir(models_dir)
    models_candidate = (models_root / raw_path).resolve()
    if models_candidate.exists():
        return str(models_candidate)

    leaf_candidate = (models_root / _leaf_name(value)).resolve()
    if leaf_candidate.exists():
        return str(leaf_candidate)

    if "/" not in value and "\\" not in value:
        return str(models_candidate)

    if _looks_like_local_path(value):
        return str(cwd_candidate)

    return value
