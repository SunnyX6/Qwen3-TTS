from __future__ import annotations

from pathlib import Path
from typing import Iterable

BASE_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)
CUSTOM_VOICE_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
)
TRAIN_SUPPORTED_MODEL_IDS: tuple[str, ...] = BASE_MODEL_IDS + CUSTOM_VOICE_MODEL_IDS
VOICE_DESIGN_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)


def _supported_ids_text(model_ids: Iterable[str]) -> str:
    return ", ".join(model_ids)


def require_supported_model_id(
    model_id: str,
    *,
    supported_model_ids: tuple[str, ...],
    field_name: str = "modelId",
) -> str:
    normalized = str(model_id or "").strip()
    if not normalized:
        raise ValueError(f"`{field_name}` is required")
    if normalized not in supported_model_ids:
        raise ValueError(
            f"Unsupported `{field_name}`: {normalized}. "
            f"Use one of: {_supported_ids_text(supported_model_ids)}"
        )
    return normalized


def _model_leaf_name(model_id: str) -> str:
    normalized = str(model_id or "").strip()
    return normalized.split("/")[-1].split("\\")[-1]


def require_model_ref(
    model_id: str,
    *,
    models_dir: Path,
    field_name: str = "modelId",
) -> str:
    normalized_model_id = str(model_id or "").strip()
    model_dir = (models_dir / _model_leaf_name(normalized_model_id)).resolve()
    config_path = model_dir / "config.json"
    if not model_dir.exists() or not model_dir.is_dir():
        raise ValueError(
            f"`{field_name}` model directory does not exist: {model_dir}. "
            f"Please download `{normalized_model_id}` into `{models_dir}` first."
        )
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(
            f"`{field_name}` is missing `config.json`: {config_path}. "
            f"Please ensure `{normalized_model_id}` is downloaded completely."
        )
    return str(model_dir)


def resolve_model_ref(model_id: str, models_dir: Path) -> str:
    from qwen_tts.path_utils import resolve_pretrained_model_ref

    return resolve_pretrained_model_ref(model_id, models_dir=models_dir)


def resolve_train_pair(model_id: str, models_dir: Path) -> tuple[str, str]:
    normalized_model_id = require_supported_model_id(
        model_id,
        supported_model_ids=TRAIN_SUPPORTED_MODEL_IDS,
        field_name="modelId",
    )

    if normalized_model_id in BASE_MODEL_IDS:
        pair_index = BASE_MODEL_IDS.index(normalized_model_id)
    else:
        pair_index = CUSTOM_VOICE_MODEL_IDS.index(normalized_model_id)

    train_model_id = BASE_MODEL_IDS[pair_index]
    runtime_model_id = CUSTOM_VOICE_MODEL_IDS[pair_index]
    return (
        require_model_ref(train_model_id, models_dir=models_dir, field_name="modelId"),
        require_model_ref(runtime_model_id, models_dir=models_dir, field_name="modelId"),
    )
