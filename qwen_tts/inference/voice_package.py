from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file

from .lora_adapter import inject_lora_adapters, collect_lora_state_dict, load_lora_state_dict


VOICE_WEIGHTS_FILENAME = "speaker.safetensors"
VOICE_CONFIG_FILENAME = "speaker_config.json"
SPEAKER_EMBEDDING_KEY = "speaker_embedding"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)


@dataclass(frozen=True)
class VoicePackageConfig:
    schema_version: int
    speaker: str
    slot_id: int
    base_model_id: str
    tokenizer_type: str
    tts_model_type: str
    adapter_type: str
    lora_rank: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VoicePackageConfig":
        return cls(
            schema_version=int(payload.get("schemaVersion", 1)),
            speaker=str(payload["speaker"]),
            slot_id=int(payload.get("slotId", 3000)),
            base_model_id=str(payload["baseModelId"]),
            tokenizer_type=str(payload["tokenizerType"]),
            tts_model_type=str(payload["ttsModelType"]),
            adapter_type=str(payload.get("adapterType", "lora")),
            lora_rank=int(payload.get("loraRank", 16)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "speaker": self.speaker,
            "slotId": self.slot_id,
            "baseModelId": self.base_model_id,
            "tokenizerType": self.tokenizer_type,
            "ttsModelType": self.tts_model_type,
            "adapterType": self.adapter_type,
            "loraRank": self.lora_rank,
        }


@dataclass(frozen=True)
class VoicePackage:
    root_dir: Path
    config: VoicePackageConfig

    @property
    def model_dir(self) -> Path:
        return self.root_dir / "model"

    @property
    def weights_path(self) -> Path:
        return self.model_dir / VOICE_WEIGHTS_FILENAME

    @property
    def config_path(self) -> Path:
        return self.model_dir / VOICE_CONFIG_FILENAME

    def load_tensors(self) -> dict[str, torch.Tensor]:
        return load_file(str(self.weights_path), device="cpu")

    @classmethod
    def load(cls, root_dir: Path) -> "VoicePackage":
        root_dir = root_dir.resolve()
        payload = _read_json(root_dir / "model" / VOICE_CONFIG_FILENAME, default=None)
        if not isinstance(payload, dict):
            raise FileNotFoundError(f"Voice package config missing: {root_dir / 'model' / VOICE_CONFIG_FILENAME}")
        return cls(root_dir=root_dir, config=VoicePackageConfig.from_payload(payload))


def save_voice_package(
    *,
    output_dir: Path,
    speaker: str,
    base_model_id: str,
    tokenizer_type: str,
    tts_model_type: str,
    speaker_embedding: torch.Tensor,
    lora_state_dict: dict[str, torch.Tensor],
    slot_id: int = 3000,
    lora_rank: int = 16,
) -> VoicePackage:
    model_dir = _ensure_dir(output_dir.resolve() / "model")
    tensors = {
        SPEAKER_EMBEDDING_KEY: speaker_embedding.detach().cpu().clone(),
        **{key: value.detach().cpu().clone() for key, value in lora_state_dict.items()},
    }
    save_file(tensors, str(model_dir / VOICE_WEIGHTS_FILENAME))
    config = VoicePackageConfig(
        schema_version=1,
        speaker=speaker,
        slot_id=slot_id,
        base_model_id=base_model_id,
        tokenizer_type=tokenizer_type,
        tts_model_type=tts_model_type,
        adapter_type="lora",
        lora_rank=lora_rank,
    )
    _write_json_atomic(model_dir / VOICE_CONFIG_FILENAME, config.to_payload())
    return VoicePackage(root_dir=output_dir.resolve(), config=config)


@contextmanager
def activate_voice_package(model, package: VoicePackage) -> Iterator[None]:
    package_tensors = package.load_tensors()
    speaker_embedding = package_tensors.get(SPEAKER_EMBEDDING_KEY)
    if speaker_embedding is None:
        raise ValueError(f"Voice package missing `{SPEAKER_EMBEDDING_KEY}` tensor")

    inject_lora_adapters(model.talker, rank=package.config.lora_rank, alpha=package.config.lora_rank)
    previous_lora_state = collect_lora_state_dict(model.talker)
    package_lora_state = {
        key: value
        for key, value in package_tensors.items()
        if key != SPEAKER_EMBEDDING_KEY
    }

    input_embeddings = model.talker.get_input_embeddings()
    slot_id = int(package.config.slot_id)
    previous_slot = input_embeddings.weight[slot_id].detach().cpu().clone()
    previous_spk_id = dict(getattr(model.config.talker_config, "spk_id", {}) or {})
    previous_spk_is_dialect = dict(getattr(model.config.talker_config, "spk_is_dialect", {}) or {})
    previous_supported_speakers = getattr(model, "supported_speakers", None)

    with torch.no_grad():
        input_embeddings.weight[slot_id].copy_(
            speaker_embedding.to(
                device=input_embeddings.weight.device,
                dtype=input_embeddings.weight.dtype,
            )
        )
    load_lora_state_dict(model.talker, package_lora_state)
    model.config.talker_config.spk_id = dict(previous_spk_id)
    model.config.talker_config.spk_id[package.config.speaker.lower()] = slot_id
    model.config.talker_config.spk_is_dialect = dict(previous_spk_is_dialect)
    model.config.talker_config.spk_is_dialect[package.config.speaker.lower()] = False
    if previous_supported_speakers is not None:
        model.supported_speakers = model.config.talker_config.spk_id.keys()

    try:
        yield
    finally:
        with torch.no_grad():
            input_embeddings.weight[slot_id].copy_(
                previous_slot.to(
                    device=input_embeddings.weight.device,
                    dtype=input_embeddings.weight.dtype,
                )
            )
        load_lora_state_dict(model.talker, previous_lora_state)
        model.config.talker_config.spk_id = previous_spk_id
        model.config.talker_config.spk_is_dialect = previous_spk_is_dialect
        if previous_supported_speakers is not None:
            model.supported_speakers = previous_supported_speakers
