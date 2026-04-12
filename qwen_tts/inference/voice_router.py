from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from .voice_registry import VoiceRecord, VoiceRegistry


@dataclass(frozen=True)
class SpeakerRoute:
    kind: Literal["builtin", "custom"]
    speaker: str
    record: Optional[VoiceRecord] = None


def resolve_speaker_route(
    *,
    speaker: str,
    builtin_speakers: Iterable[str],
    voice_registry: Optional[VoiceRegistry],
    base_model_id: Optional[str],
) -> SpeakerRoute:
    normalized = speaker.strip()
    if not normalized:
        raise ValueError("Speaker is required")

    builtin_map = {str(item).strip().lower(): str(item).strip() for item in builtin_speakers if str(item).strip()}
    builtin_speaker = builtin_map.get(normalized.lower())
    if builtin_speaker is not None:
        return SpeakerRoute(kind="builtin", speaker=builtin_speaker)

    if voice_registry is None:
        raise ValueError(f"Unsupported speakers: [{speaker}]")

    record = voice_registry.require_speaker(normalized, base_model_id=base_model_id)
    return SpeakerRoute(kind="custom", speaker=record.speaker, record=record)
