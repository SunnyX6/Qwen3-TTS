from __future__ import annotations

import json
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


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


def _make_voice_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"voice_{stamp}_{datetime.now().microsecond:06d}"


@dataclass(frozen=True)
class VoiceRecord:
    voice_id: str
    speaker: str
    train_model_id: Optional[str]
    speak_model_id: str
    path: str
    enabled: bool
    created_at: str
    source_task_id: Optional[str] = None
    tokenizer_type: Optional[str] = None
    tts_model_type: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VoiceRecord":
        return cls(
            voice_id=str(payload["voiceId"]),
            speaker=str(payload["speaker"]),
            train_model_id=(
                str(payload.get("trainModelId")).strip()
                if payload.get("trainModelId") is not None
                else None
            ),
            speak_model_id=str(payload["speakModelId"]),
            path=str(payload["path"]),
            enabled=bool(payload.get("enabled", True)),
            created_at=str(payload["createdAt"]),
            source_task_id=payload.get("sourceTaskId"),
            tokenizer_type=payload.get("tokenizerType"),
            tts_model_type=payload.get("ttsModelType"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "voiceId": payload["voice_id"],
            "speaker": payload["speaker"],
            "trainModelId": payload["train_model_id"],
            "speakModelId": payload["speak_model_id"],
            "path": payload["path"],
            "enabled": payload["enabled"],
            "createdAt": payload["created_at"],
            "sourceTaskId": payload["source_task_id"],
            "tokenizerType": payload["tokenizer_type"],
            "ttsModelType": payload["tts_model_type"],
        }


class VoiceRegistry:
    def __init__(self, voices_dir: Path):
        self.voices_dir = _ensure_dir(voices_dir.resolve())
        self.index_path = self.voices_dir / "index.json"
        self._lock = threading.Lock()
        if not self.index_path.exists():
            _write_json_atomic(self.index_path, {"schemaVersion": 1, "voices": []})

    def list(self, *, speak_model_id: Optional[str] = None, enabled_only: bool = True) -> list[VoiceRecord]:
        with self._lock:
            index = self._load_index_locked()
        out: list[VoiceRecord] = []
        for payload in index["voices"]:
            record = VoiceRecord.from_payload(payload)
            if enabled_only and not record.enabled:
                continue
            if speak_model_id is not None and record.speak_model_id != speak_model_id:
                continue
            out.append(record)
        return out

    def find_by_voice_id(self, voice_id: str) -> Optional[VoiceRecord]:
        voice_id = voice_id.strip()
        if not voice_id:
            return None
        with self._lock:
            return self._find_by_voice_id_locked(voice_id)

    def find_by_speaker(self, speaker: str, *, speak_model_id: Optional[str] = None) -> Optional[VoiceRecord]:
        normalized = speaker.strip().lower()
        if not normalized:
            return None
        with self._lock:
            return self._find_by_speaker_locked(normalized, speak_model_id=speak_model_id)

    def require_speaker(self, speaker: str, *, speak_model_id: Optional[str] = None) -> VoiceRecord:
        record = self.find_by_speaker(speaker, speak_model_id=speak_model_id)
        if record is None:
            raise ValueError(f"Unknown custom speaker: {speaker}")
        return record

    def assert_speaker_available(
        self,
        speaker: str,
        *,
        speak_model_id: Optional[str] = None,
        builtin_speakers: Optional[Iterable[str]] = None,
    ) -> None:
        normalized = speaker.strip().lower()
        if not normalized:
            raise ValueError("Speaker name is required")
        if builtin_speakers is not None:
            builtin_set = {str(item).strip().lower() for item in builtin_speakers if str(item).strip()}
            if normalized in builtin_set:
                raise ValueError(f"Speaker name already conflicts with built-in speaker: {speaker}")
        existing = self.find_by_speaker(speaker, speak_model_id=speak_model_id)
        if existing is not None:
            raise ValueError(f"Speaker name already exists: {speaker}")

    def register(
        self,
        *,
        package_dir: Path,
        speaker: str,
        train_model_id: Optional[str],
        speak_model_id: str,
        source_task_id: Optional[str],
        tokenizer_type: Optional[str],
        tts_model_type: Optional[str],
        builtin_speakers: Optional[Iterable[str]] = None,
    ) -> VoiceRecord:
        package_dir = package_dir.resolve()
        if not package_dir.exists():
            raise FileNotFoundError(f"Voice package directory not found: {package_dir}")

        with self._lock:
            normalized = speaker.strip().lower()
            if not normalized:
                raise ValueError("Speaker name is required")
            if builtin_speakers is not None:
                builtin_set = {str(item).strip().lower() for item in builtin_speakers if str(item).strip()}
                if normalized in builtin_set:
                    raise ValueError(f"Speaker name already conflicts with built-in speaker: {speaker}")
            if self._find_by_speaker_locked(normalized, speak_model_id=speak_model_id) is not None:
                raise ValueError(f"Speaker name already exists: {speaker}")
            voice_id = _make_voice_id()
            target_dir = self.voices_dir / voice_id
            shutil.copytree(package_dir, target_dir)
            record = VoiceRecord(
                voice_id=voice_id,
                speaker=speaker,
                train_model_id=(str(train_model_id).strip() if train_model_id is not None else None),
                speak_model_id=speak_model_id,
                path=str(target_dir),
                enabled=True,
                created_at=datetime.now().isoformat(),
                source_task_id=source_task_id,
                tokenizer_type=tokenizer_type,
                tts_model_type=tts_model_type,
            )
            meta_payload = {
                "voiceId": record.voice_id,
                "speaker": record.speaker,
                "trainModelId": record.train_model_id,
                "speakModelId": record.speak_model_id,
                "path": record.path,
                "enabled": record.enabled,
                "createdAt": record.created_at,
                "sourceTaskId": record.source_task_id,
                "tokenizerType": record.tokenizer_type,
                "ttsModelType": record.tts_model_type,
            }
            _write_json_atomic(target_dir / "meta.json", meta_payload)

            index = self._load_index_locked()
            index["voices"].append(record.to_payload())
            _write_json_atomic(self.index_path, index)
            return record

    def delete(self, voice_id: str) -> VoiceRecord:
        voice_id = voice_id.strip()
        if not voice_id:
            raise ValueError("voiceId is required")

        with self._lock:
            index = self._load_index_locked()
            remaining: list[dict[str, Any]] = []
            removed: Optional[VoiceRecord] = None
            for payload in index["voices"]:
                record = VoiceRecord.from_payload(payload)
                if record.voice_id == voice_id:
                    removed = record
                    continue
                remaining.append(payload)
            if removed is None:
                raise FileNotFoundError(f"Unknown voiceId: {voice_id}")
            index["voices"] = remaining
            _write_json_atomic(self.index_path, index)

        voice_dir = Path(removed.path)
        if voice_dir.exists():
            shutil.rmtree(voice_dir, ignore_errors=True)
        return removed

    def _load_index_locked(self) -> dict[str, Any]:
        payload = _read_json(self.index_path, default=None)
        if not isinstance(payload, dict):
            payload = {"schemaVersion": 1, "voices": []}
        voices = payload.get("voices")
        if not isinstance(voices, list):
            payload["voices"] = []
        return payload

    def _find_by_voice_id_locked(self, voice_id: str) -> Optional[VoiceRecord]:
        index = self._load_index_locked()
        for payload in index["voices"]:
            record = VoiceRecord.from_payload(payload)
            if record.voice_id == voice_id:
                return record
        return None

    def _find_by_speaker_locked(self, normalized_speaker: str, *, speak_model_id: Optional[str]) -> Optional[VoiceRecord]:
        index = self._load_index_locked()
        for payload in index["voices"]:
            record = VoiceRecord.from_payload(payload)
            if not record.enabled:
                continue
            if speak_model_id is not None and record.speak_model_id != speak_model_id:
                continue
            if record.speaker.strip().lower() == normalized_speaker:
                return record
        return None
