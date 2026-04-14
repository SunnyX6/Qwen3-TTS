from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from runtime.catalog import (
    BASE_MODEL_IDS,
    CUSTOM_VOICE_MODEL_IDS,
    VOICE_DESIGN_MODEL_IDS,
    require_model_ref,
    require_supported_model_id,
    resolve_model_ref,
    resolve_train_pair,
)
from qwen_tts.inference.voice_registry import VoiceRegistry

ASR_AUDIO_SAMPLE_RATE = 16000
DEFAULT_ASR_MODEL_SIZE = "large-v3"
DEFAULT_GENERATION_SEED = 0
DEFAULT_GENERATION_MAX_NEW_TOKENS = 2048
DEFAULT_GENERATION_TEMPERATURE = 0.9
DEFAULT_GENERATION_TOP_P = 1.0
DEFAULT_GENERATION_REPETITION_PENALTY = 1.05
ASR_LOCAL_ROOT_NAME = "asr"
FASTER_WHISPER_LOCAL_ROOT_NAME = "faster-whisper"
FASTER_WHISPER_CACHE_ROOT_NAME = "faster-whisper-cache"
SUPPORTED_ASR_MODEL_SIZES = {
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
}
INTERNAL_ASR_MODEL_SIZES = SUPPORTED_ASR_MODEL_SIZES | {"medium.en"}
TRAINING_AUDIO_SAMPLE_RATE = 24000
_STDOUT_LOCK = threading.Lock()

_QWEN_LANGUAGE_ALIASES = {
    "auto": {"auto", "自动"},
    "chinese": {"chinese", "中文", "汉语", "普通话", "mandarin"},
    "english": {"english", "英文", "英语"},
    "japanese": {"japanese", "日语", "日文"},
    "korean": {"korean", "韩语", "韩文"},
    "german": {"german", "德语", "德文"},
    "french": {"french", "法语", "法文"},
    "russian": {"russian", "俄语", "俄文"},
    "portuguese": {"portuguese", "葡萄牙语", "葡语"},
    "spanish": {"spanish", "西班牙语", "西语"},
    "italian": {"italian", "意大利语", "意语"},
    "beijing_dialect": {
        "beijing_dialect",
        "beijing dialect",
        "北京话",
        "北京方言",
        "北京腔",
        "京片子",
    },
    "sichuan_dialect": {
        "sichuan_dialect",
        "sichuan dialect",
        "四川话",
        "四川方言",
        "川话",
        "成都话",
    },
}
_QWEN_CANONICAL_TO_ASR_CODE = {
    "auto": "auto",
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "german": "de",
    "french": "fr",
    "russian": "ru",
    "portuguese": "pt",
    "spanish": "es",
    "italian": "it",
    "beijing_dialect": "zh",
    "sichuan_dialect": "zh",
}
_LANGUAGE_ALIASES: Dict[str, str] = {}
for canonical_name, aliases in _QWEN_LANGUAGE_ALIASES.items():
    mapped_code = _QWEN_CANONICAL_TO_ASR_CODE[canonical_name]
    _LANGUAGE_ALIASES[canonical_name] = mapped_code
    for alias in aliases:
        _LANGUAGE_ALIASES[str(alias).strip().lower()] = mapped_code
_LANGUAGE_DISPLAY_NAMES = {
    "auto": "Auto",
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}
_LANGUAGE_OPTIONS_HINT = (
    "Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, "
    "Beijing_Dialect, Sichuan_Dialect"
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _write_stdout_line(message: str) -> None:
    line = message.rstrip() + "\n"
    with _STDOUT_LOCK:
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
            sys.stdout.write(safe_line)
        sys.stdout.flush()


def _wav_bytes_from_array(wav, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    return buffer.getvalue()


def _set_generation_seed(seed: Optional[int]) -> int:
    import torch

    resolved_seed = DEFAULT_GENERATION_SEED if seed is None else int(seed)
    np.random.seed(resolved_seed % (2**32))
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    return resolved_seed


def _normalize_demo_audio(wav, eps: float = 1e-12, clip: bool = True):
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


def _load_demo_audio_bytes(audio_bytes: bytes, *, target_sample_rate: Optional[int] = None):
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
    normalized_wav = _normalize_demo_audio(wav)
    resolved_sample_rate = int(sample_rate)
    if target_sample_rate is not None and resolved_sample_rate != int(target_sample_rate):
        normalized_wav = librosa.resample(
            y=normalized_wav,
            orig_sr=resolved_sample_rate,
            target_sr=int(target_sample_rate),
        ).astype("float32")
        resolved_sample_rate = int(target_sample_rate)
    return _normalize_demo_audio(normalized_wav), resolved_sample_rate


@dataclass
class AudioResult:
    wav: Any
    sample_rate: int
    response_format: str
    extra: Dict[str, Any]


@dataclass
class PreparedAudioResponse:
    response_format: str
    audio_bytes: Optional[bytes] = None
    json_payload: Optional[Dict[str, Any]] = None


def normalize_transcribe_language(language: str) -> str:
    normalized = str(language or "Auto").strip().lower()
    if not normalized:
        return "auto"
    resolved = _LANGUAGE_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported `language`. Expected one of: {_LANGUAGE_OPTIONS_HINT}")
    return resolved


def normalize_asr_model_size(model_size: str) -> str:
    normalized = str(model_size or DEFAULT_ASR_MODEL_SIZE).strip()
    if normalized not in SUPPORTED_ASR_MODEL_SIZES:
        raise ValueError(
            "Unsupported `modelSize`. Expected one of: "
            + ", ".join(sorted(SUPPORTED_ASR_MODEL_SIZES))
        )
    return normalized


def resolve_internal_asr_model_size(model_size: str, language_code: str) -> str:
    normalized_size = normalize_asr_model_size(model_size)
    normalized_language = str(language_code or "").strip().lower()
    if normalized_size == "medium" and normalized_language == "en":
        return "medium.en"
    return normalized_size


def normalize_internal_asr_model_size(model_size: str) -> str:
    normalized = str(model_size or DEFAULT_ASR_MODEL_SIZE).strip()
    if normalized not in INTERNAL_ASR_MODEL_SIZES:
        raise ValueError(
            "Unsupported internal `modelSize`. Expected one of: "
            + ", ".join(sorted(INTERNAL_ASR_MODEL_SIZES))
        )
    return normalized


def display_transcribe_language(language_code: Optional[str]) -> Optional[str]:
    if language_code is None:
        return None
    normalized = str(language_code).strip().lower()
    if not normalized:
        return None
    return _LANGUAGE_DISPLAY_NAMES.get(normalized, normalized.upper())


class AsrManager:
    def __init__(self, *, device: str, device_mode: str, dtype: str, models_dir: Path):
        self.device = device
        self.device_mode = device_mode
        self.dtype = dtype
        self.models_dir = models_dir
        self._lock = threading.Lock()
        self._faster_whisper_models: Dict[tuple[str, str, str], Any] = {}

    @property
    def asr_models_dir(self) -> Path:
        return self.models_dir / ASR_LOCAL_ROOT_NAME

    @property
    def faster_whisper_models_dir(self) -> Path:
        return self.asr_models_dir / FASTER_WHISPER_LOCAL_ROOT_NAME

    @property
    def faster_whisper_cache_dir(self) -> Path:
        return self.asr_models_dir / FASTER_WHISPER_CACHE_ROOT_NAME

    def clear(self) -> None:
        with self._lock:
            self._faster_whisper_models.clear()

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str,
        model_size: str,
    ) -> Dict[str, Any]:
        normalized_language = normalize_transcribe_language(language)
        internal_model_size = resolve_internal_asr_model_size(model_size, normalized_language)
        normalized_audio = np.asarray(audio, dtype=np.float32)
        return self._transcribe_with_faster_whisper(
            normalized_audio,
            language_code="auto" if normalized_language == "auto" else normalized_language,
            model_size=internal_model_size,
        )

    def validate_transcribe_request(self, *, language: str, model_size: str) -> Dict[str, str]:
        normalized_language = normalize_transcribe_language(language)
        internal_model_size = resolve_internal_asr_model_size(model_size, normalized_language)
        self._require_faster_whisper_dependency()
        self._resolve_faster_whisper_model_ref(internal_model_size)
        return {
            "languageCode": normalized_language,
            "internalModelSize": internal_model_size,
        }

    def _transcribe_with_faster_whisper(
        self,
        audio: np.ndarray,
        *,
        language_code: str,
        model_size: str,
    ) -> Dict[str, Any]:
        model = self._get_faster_whisper_model(model_size=model_size)
        requested_language = None if language_code == "auto" else language_code
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 700},
            language=requested_language,
        )
        text = "".join(segment.text for segment in segments).strip()
        detected_language_code = str(getattr(info, "language", None) or requested_language or language_code or "").strip().lower()
        if not detected_language_code:
            detected_language_code = "auto"
        return {
            "text": text,
            "languageCode": detected_language_code,
            "languageDetected": display_transcribe_language(detected_language_code),
            "providerUsed": "faster-whisper",
        }

    def _get_faster_whisper_model(self, *, model_size: str):
        normalized_model_size = normalize_internal_asr_model_size(model_size)
        device, device_index, compute_type = self._faster_whisper_runtime()
        cache_key = (normalized_model_size, f"{device}:{device_index}", compute_type)
        with self._lock:
            cached = self._faster_whisper_models.get(cache_key)
            if cached is not None:
                return cached
            WhisperModel = self._require_faster_whisper_dependency()
            model_ref = self._resolve_faster_whisper_model_ref(normalized_model_size)
            try:
                model = WhisperModel(
                    model_ref,
                    device=device,
                    device_index=device_index,
                    compute_type=compute_type,
                    download_root=str(self.faster_whisper_cache_dir),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load faster-whisper model "
                    f"`{normalized_model_size}`. Pre-download it to "
                    f"`{self._local_faster_whisper_model_dir(normalized_model_size)}` "
                    f"before starting API. Original error: {type(exc).__name__}: {exc}"
                ) from exc
            self._faster_whisper_models = {cache_key: model}
            return model

    @staticmethod
    def _require_faster_whisper_dependency():
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency `faster-whisper`. Install with: pip install -e \".[api,asr]\""
            ) from exc
        return WhisperModel

    def _local_faster_whisper_model_dir(self, model_size: str) -> Path:
        return self.faster_whisper_models_dir / model_size

    def _resolve_faster_whisper_model_ref(self, model_size: str) -> str:
        local_dir = self._local_faster_whisper_model_dir(model_size)
        if local_dir.is_dir():
            return str(local_dir)
        raise FileNotFoundError(
            "Missing local faster-whisper model directory: "
            f"`{local_dir}`. Download it before calling /api/translate."
        )

    def _faster_whisper_runtime(self) -> tuple[str, int, str]:
        if self.device_mode == "CUDA":
            normalized_device = str(self.device).strip().lower()
            if normalized_device.startswith("cuda:"):
                return "cuda", int(normalized_device.split(":", 1)[1]), "float16"
            return "cuda", 0, "float16"
        return "cpu", 0, "int8"


def load_model_config_payload(model_ref: str) -> dict[str, Any]:
    config_path = Path(model_ref) / "config.json"
    payload = _read_json(config_path, default=None)
    return payload if isinstance(payload, dict) else {}


def resolve_train_model_pair(
    model_id: str,
    models_dir: Path,
    *,
    expected_runtime_model_id: Optional[str] = None,
) -> tuple[str, str]:
    train_model_ref, runtime_model_ref = resolve_train_pair(model_id, models_dir)
    train_config = load_model_config_payload(train_model_ref)
    runtime_config = load_model_config_payload(runtime_model_ref)
    if train_config and str(train_config.get("tts_model_type", "")).strip().lower() != "base":
        raise ValueError(f"Training source model must resolve to a Base model: {train_model_ref}")
    if runtime_config and str(runtime_config.get("tts_model_type", "")).strip().lower() != "custom_voice":
        raise ValueError(f"Runtime backbone must resolve to a CustomVoice model: {runtime_model_ref}")
    if expected_runtime_model_id is not None:
        resolved_expected_runtime_model_id = resolve_model_ref(expected_runtime_model_id, models_dir)
        if runtime_model_ref != resolved_expected_runtime_model_id:
            raise ValueError(
                "Training request does not match the configured shared CustomVoice backbone: "
                f"expected {resolved_expected_runtime_model_id}, got {runtime_model_ref}"
            )
    return train_model_ref, runtime_model_ref


def resolve_public_base_model_id(model_id: str, models_dir: Path) -> str:
    normalized_model_id = str(model_id or "").strip()
    if normalized_model_id in BASE_MODEL_IDS:
        return resolve_model_ref(normalized_model_id, models_dir)
    if normalized_model_id in CUSTOM_VOICE_MODEL_IDS:
        pair_index = CUSTOM_VOICE_MODEL_IDS.index(normalized_model_id)
        return resolve_model_ref(BASE_MODEL_IDS[pair_index], models_dir)
    return resolve_model_ref(normalized_model_id, models_dir)


def extract_builtin_speakers_from_config_payload(config_payload: dict[str, Any]) -> List[str]:
    talker_config = config_payload.get("talker_config")
    if not isinstance(talker_config, dict):
        return []
    speaker_map = talker_config.get("spk_id")
    if not isinstance(speaker_map, dict):
        return []
    return sorted(
        {str(name).strip() for name in speaker_map.keys() if str(name).strip()},
        key=str.lower,
    )


def extract_supported_dialects_from_config_payload(config_payload: dict[str, Any]) -> List[str]:
    talker_config = config_payload.get("talker_config")
    if not isinstance(talker_config, dict):
        return []
    codec_language_id = talker_config.get("codec_language_id")
    if not isinstance(codec_language_id, dict):
        return []
    return sorted(
        {
            str(name).strip()
            for name in codec_language_id.keys()
            if str(name).strip() and str(name).strip().lower().endswith("_dialect")
        },
        key=str.lower,
    )


def extract_native_dialect_map_from_config_payload(config_payload: dict[str, Any]) -> Dict[str, Optional[str]]:
    talker_config = config_payload.get("talker_config")
    if not isinstance(talker_config, dict):
        return {}
    speaker_map = talker_config.get("spk_id")
    dialect_map = talker_config.get("spk_is_dialect")
    if not isinstance(speaker_map, dict) or not isinstance(dialect_map, dict):
        return {}
    out: Dict[str, Optional[str]] = {}
    for speaker in speaker_map.keys():
        speaker_name = str(speaker).strip()
        if not speaker_name:
            continue
        dialect_value = dialect_map.get(speaker_name.lower())
        if isinstance(dialect_value, str) and dialect_value.strip():
            out[speaker_name.lower()] = dialect_value.strip()
        else:
            out[speaker_name.lower()] = None
    return out


def get_custom_voice_capabilities(
    models_dir: Path,
    runtime_model_id: str,
) -> tuple[str, List[str], Dict[str, Optional[str]]]:
    resolved_runtime_model_id = resolve_model_ref(runtime_model_id, models_dir)
    config_payload = load_model_config_payload(resolved_runtime_model_id)
    supported_dialects = extract_supported_dialects_from_config_payload(config_payload)
    native_dialect_map = extract_native_dialect_map_from_config_payload(config_payload)
    return resolved_runtime_model_id, supported_dialects, native_dialect_map


def get_builtin_speakers_for_base_model(models_dir: Path, base_model_id: str) -> List[str]:
    resolved_model_ref = resolve_model_ref(base_model_id, models_dir)
    return extract_builtin_speakers_from_config_payload(load_model_config_payload(resolved_model_ref))


def normalize_voice_key(payload: Dict[str, Any]) -> str:
    return str(payload.get("speaker") or "").strip().lower()


def format_builtin_speaker_display_name(speaker: str) -> str:
    normalized = str(speaker).strip()
    if not normalized:
        return normalized
    return "_".join(part[:1].upper() + part[1:] for part in normalized.split("_"))


def list_available_voices(
    *,
    voice_registry: VoiceRegistry,
    models_dir: Path,
    runtime_model_id: str,
) -> List[Dict[str, Any]]:
    resolved_runtime_model_id, supported_dialects, native_dialect_map = get_custom_voice_capabilities(
        models_dir,
        runtime_model_id,
    )
    merged: Dict[str, Dict[str, Any]] = {}
    for speaker in get_builtin_speakers_for_base_model(models_dir, resolved_runtime_model_id):
        key = str(speaker).strip().lower()
        if key:
            merged[key] = {
                "voiceId": None,
                "speaker": format_builtin_speaker_display_name(speaker),
                "baseModelId": resolved_runtime_model_id,
                "enabled": True,
                "supportedDialects": supported_dialects,
                "nativeDialect": native_dialect_map.get(speaker.lower()),
                "source": "builtin",
                "deletable": False,
            }
    for record in voice_registry.list(speak_model_id=resolved_runtime_model_id):
        meta = _read_json(Path(record.path) / "meta.json", default=None)
        if meta:
            payload = {
                "voiceId": record.voice_id,
                "speaker": str(meta.get("speaker") or record.speaker or "").strip(),
                "baseModelId": str(meta.get("speakModelId") or record.speak_model_id).strip(),
                "enabled": bool(meta.get("enabled", True)),
                "createdAt": str(meta.get("createdAt") or record.created_at),
                "supportedDialects": supported_dialects,
                "nativeDialect": None,
                "source": "custom",
                "deletable": True,
            }
            key = normalize_voice_key(payload)
            if key:
                merged[key] = payload
    custom_voices = [item for item in merged.values() if item.get("source") != "builtin"]
    builtin_voices = [item for item in merged.values() if item.get("source") == "builtin"]
    custom_voices.sort(
        key=lambda item: (
            str(item.get("createdAt") or ""),
            str(item.get("speaker") or "").lower(),
        ),
        reverse=True,
    )
    builtin_voices.sort(key=lambda item: str(item.get("speaker") or "").lower())
    return custom_voices + builtin_voices


class TaskRunner:
    def __init__(self, state: Any):
        self.state = state

    @staticmethod
    def prepare_audio_response(result: AudioResult) -> PreparedAudioResponse:
        wav = result.wav
        sample_rate = int(result.sample_rate)
        response_format = str(result.response_format)
        extra = dict(result.extra)
        try:
            audio_bytes = _wav_bytes_from_array(wav, sample_rate)
            if response_format == "wav":
                return PreparedAudioResponse(
                    response_format=response_format,
                    audio_bytes=audio_bytes,
                )
            payload = {
                "ok": True,
                "sampleRate": sample_rate,
                "audioBase64": base64.b64encode(audio_bytes).decode("utf-8"),
            }
            payload.update(extra)
            return PreparedAudioResponse(
                response_format=response_format,
                json_payload=payload,
            )
        finally:
            result.wav = None
            result.extra = {}

    @staticmethod
    def serialize_audio_response(result: PreparedAudioResponse) -> Dict[str, Any]:
        return {
            "response_format": result.response_format,
            "audio_bytes": result.audio_bytes,
            "json_payload": dict(result.json_payload) if result.json_payload is not None else None,
        }

    @staticmethod
    def _resolve_requested_language(payload: Dict[str, Any]) -> str:
        language = payload.get("language", "Auto")
        language_value = str(language).strip()
        return language_value or "Auto"

    @staticmethod
    def _extract_generation_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
        max_new_tokens = payload.get("max_new_tokens")
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        repetition_penalty = payload.get("repetition_penalty")
        return {
            "max_new_tokens": int(DEFAULT_GENERATION_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens),
            "temperature": float(DEFAULT_GENERATION_TEMPERATURE if temperature is None else temperature),
            "top_p": float(DEFAULT_GENERATION_TOP_P if top_p is None else top_p),
            "repetition_penalty": float(
                DEFAULT_GENERATION_REPETITION_PENALTY if repetition_penalty is None else repetition_penalty
            ),
        }

    @staticmethod
    def _resolve_generation_seed(payload: Dict[str, Any]) -> int:
        seed = payload.get("seed")
        return DEFAULT_GENERATION_SEED if seed is None else int(seed)

    @staticmethod
    def _post_process_audio(wav, sample_rate: int):
        return wav

    def _seeded_model_call(self, seed: int, fn):
        _set_generation_seed(seed)
        return fn()

    def _build_audio_result(self, wav, sample_rate: int, response_format: str, extra: Dict[str, Any]) -> AudioResult:
        return AudioResult(
            wav=self._post_process_audio(wav, sample_rate),
            sample_rate=sample_rate,
            response_format=response_format,
            extra=extra,
        )

    def _require_model(
        self,
        model_id: str,
        *,
        supported_model_ids: tuple[str, ...],
        field_name: str = "modelId",
    ) -> tuple[str, str]:
        normalized_model_id = require_supported_model_id(
            model_id,
            supported_model_ids=supported_model_ids,
            field_name=field_name,
        )
        resolved_model_ref = require_model_ref(
            normalized_model_id,
            models_dir=self.state.config.models_dir,
            field_name=field_name,
        )
        return normalized_model_id, resolved_model_ref

    def transcribe(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        audio_payloads = payload.get("audios")
        if not isinstance(audio_payloads, list) or not audio_payloads:
            raise ValueError("`audios` must be a non-empty list")
        requested_language_raw = str(payload.get("language", "Auto") or "Auto")
        requested_language_code = normalize_transcribe_language(requested_language_raw)
        requested_language = display_transcribe_language(requested_language_code) or "Auto"
        raw_model_size = str(payload.get("modelSize") or DEFAULT_ASR_MODEL_SIZE).strip()
        requested_model_size = normalize_asr_model_size(raw_model_size)
        self.state.asr.validate_transcribe_request(
            language=requested_language_raw,
            model_size=requested_model_size,
        )

        def _run_batch() -> list[Dict[str, Any]]:
            results: list[Dict[str, Any]] = []
            for index, audio_payload in enumerate(audio_payloads):
                filename = str(audio_payload.get("audioFilename") or f"audio_{index}.wav").strip()
                try:
                    audio_bytes = audio_payload.get("audioBytes")
                    if not audio_bytes:
                        raise ValueError("Uploaded audio file is empty")
                    audio, _ = _load_demo_audio_bytes(
                        audio_bytes,
                        target_sample_rate=ASR_AUDIO_SAMPLE_RATE,
                    )
                    transcribe_result = self.state.asr.transcribe(
                        audio,
                        language=requested_language_raw,
                        model_size=requested_model_size,
                    )
                    results.append(
                        {
                            "index": index,
                            "fileName": filename,
                            "ok": True,
                            "text": transcribe_result.get("text") or "",
                            "languageDetected": transcribe_result.get("languageDetected"),
                            "languageCode": transcribe_result.get("languageCode"),
                            "providerUsed": transcribe_result.get("providerUsed"),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "index": index,
                            "fileName": filename,
                            "ok": False,
                            "text": "",
                            "languageDetected": None,
                            "languageCode": None,
                            "providerUsed": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            return results

        results = _run_batch()
        success_count = sum(1 for item in results if item.get("ok"))
        failed_count = len(results) - success_count
        return {
            "ok": failed_count == 0,
            "languageRequested": requested_language,
            "modelSize": requested_model_size,
            "total": len(results),
            "successCount": success_count,
            "failedCount": failed_count,
            "results": results,
        }

    def voice_design(self, payload: Dict[str, Any]) -> PreparedAudioResponse:
        model_id, model_ref = self._require_model(
            payload["modelId"],
            supported_model_ids=VOICE_DESIGN_MODEL_IDS,
            field_name="modelId",
        )
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)
        requested_language = self._resolve_requested_language(payload)
        wavs, sample_rate = self._seeded_model_call(
            generation_seed,
            lambda: self.state.models.get(model_ref).generate_voice_design(
                text=payload["text"],
                language=requested_language,
                instruct=payload["instruct"],
                **generation_kwargs,
            ),
        )
        return self.prepare_audio_response(
            self._build_audio_result(
                wav=wavs[0],
                sample_rate=sample_rate,
                response_format=payload.get("responseFormat", "base64"),
                extra={"modelId": model_id},
            )
        )

    def clone(self, payload: Dict[str, Any]) -> PreparedAudioResponse:
        model_id, model_ref = self._require_model(
            payload["modelId"],
            supported_model_ids=BASE_MODEL_IDS,
            field_name="modelId",
        )
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)
        requested_language = self._resolve_requested_language(payload)
        ref_audio = _load_demo_audio_bytes(payload["refAudioBytes"])
        wavs, sample_rate = self._seeded_model_call(
            generation_seed,
            lambda: self.state.models.get(model_ref).generate_voice_clone(
                text=payload["text"].strip(),
                language=requested_language,
                ref_audio=ref_audio,
                ref_text=(payload.get("refText") or "").strip() or None,
                x_vector_only_mode=bool(payload.get("xVectorOnlyMode", False)),
                **generation_kwargs,
            ),
        )
        return self.prepare_audio_response(
            self._build_audio_result(
                wav=wavs[0],
                sample_rate=sample_rate,
                response_format=payload.get("responseFormat", "base64"),
                extra={"modelId": model_id},
            )
        )

    def custom_voice(self, payload: Dict[str, Any]) -> PreparedAudioResponse:
        requested_speaker = (payload.get("speaker") or "").strip()
        if not requested_speaker:
            raise ValueError("`speaker` is required")
        runtime_model_id, runtime_model_ref = self._require_model(
            str(payload.get("modelId") or self.state.config.custom_voice_model_id).strip(),
            supported_model_ids=CUSTOM_VOICE_MODEL_IDS,
            field_name="modelId",
        )
        generation_seed = self._resolve_generation_seed(payload)
        generation_kwargs = self._extract_generation_kwargs(payload)
        requested_language = self._resolve_requested_language(payload)
        requested_dialect = payload.get("dialect")

        def _run_custom_voice():
            def _generate():
                model = self.state.models.get(runtime_model_ref)
                speaker = self._resolve_custom_voice(model, requested_speaker)
                wavs, sample_rate = model.generate_custom_voice(
                    text=payload["text"],
                    language=requested_language,
                    dialect=requested_dialect,
                    speaker=speaker,
                    instruct=payload.get("instruct"),
                    **generation_kwargs,
                )
                return wavs, sample_rate, speaker

            return self._seeded_model_call(generation_seed, _generate)

        wavs, sample_rate, speaker = _run_custom_voice()
        return self.prepare_audio_response(
            self._build_audio_result(
                wav=wavs[0],
                sample_rate=sample_rate,
                response_format=payload.get("responseFormat", "base64"),
                extra={"modelId": runtime_model_id, "speaker": speaker},
            )
        )

    @staticmethod
    def _resolve_custom_voice(model: Any, requested_speaker: Optional[str]) -> str:
        supported = model.get_supported_speakers() or []
        if requested_speaker:
            return requested_speaker
        if len(supported) == 1:
            return supported[0]
        raise ValueError("`speaker` is required for models that expose multiple speakers")


def update_task_meta(current_meta: Optional[Dict[str, Any]], updates: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(current_meta or {})
    meta.update(updates)
    return meta


def write_train_log(task_id: str, speaker_name: str, message: str) -> None:
    _write_stdout_line(f"[train][{task_id}][{speaker_name}] {message}")


class TrainCanceledError(RuntimeError):
    pass


def _raise_if_train_canceled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TrainCanceledError("Canceled by request")


def prepare_training_records(
    payload: Dict[str, Any],
    *,
    check_cancel: Optional[Callable[[], None]] = None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    logs: List[str] = []
    if check_cancel is not None:
        check_cancel()
    ref_audio, ref_sample_rate = _load_demo_audio_bytes(
        payload["refAudioBytes"],
        target_sample_rate=TRAINING_AUDIO_SAMPLE_RATE,
    )
    if ref_sample_rate != TRAINING_AUDIO_SAMPLE_RATE:
        raise ValueError(f"Reference audio must resolve to {TRAINING_AUDIO_SAMPLE_RATE}Hz")
    logs.append(
        "Prepared refAudio in memory "
        f"({payload.get('refAudioFilename') or 'refAudio'} -> {ref_sample_rate}Hz)"
    )
    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(payload["samples"], start=1):
        if check_cancel is not None:
            check_cancel()
        sample_audio, sample_rate = _load_demo_audio_bytes(
            sample["audioBytes"],
            target_sample_rate=TRAINING_AUDIO_SAMPLE_RATE,
        )
        if sample_rate != TRAINING_AUDIO_SAMPLE_RATE:
            raise ValueError(f"Sample audio must resolve to {TRAINING_AUDIO_SAMPLE_RATE}Hz")
        records.append(
            {
                "audio": sample_audio,
                "text": sample["text"],
                "ref_audio": (ref_audio, ref_sample_rate),
            }
        )
        logs.append(
            "Prepared sample audio in memory "
            f"(#{index} {sample.get('audioFilename') or 'sample'} -> {sample_rate}Hz)"
        )
        if check_cancel is not None:
            check_cancel()
    return records, logs


def register_trained_voice(
    state: Any,
    *,
    task_id: str,
    package_dir: Path,
    speaker_name: str,
    train_model_id: str,
    runtime_model_id: str,
    tokenizer_type: str = "12hz",
    tts_model_type: str = "custom_voice",
    slot_id: int = 3000,
    adapter_type: str = "lora",
    lora_rank: int = 16,
) -> dict[str, Any]:
    builtin_speakers = get_builtin_speakers_for_base_model(state.config.models_dir, runtime_model_id)
    record = state.voice_registry.register(
        package_dir=package_dir,
        speaker=speaker_name,
        train_model_id=train_model_id,
        speak_model_id=runtime_model_id,
        source_task_id=task_id,
        tokenizer_type=tokenizer_type,
        tts_model_type=tts_model_type,
        builtin_speakers=builtin_speakers,
    )
    meta_path = Path(record.path) / "meta.json"
    meta = {
        "schemaVersion": 1,
        "voiceId": record.voice_id,
        "speaker": record.speaker,
        "trainModelId": train_model_id,
        "speakModelId": runtime_model_id,
        "tokenizerType": tokenizer_type,
        "ttsModelType": tts_model_type,
        "slotId": int(slot_id),
        "adapterType": str(adapter_type or "lora"),
        "loraRank": int(lora_rank),
        "createdAt": record.created_at,
        "sourceTaskId": record.source_task_id,
        "enabled": True,
    }
    _write_json(meta_path, meta)
    return {
        "voiceId": record.voice_id,
        "speaker": record.speaker,
        "trainModelId": train_model_id,
        "speakModelId": runtime_model_id,
        "createdAt": record.created_at,
        "sourceTaskId": record.source_task_id,
    }


def run_train_task(
    state: Any,
    task_id: str,
    payload: Dict[str, Any],
    *,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    speaker_name = payload["speakerName"]
    log = lambda message: write_train_log(task_id, speaker_name, message)

    def check_cancel() -> None:
        _raise_if_train_canceled(cancel_event)

    try:
        from qwen_tts.training import SpeakerPackageTrainConfig, encode_training_records, train_speaker_package

        check_cancel()
        resolved_train_model_id, resolved_runtime_model_id = resolve_train_model_pair(
            payload["modelId"],
            state.config.models_dir,
        )
        resolved_tokenizer_model_id = resolve_model_ref(payload["tokenizerModelId"], state.config.models_dir)
        running_meta = update_task_meta(
            state.tasks.get(task_id),
            {
                "taskId": task_id,
                "requestId": task_id,
                "status": "running",
                "speaker": speaker_name,
                "trainModelId": resolved_train_model_id,
                "speakModelId": resolved_runtime_model_id,
                "updatedAt": datetime.now().isoformat(),
                "language": payload.get("language", "Auto"),
            },
        )
        state.tasks.set(task_id, running_meta)
        log(f"Training source model: {resolved_train_model_id}")
        log(f"Runtime CustomVoice backbone: {resolved_runtime_model_id}")
        log(f"Tokenizer model: {resolved_tokenizer_model_id}")
        check_cancel()
        train_records, preparation_logs = prepare_training_records(payload, check_cancel=check_cancel)
        for message in preparation_logs:
            log(message)
        check_cancel()
        encoded_records = encode_training_records(
            records=train_records,
            tokenizer_model_path=resolved_tokenizer_model_id,
            device=state.config.device,
            models_dir=state.config.models_dir,
            audio_sample_rate=TRAINING_AUDIO_SAMPLE_RATE,
            log_fn=log,
            cancel_fn=check_cancel,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"qwen3_tts_train_{task_id}_",
            dir=state.train_tmp_root_dir(),
        ) as temp_dir:
            train_output_dir = _ensure_dir(Path(temp_dir) / "export")
            check_cancel()
            train_result = train_speaker_package(
                SpeakerPackageTrainConfig(
                    train_model_id=resolved_train_model_id,
                    runtime_model_id=resolved_runtime_model_id,
                    tokenizer_model_id=resolved_tokenizer_model_id,
                    train_jsonl=None,
                    train_records=encoded_records,
                    output_dir=train_output_dir,
                    speaker_name=speaker_name,
                    device=state.config.device,
                    dtype=state.config.dtype,
                    flash_attn=state.config.flash_attn,
                    batch_size=int(payload.get("batchSize", 8)),
                    lr=float(payload.get("lr", 2e-6)),
                    num_epochs=int(payload.get("numEpochs", 3)),
                    models_dir=state.config.models_dir,
                ),
                log_fn=log,
                cancel_fn=check_cancel,
            )
            check_cancel()
            registered_voice = register_trained_voice(
                state,
                task_id=task_id,
                package_dir=train_result.package_dir,
                speaker_name=speaker_name,
                train_model_id=train_result.train_model_id,
                runtime_model_id=train_result.runtime_model_id,
                tokenizer_type=train_result.tokenizer_type,
                tts_model_type=train_result.tts_model_type,
                slot_id=train_result.slot_id,
                adapter_type=train_result.adapter_type,
                lora_rank=train_result.lora_rank,
            )
        registered_meta = update_task_meta(
            state.tasks.get(task_id),
            {
                "status": "completed",
                **registered_voice,
                "updatedAt": datetime.now().isoformat(),
            },
        )
        state.tasks.set(task_id, registered_meta)
    except TrainCanceledError as exc:
        log(f"[CANCELED] {exc}")
        canceled_meta = update_task_meta(
            state.tasks.get(task_id),
            {
                "taskId": task_id,
                "requestId": task_id,
                "status": "canceled",
                "voiceId": None,
                "error": None,
                "updatedAt": datetime.now().isoformat(),
            },
        )
        state.tasks.set(task_id, canceled_meta)
    except Exception as exc:
        log(f"[ERROR] {type(exc).__name__}: {exc}")
        failed_meta = update_task_meta(
            state.tasks.get(task_id),
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updatedAt": datetime.now().isoformat(),
            },
        )
        state.tasks.set(task_id, failed_meta)
