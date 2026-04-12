from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

ASR_AUDIO_SAMPLE_RATE = 16000
DEFAULT_ASR_MODEL_SIZE = "large-v3"
ASR_LOCAL_ROOT_NAME = "asr"
FASTER_WHISPER_LOCAL_ROOT_NAME = "faster-whisper"
FASTER_WHISPER_CACHE_ROOT_NAME = "faster-whisper-cache"
FUNASR_LOCAL_ROOT_NAME = "funasr"
SUPPORTED_ASR_MODEL_SIZES = {
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
}
SUPPORTED_TRANSCRIBE_PROVIDERS = {"auto", "faster-whisper", "funasr"}
_LANGUAGE_ALIASES = {
    "auto": "auto",
    "chinese": "zh",
    "zh": "zh",
    "cantonese": "yue",
    "yue": "yue",
    "english": "en",
    "en": "en",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
}
_LANGUAGE_DISPLAY_NAMES = {
    "auto": "Auto",
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
}
FUNASR_ZH_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
FUNASR_ZH_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
FUNASR_ZH_PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
FUNASR_YUE_MODEL_ID = "iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"


def normalize_transcribe_language(language: str) -> str:
    normalized = str(language or "Auto").strip().lower()
    if not normalized:
        return "auto"
    resolved = _LANGUAGE_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            "Unsupported `language`. Expected one of: Auto, Chinese, Cantonese, English, Japanese, Korean"
        )
    return resolved


def normalize_transcribe_provider(provider: str) -> str:
    normalized = str(provider or "auto").strip().lower()
    if normalized not in SUPPORTED_TRANSCRIBE_PROVIDERS:
        raise ValueError("Unsupported `provider`. Expected one of: auto, faster-whisper, funasr")
    return normalized


def normalize_asr_model_size(model_size: str) -> str:
    normalized = str(model_size or DEFAULT_ASR_MODEL_SIZE).strip()
    if normalized not in SUPPORTED_ASR_MODEL_SIZES:
        raise ValueError(
            "Unsupported `modelSize`. Expected one of: "
            + ", ".join(sorted(SUPPORTED_ASR_MODEL_SIZES))
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
        self._funasr_models: Dict[tuple[str, str], Any] = {}

    @property
    def asr_models_dir(self) -> Path:
        return self.models_dir / ASR_LOCAL_ROOT_NAME

    @property
    def faster_whisper_models_dir(self) -> Path:
        return self.asr_models_dir / FASTER_WHISPER_LOCAL_ROOT_NAME

    @property
    def faster_whisper_cache_dir(self) -> Path:
        return self.asr_models_dir / FASTER_WHISPER_CACHE_ROOT_NAME

    @property
    def funasr_models_dir(self) -> Path:
        return self.asr_models_dir / FUNASR_LOCAL_ROOT_NAME

    def clear(self) -> None:
        with self._lock:
            self._faster_whisper_models.clear()
            self._funasr_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str,
        provider: str,
        model_size: str,
    ) -> Dict[str, Any]:
        normalized_language = normalize_transcribe_language(language)
        normalized_provider = normalize_transcribe_provider(provider)
        normalized_audio = np.asarray(audio, dtype=np.float32)

        if normalized_provider == "funasr":
            return self._transcribe_with_funasr(normalized_audio, language_code=normalized_language)

        if normalized_provider == "faster-whisper":
            return self._transcribe_with_faster_whisper(
                normalized_audio,
                language_code=normalized_language,
                model_size=model_size,
            )

        if normalized_language in {"zh", "yue"}:
            return self._transcribe_with_funasr(normalized_audio, language_code=normalized_language)

        if normalized_language != "auto":
            return self._transcribe_with_faster_whisper(
                normalized_audio,
                language_code=normalized_language,
                model_size=model_size,
            )

        faster_result = self._transcribe_with_faster_whisper(
            normalized_audio,
            language_code="auto",
            model_size=model_size,
        )
        detected_language_code = str(faster_result.get("languageCode") or "").strip().lower()
        if detected_language_code in {"zh", "yue"}:
            return self._transcribe_with_funasr(normalized_audio, language_code=detected_language_code)
        return faster_result

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

    def _transcribe_with_funasr(self, audio: np.ndarray, *, language_code: str) -> Dict[str, Any]:
        if language_code not in {"zh", "yue"}:
            raise ValueError("FunASR currently only supports Chinese or Cantonese in this API")
        model = self._get_funasr_model(language_code=language_code)
        result = model.generate(input=audio)
        if not result:
            raise RuntimeError("FunASR returned an empty result")
        text = str(result[0].get("text") or "").strip()
        return {
            "text": text,
            "languageCode": language_code,
            "languageDetected": display_transcribe_language(language_code),
            "providerUsed": "funasr",
        }

    def _get_faster_whisper_model(self, *, model_size: str):
        normalized_model_size = normalize_asr_model_size(model_size)
        device, device_index, compute_type = self._faster_whisper_runtime()
        cache_key = (normalized_model_size, f"{device}:{device_index}", compute_type)
        with self._lock:
            cached = self._faster_whisper_models.get(cache_key)
            if cached is not None:
                return cached
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Missing dependency `faster-whisper`. Install with: pip install -e \".[api,asr]\""
                ) from exc

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
                    "or allow network access so upstream can download it into "
                    f"`{self.faster_whisper_cache_dir}`. Original error: {type(exc).__name__}: {exc}"
                ) from exc
            self._faster_whisper_models = {cache_key: model}
            return model

    def _get_funasr_model(self, *, language_code: str):
        device = self._funasr_runtime_device()
        cache_key = (language_code, device)
        with self._lock:
            cached = self._funasr_models.get(cache_key)
            if cached is not None:
                return cached
            try:
                from funasr import AutoModel
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Missing dependency `funasr`. Install with: pip install -e \".[api,asr]\""
                ) from exc

            try:
                if language_code == "zh":
                    model = AutoModel(
                        model=self._resolve_funasr_model_ref(FUNASR_ZH_MODEL_ID),
                        vad_model=self._resolve_funasr_model_ref(FUNASR_ZH_VAD_MODEL_ID),
                        punc_model=self._resolve_funasr_model_ref(FUNASR_ZH_PUNC_MODEL_ID),
                        device=device,
                    )
                elif language_code == "yue":
                    model = AutoModel(
                        model=self._resolve_funasr_model_ref(FUNASR_YUE_MODEL_ID),
                        device=device,
                    )
                else:
                    raise ValueError(f"Unsupported FunASR language: {language_code}")
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load FunASR model assets. Pre-download them under "
                    f"`{self.funasr_models_dir}` or allow upstream hub download. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            self._funasr_models = {cache_key: model}
            return model

    def _local_faster_whisper_model_dir(self, model_size: str) -> Path:
        return self.faster_whisper_models_dir / model_size

    def _local_funasr_model_dir(self, model_id: str) -> Path:
        repo_name = str(model_id).split("/", 1)[-1]
        return self.funasr_models_dir / repo_name

    def _resolve_faster_whisper_model_ref(self, model_size: str) -> str:
        local_dir = self._local_faster_whisper_model_dir(model_size)
        if local_dir.is_dir():
            return str(local_dir)
        return model_size

    def _resolve_funasr_model_ref(self, model_id: str) -> str:
        local_dir = self._local_funasr_model_dir(model_id)
        if local_dir.is_dir():
            return str(local_dir)
        return model_id

    def _faster_whisper_runtime(self) -> tuple[str, int, str]:
        if self.device_mode == "CUDA":
            normalized_device = str(self.device).strip().lower()
            if normalized_device.startswith("cuda:"):
                return "cuda", int(normalized_device.split(":", 1)[1]), "float16"
            return "cuda", 0, "float16"
        return "cpu", 0, "int8"

    def _funasr_runtime_device(self) -> str:
        if self.device_mode == "CUDA":
            return self.device
        return "cpu"
