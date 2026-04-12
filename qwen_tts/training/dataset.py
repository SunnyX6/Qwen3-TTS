from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
]

MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig, lag_num: int = -1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for audio_item in items:
            if isinstance(audio_item, str):
                out.append(self._load_audio_to_np(audio_item))
            elif isinstance(audio_item, tuple) and len(audio_item) == 2 and isinstance(audio_item[0], np.ndarray):
                out.append((audio_item[0].astype(np.float32), int(audio_item[1])))
            elif isinstance(audio_item, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(audio_item)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, value: MaybeList) -> List[Any]:
        return value if isinstance(value, list) else [value]

    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        encoded = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = encoded["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        return mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)

    def load_reference_mel(self, ref_audio_path: str | Path) -> torch.Tensor:
        normalized = self._normalize_audio_inputs([str(Path(ref_audio_path).resolve())])
        wav, sr = normalized[0]
        return self.extract_mels(audio=wav, sr=sr)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = self._build_assistant_text(item["text"])
        text_ids = self._tokenize_texts(text)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        ref_audio_list = self._ensure_list(item["ref_audio"])
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]
        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        batch_size, max_tokens = len(batch), max_length

        input_ids = torch.zeros((batch_size, max_tokens, 2), dtype=torch.long)
        codec_ids = torch.zeros((batch_size, max_tokens, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
        codec_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
        attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
        codec_0_labels = torch.full((batch_size, max_tokens), -100, dtype=torch.long)

        for index, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            input_ids[index, :3, 0] = text_ids[0, :3]
            input_ids[index, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[index, 7, 0] = self.config.tts_bos_token_id
            input_ids[index, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[index, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[index, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[index, :8 + text_ids_len + codec_ids_len] = True

            input_ids[index, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[index, 8:8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[index, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[index, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[index, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[index, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[index, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[index, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[index, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            codec_embedding_mask[index, 3:8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[index, 6] = False

            codec_mask[index, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[index, :8 + text_ids_len + codec_ids_len] = True

        ref_mels = torch.cat([data["ref_mel"] for data in batch], dim=0)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }
