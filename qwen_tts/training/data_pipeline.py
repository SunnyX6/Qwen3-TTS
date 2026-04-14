from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from qwen_tts import Qwen3TTSTokenizer

DEFAULT_TOKENIZER_BATCH_SIZE = 32


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def encode_training_records(
    *,
    records: list[dict[str, Any]],
    tokenizer_model_path: str,
    device: str,
    models_dir: Optional[Path],
    batch_size: int = DEFAULT_TOKENIZER_BATCH_SIZE,
    audio_sample_rate: int = 24000,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_fn: Optional[Callable[[], None]] = None,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("Tokenizer batch size must be greater than 0")
    if not records:
        raise ValueError("Training records are empty")

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
        models_dir=models_dir,
    )

    encoded_records: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []
    pending_audios: list[Any] = []

    def flush_batch() -> None:
        if not pending_records:
            return
        if cancel_fn is not None:
            cancel_fn()
        first_audio = pending_audios[0]
        if isinstance(first_audio, str):
            encode_result = tokenizer.encode(pending_audios)
        else:
            encode_result = tokenizer.encode(pending_audios, sr=audio_sample_rate)
        for audio_codes, record in zip(encode_result.audio_codes, pending_records):
            next_record = {key: value for key, value in record.items() if key != "audio"}
            next_record["audio_codes"] = audio_codes.detach().cpu().tolist()
            encoded_records.append(next_record)
        pending_records.clear()
        pending_audios.clear()
        if cancel_fn is not None:
            cancel_fn()

    for index, record in enumerate(records, start=1):
        if cancel_fn is not None:
            cancel_fn()
        pending_records.append(record)
        pending_audios.append(record["audio"])
        if len(pending_records) >= batch_size:
            flush_batch()
            if log_fn is not None:
                log_fn(f"Encoded audio codes for {min(index, len(records))}/{len(records)} samples")

    flush_batch()
    if log_fn is not None:
        log_fn(f"Prepared encoded training records: {len(encoded_records)}")
    return encoded_records


def encode_training_jsonl(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    tokenizer_model_path: str,
    device: str,
    models_dir: Optional[Path],
    batch_size: int = DEFAULT_TOKENIZER_BATCH_SIZE,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_fn: Optional[Callable[[], None]] = None,
) -> list[dict[str, Any]]:
    raw_records = read_jsonl(input_jsonl)
    if not raw_records:
        raise ValueError(f"Training jsonl is empty: {input_jsonl}")
    encoded_records = encode_training_records(
        records=raw_records,
        tokenizer_model_path=tokenizer_model_path,
        device=device,
        models_dir=models_dir,
        batch_size=batch_size,
        log_fn=log_fn,
        cancel_fn=cancel_fn,
    )
    write_jsonl(output_jsonl, encoded_records)
    if log_fn is not None:
        log_fn(f"Prepared coded training jsonl: {output_jsonl}")
    return encoded_records
