from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

try:
    from accelerate import Accelerator
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Fine-tuning requires `accelerate`, which is not installed by default. "
        "Install a PyTorch build that matches your machine from https://pytorch.org/get-started/locally/, "
        "then install runtime dependencies with: pip install -e \".[runtime]\""
    ) from exc

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import register_qwen3_tts_auto_classes
from qwen_tts.inference.lora_adapter import (
    collect_lora_state_dict,
    inject_lora_adapters,
    mark_only_lora_trainable,
)
from qwen_tts.inference.voice_package import save_voice_package
from qwen_tts.training.data_pipeline import read_jsonl
from qwen_tts.training.dataset import TTSDataset


def _parse_dtype(value: str) -> torch.dtype:
    normalized = (value or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _mixed_precision(dtype: str) -> str:
    normalized = (dtype or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    if normalized in {"fp16", "float16", "half"}:
        return "fp16"
    return "no"


@dataclass(frozen=True)
class SpeakerPackageTrainConfig:
    train_model_id: str
    runtime_model_id: str
    tokenizer_model_id: str
    train_jsonl: Optional[Path]
    output_dir: Path
    speaker_name: str
    device: str
    dtype: str
    flash_attn: bool
    batch_size: int
    lr: float
    num_epochs: int
    models_dir: Optional[Path]
    train_records: Optional[list[dict[str, Any]]] = None
    gradient_accumulation_steps: int = 4
    lora_rank: int = 16


@dataclass(frozen=True)
class SpeakerPackageTrainResult:
    package_dir: Path
    train_model_id: str
    runtime_model_id: str
    tokenizer_type: str
    tts_model_type: str
    slot_id: int
    adapter_type: str
    lora_rank: int


class SpeakerPackageTrainer(nn.Module):
    def __init__(self, qwen_model, initial_speaker_embedding: torch.Tensor, *, lora_rank: int):
        super().__init__()
        self.qwen_model = qwen_model
        inject_lora_adapters(self.qwen_model.talker, rank=lora_rank, alpha=lora_rank)
        self.qwen_model.requires_grad_(False)
        mark_only_lora_trainable(self.qwen_model.talker)
        self.speaker_embedding = nn.Parameter(initial_speaker_embedding.detach().clone())

    def forward(self, batch):
        model = self.qwen_model
        input_ids = batch["input_ids"]
        codec_ids = batch["codec_ids"]
        text_embedding_mask = batch["text_embedding_mask"]
        codec_embedding_mask = batch["codec_embedding_mask"]
        attention_mask = batch["attention_mask"]
        codec_0_labels = batch["codec_0_labels"]
        codec_mask = batch["codec_mask"]

        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

        speaker_embedding = self.speaker_embedding.to(
            device=input_codec_embedding.device,
            dtype=input_codec_embedding.dtype,
        )
        input_codec_embedding[:, 6, :] = speaker_embedding.unsqueeze(0).expand(input_codec_embedding.shape[0], -1)
        input_embeddings = input_text_embedding + input_codec_embedding

        for group_index in range(1, codec_ids.shape[-1]):
            codec_group_embedding = model.talker.code_predictor.get_input_embeddings()[group_index - 1](
                codec_ids[:, :, group_index]
            )
            codec_group_embedding = codec_group_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_group_embedding

        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, :-1]]
        talker_codec_ids = codec_ids[codec_mask]
        _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
        return outputs.loss + 0.3 * sub_talker_loss


def _validate_model_pair(train_model_id: str, runtime_model_id: str) -> tuple[str, str]:
    register_qwen3_tts_auto_classes()
    train_config = AutoConfig.from_pretrained(train_model_id)
    runtime_config = AutoConfig.from_pretrained(runtime_model_id)

    if getattr(train_config, "tts_model_type", None) != "base":
        raise ValueError(f"Training source model must be a Base model: {train_model_id}")
    if getattr(runtime_config, "tts_model_type", None) != "custom_voice":
        raise ValueError(f"Runtime backbone must be a CustomVoice model: {runtime_model_id}")

    if getattr(train_config, "tokenizer_type", None) != getattr(runtime_config, "tokenizer_type", None):
        raise ValueError("Training Base model and runtime CustomVoice model must use the same tokenizer type")
    if getattr(train_config, "tts_model_size", None) != getattr(runtime_config, "tts_model_size", None):
        raise ValueError("Training Base model and runtime CustomVoice model must use the same model size")

    train_talker = getattr(train_config, "talker_config", None)
    runtime_talker = getattr(runtime_config, "talker_config", None)
    if train_talker is None or runtime_talker is None:
        raise ValueError("Qwen3-TTS config is missing talker_config")

    comparable_fields = [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "num_code_groups",
    ]
    for field_name in comparable_fields:
        if getattr(train_talker, field_name, None) != getattr(runtime_talker, field_name, None):
            raise ValueError(
                f"Training Base model and runtime CustomVoice model mismatch on talker_config.{field_name}"
            )

    return str(getattr(runtime_config, "tokenizer_type")), str(getattr(runtime_config, "tts_model_type"))


def _load_initial_speaker_embedding(dataset: TTSDataset, qwen_model) -> torch.Tensor:
    if qwen_model.speaker_encoder is None:
        raise ValueError("Training source model does not expose `speaker_encoder`; use a Base model")
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")

    reference_mel = dataset[0]["ref_mel"]
    model_dtype = next(qwen_model.parameters()).dtype
    model_device = next(qwen_model.parameters()).device
    with torch.no_grad():
        embedding = qwen_model.speaker_encoder(
            reference_mel.to(device=model_device, dtype=model_dtype)
        ).detach()
    return embedding[0].to(dtype=model_dtype, device="cpu")

def train_speaker_package(
    config: SpeakerPackageTrainConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_fn: Optional[Callable[[], None]] = None,
) -> SpeakerPackageTrainResult:
    if cancel_fn is not None:
        cancel_fn()
    tokenizer_type, runtime_tts_model_type = _validate_model_pair(
        config.train_model_id,
        config.runtime_model_id,
    )
    register_qwen3_tts_auto_classes()
    train_dtype = _parse_dtype(config.dtype)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=_mixed_precision(config.dtype),
    )
    train_wrapper = Qwen3TTSModel.from_pretrained(
        config.train_model_id,
        dtype=train_dtype,
        attn_implementation="flash_attention_2" if config.flash_attn else None,
        models_dir=config.models_dir,
    )
    if cancel_fn is not None:
        cancel_fn()
    if config.train_records is not None:
        train_records = list(config.train_records)
    elif config.train_jsonl is not None:
        train_records = read_jsonl(config.train_jsonl)
    else:
        raise ValueError("Either `train_records` or `train_jsonl` is required")
    dataset_config = AutoConfig.from_pretrained(config.train_model_id)
    dataset = TTSDataset(train_records, train_wrapper.processor, dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    if cancel_fn is not None:
        cancel_fn()

    initial_speaker_embedding = _load_initial_speaker_embedding(dataset, train_wrapper.model)
    trainer = SpeakerPackageTrainer(
        train_wrapper.model,
        initial_speaker_embedding,
        lora_rank=config.lora_rank,
    )

    optimizer = AdamW(
        [parameter for parameter in trainer.parameters() if parameter.requires_grad],
        lr=config.lr,
        weight_decay=0.01,
    )
    trainer, optimizer, dataloader = accelerator.prepare(trainer, optimizer, dataloader)
    trainer.train()

    if log_fn is not None:
        log_fn(
            "Starting speaker package training "
            f"(epochs={config.num_epochs}, batch_size={config.batch_size}, lr={config.lr})"
        )

    last_loss: Optional[float] = None
    for epoch in range(config.num_epochs):
        if cancel_fn is not None:
            cancel_fn()
        for step, batch in enumerate(dataloader):
            if cancel_fn is not None:
                cancel_fn()
            with accelerator.accumulate(trainer):
                loss = trainer(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            last_loss = float(loss.detach().item())
            if log_fn is not None and step % 10 == 0:
                log_fn(f"Epoch {epoch} | Step {step} | Loss: {last_loss:.4f}")

    if cancel_fn is not None:
        cancel_fn()
    accelerator.wait_for_everyone()
    unwrapped_trainer = accelerator.unwrap_model(trainer)
    if cancel_fn is not None:
        cancel_fn()
    package = save_voice_package(
        output_dir=config.output_dir,
        speaker=config.speaker_name,
        speak_model_id=config.runtime_model_id,
        tokenizer_type=tokenizer_type,
        tts_model_type=runtime_tts_model_type,
        speaker_embedding=unwrapped_trainer.speaker_embedding.detach().cpu(),
        lora_state_dict=collect_lora_state_dict(unwrapped_trainer.qwen_model.talker),
        slot_id=3000,
        lora_rank=config.lora_rank,
    )
    if log_fn is not None:
        summary = "unknown" if last_loss is None else f"{last_loss:.4f}"
        log_fn(
            f"Exported speaker package: {package.root_dir} "
            f"(slotId={package.config.slot_id}, loraRank={package.config.lora_rank}, lastLoss={summary})"
        )
    return SpeakerPackageTrainResult(
        package_dir=package.root_dir,
        train_model_id=config.train_model_id,
        runtime_model_id=config.runtime_model_id,
        tokenizer_type=tokenizer_type,
        tts_model_type=runtime_tts_model_type,
        slot_id=package.config.slot_id,
        adapter_type=package.config.adapter_type,
        lora_rank=package.config.lora_rank,
    )
