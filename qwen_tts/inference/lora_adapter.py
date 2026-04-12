from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 16):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be greater than 0")
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        parameter_device = base.weight.device
        parameter_dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            torch.zeros(
                self.rank,
                base.in_features,
                device=parameter_device,
                dtype=parameter_dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=parameter_device,
                dtype=parameter_dtype,
            )
        )
        self.adapter_enabled = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.base(input_tensor)
        if not self.adapter_enabled:
            return output
        delta = F.linear(F.linear(input_tensor, self.lora_A), self.lora_B) * self.scaling
        return output + delta


def inject_lora_adapters(module: nn.Module, *, rank: int = 16, alpha: int = 16) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
            continue
        inject_lora_adapters(child, rank=rank, alpha=alpha)


def iter_lora_modules(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, LoRALinear]]:
    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            yield child_prefix, child
            continue
        yield from iter_lora_modules(child, prefix=child_prefix)


def collect_lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for prefix, adapter in iter_lora_modules(module):
        state[f"{prefix}.lora_A"] = adapter.lora_A.detach().cpu().clone()
        state[f"{prefix}.lora_B"] = adapter.lora_B.detach().cpu().clone()
    return state


def load_lora_state_dict(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    for prefix, adapter in iter_lora_modules(module):
        key_a = f"{prefix}.lora_A"
        key_b = f"{prefix}.lora_B"
        source_a = state_dict.get(key_a)
        source_b = state_dict.get(key_b)
        if source_a is not None:
            adapter.lora_A.data.copy_(source_a.to(device=adapter.lora_A.device, dtype=adapter.lora_A.dtype))
        else:
            adapter.lora_A.data.zero_()
        if source_b is not None:
            adapter.lora_B.data.copy_(source_b.to(device=adapter.lora_B.device, dtype=adapter.lora_B.dtype))
        else:
            adapter.lora_B.data.zero_()


def zero_lora_state(module: nn.Module) -> None:
    for _, adapter in iter_lora_modules(module):
        adapter.lora_A.data.zero_()
        adapter.lora_B.data.zero_()


def mark_only_lora_trainable(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    for _, adapter in iter_lora_modules(module):
        adapter.lora_A.requires_grad = True
        adapter.lora_B.requires_grad = True


def collect_trainable_lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, adapter in iter_lora_modules(module):
        params.append(adapter.lora_A)
        params.append(adapter.lora_B)
    return params
