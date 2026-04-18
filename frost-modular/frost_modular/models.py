from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelSpec


_DTYPE_ALIASES = {
    "auto": None,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_torch_dtype(dtype_name: Optional[str]):
    if dtype_name is None:
        return None
    return _DTYPE_ALIASES.get(str(dtype_name).lower(), None)


def load_model_bundle(
    spec: ModelSpec,
    dtype_override: Optional[str] = None,
    device_map_override: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(spec.extra_kwargs)
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir

    dtype_name = dtype_override or spec.dtype
    torch_dtype = resolve_torch_dtype(dtype_name)
    device_map = device_map_override or spec.device_map

    if device_map == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
        torch_dtype = torch.float32

    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if device_map:
        load_kwargs["device_map"] = device_map
    if spec.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **load_kwargs)
    model.eval()
    return tokenizer, model
