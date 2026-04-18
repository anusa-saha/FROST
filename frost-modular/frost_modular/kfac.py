from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .utils import model_device, stable_seed


@dataclass
class KFACConfig:
    rank: int = 64
    damping: float = 1e-3
    layer_limit: int = 16


@dataclass
class KFACLayerState:
    name: str
    module: nn.Linear
    input_proj: torch.Tensor
    output_proj: torch.Tensor
    a_cov: torch.Tensor
    g_cov: torch.Tensor
    a_inv: torch.Tensor
    g_inv: torch.Tensor
    a_count: int = 0
    g_count: int = 0
    cached_a_proj: Optional[torch.Tensor] = None


@dataclass
class ProxyKFACState:
    name: str
    model: nn.Module
    tokenizer: Any
    config: KFACConfig = field(default_factory=KFACConfig)
    layers: Dict[str, KFACLayerState] = field(default_factory=dict)
    handles: List[Any] = field(default_factory=list)
    calibrating: bool = False


def encode_prompt_response(tokenizer, prompt, response, device, max_length=512):
    full_text = prompt + response
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    prompt_len = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"].shape[-1]
    labels = enc["input_ids"].clone()
    if prompt_len < labels.shape[-1]:
        labels[:, :prompt_len] = -100
    labels[enc["attention_mask"] == 0] = -100
    return enc, labels, prompt_len


def select_linear_modules(model, limit=16):
    linear_items = []
    for name, module in model.named_modules():
        if not name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name or "embed_tokens" in name:
            continue
        linear_items.append((name, module))
    if limit is not None and limit > 0:
        linear_items = linear_items[-limit:]
    return linear_items


def build_projection(in_dim, rank, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    proj = torch.randn(in_dim, rank, generator=generator, device=device, dtype=torch.float32)
    return proj / math.sqrt(rank)


def build_proxy_state(name, model, tokenizer, config: Optional[KFACConfig] = None):
    config = config or KFACConfig()
    layers = {}
    for layer_name, module in select_linear_modules(model, limit=config.layer_limit):
        device = module.weight.device
        in_rank = max(1, min(config.rank, module.in_features))
        out_rank = max(1, min(config.rank, module.out_features))
        seed = stable_seed(f"{name}:{layer_name}")
        input_proj = build_projection(module.in_features, in_rank, device, seed)
        output_proj = build_projection(module.out_features, out_rank, device, seed + 1)
        a_cov = torch.zeros(in_rank, in_rank, device=device, dtype=torch.float32)
        g_cov = torch.zeros(out_rank, out_rank, device=device, dtype=torch.float32)
        eye_a = torch.eye(in_rank, device=device, dtype=torch.float32)
        eye_g = torch.eye(out_rank, device=device, dtype=torch.float32)
        layers[layer_name] = KFACLayerState(
            name=layer_name,
            module=module,
            input_proj=input_proj,
            output_proj=output_proj,
            a_cov=a_cov,
            g_cov=g_cov,
            a_inv=eye_a.clone(),
            g_inv=eye_g.clone(),
        )
    return ProxyKFACState(name=name, model=model, tokenizer=tokenizer, config=config, layers=layers)


def register_kfac_hooks(proxy: ProxyKFACState):
    for state in proxy.layers.values():
        module = state.module

        def forward_hook(mod, inputs, output, state=state, proxy=proxy):
            if not proxy.calibrating or not inputs:
                return
            activations = inputs[0].detach()
            activations = activations.reshape(-1, activations.shape[-1])
            activations = activations.to(device=state.input_proj.device, dtype=torch.float32)
            state.cached_a_proj = activations @ state.input_proj

        def backward_hook(mod, grad_input, grad_output, state=state, proxy=proxy):
            if not proxy.calibrating or state.cached_a_proj is None or not grad_output:
                return
            gradients = grad_output[0].detach()
            gradients = gradients.reshape(-1, gradients.shape[-1])
            gradients = gradients.to(device=state.output_proj.device, dtype=torch.float32)
            g_proj = gradients @ state.output_proj
            state.a_cov += state.cached_a_proj.T @ state.cached_a_proj
            state.g_cov += g_proj.T @ g_proj
            state.a_count += int(state.cached_a_proj.shape[0])
            state.g_count += int(g_proj.shape[0])
            state.cached_a_proj = None

        proxy.handles.append(module.register_forward_hook(forward_hook))
        proxy.handles.append(module.register_full_backward_hook(backward_hook))


def clear_proxy_hooks(proxy: ProxyKFACState):
    for handle in proxy.handles:
        handle.remove()
    proxy.handles.clear()


def safe_inverse(matrix, damping):
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    damped = matrix + damping * eye
    try:
        return torch.linalg.inv(damped)
    except RuntimeError:
        return torch.linalg.pinv(damped)


def finalize_proxy(proxy: ProxyKFACState):
    for state in proxy.layers.values():
        a_count = max(1, state.a_count)
        g_count = max(1, state.g_count)
        a_cov = state.a_cov / a_count
        g_cov = state.g_cov / g_count
        state.a_inv = safe_inverse(a_cov, proxy.config.damping)
        state.g_inv = safe_inverse(g_cov, proxy.config.damping)


def calibrate_proxy(proxy: ProxyKFACState, calibration_pairs, max_samples=8, max_length=512):
    device = model_device(proxy.model)
    proxy.model.eval()
    proxy.calibrating = True
    proxy.model.zero_grad(set_to_none=True)
    for prompt, response in calibration_pairs[:max_samples]:
        enc, labels, _ = encode_prompt_response(proxy.tokenizer, prompt, response, device, max_length=max_length)
        if (labels != -100).sum().item() == 0:
            continue
        outputs = proxy.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss.float()
        proxy.model.zero_grad(set_to_none=True)
        loss.backward()
    proxy.calibrating = False
    finalize_proxy(proxy)


def score_proxy_candidate(proxy: ProxyKFACState, prompt: str, full_candidate_text: str, prompt_len: Optional[int] = None, max_length: int = 512) -> float:
    device = model_device(proxy.model)
    enc = proxy.tokenizer(
        full_candidate_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    if prompt_len is None:
        prompt_len = proxy.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].shape[-1]
    labels = enc["input_ids"].clone()
    if prompt_len < labels.shape[-1]:
        labels[:, :prompt_len] = -100
    labels[enc["attention_mask"] == 0] = -100
    if (labels != -100).sum().item() == 0:
        return 0.0

    proxy.model.zero_grad(set_to_none=True)
    outputs = proxy.model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=labels,
    )
    loss = outputs.loss.float()
    loss.backward()

    energy = 0.0
    with torch.no_grad():
        for state in proxy.layers.values():
            grad = state.module.weight.grad
            if grad is None:
                continue
            grad = grad.detach().float()
            dW_proj = state.output_proj.T @ grad @ state.input_proj
            layer_energy = torch.trace(state.g_inv @ dW_proj @ state.a_inv @ dW_proj.T)
            energy += float(layer_energy.item())
            if state.module.bias is not None and state.module.bias.grad is not None:
                bias_grad = state.module.bias.grad.detach().float()
                bias_proj = bias_grad @ state.output_proj
                energy += float(torch.dot(bias_proj, state.g_inv @ bias_proj).item())

    proxy.model.zero_grad(set_to_none=True)
    return float(energy)


def score_proxy_candidates(proxy: ProxyKFACState, prompt: str, candidate_texts: Sequence[str], prompt_len: Optional[int] = None, max_length: int = 512):
    return [
        score_proxy_candidate(proxy, prompt, text, prompt_len=prompt_len, max_length=max_length)
        for text in candidate_texts
    ]
