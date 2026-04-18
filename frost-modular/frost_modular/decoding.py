from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .kfac import ProxyKFACState, score_proxy_candidate
from .utils import model_device, move_batch_to_device


EPS = 1e-6


@dataclass
class DecodeConfig:
    max_sequence_length: int = 512
    max_new_tokens: int = 32
    shortlist_k: int = 5
    beta: float = 0.5
    sample: bool = False
    lambda_geom: float = 1.0
    lambda_struct: float = 0.0
    lambda_prov: float = 0.0


def structural_energy(*args, **kwargs):
    return 0.0


def provenance_energy(*args, **kwargs):
    return 0.0


def teacher_generate(teacher_model, teacher_tokenizer, prompt: str, config: DecodeConfig, sample: bool = False, return_stats: bool = False):
    device = model_device(teacher_model)
    inputs = teacher_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_sequence_length,
    )
    inputs = move_batch_to_device(inputs, device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = teacher_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=sample,
            pad_token_id=teacher_tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    text = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)
    stats = {
        "elapsed_sec": float(elapsed),
        "num_tokens": int(outputs.shape[-1]),
        "generated_tokens": int(outputs.shape[-1] - inputs["input_ids"].shape[-1]),
    }
    if return_stats:
        return text, stats
    return text


def frost_generate(
    teacher_model,
    teacher_tokenizer,
    proxy_states: Sequence[ProxyKFACState],
    prompt: str,
    config: DecodeConfig,
    sample: bool = False,
    return_stats: bool = False,
):
    if not proxy_states:
        raise RuntimeError("proxy_states is empty; calibrate at least one proxy before decoding.")

    device = model_device(teacher_model)
    input_ids = teacher_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_sequence_length,
    )["input_ids"].to(device)
    decoded_text = prompt
    step_records: List[Dict[str, Any]] = []
    prompt_len_cache = {
        proxy.name: proxy.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_sequence_length,
        )["input_ids"].shape[-1]
        for proxy in proxy_states
    }
    start = time.perf_counter()

    for step in range(config.max_new_tokens):
        step_start = time.perf_counter()
        with torch.no_grad():
            logits = teacher_model(input_ids).logits[:, -1, :]

        topk = torch.topk(logits, k=config.shortlist_k)
        candidate_ids = topk.indices[0]
        candidate_texts: List[str] = []
        candidate_geom: List[float] = []
        candidate_proxy_breakdown: List[List[float]] = []

        for token_id in candidate_ids:
            candidate_input_ids = torch.cat([input_ids, token_id.view(1, 1)], dim=1)
            full_candidate_text = teacher_tokenizer.decode(candidate_input_ids[0], skip_special_tokens=True)
            candidate_texts.append(full_candidate_text)

            proxy_scores = []
            for proxy in proxy_states:
                proxy_scores.append(
                    score_proxy_candidate(
                        proxy,
                        prompt,
                        full_candidate_text,
                        prompt_len=prompt_len_cache[proxy.name],
                        max_length=config.max_sequence_length,
                    )
                )
            candidate_proxy_breakdown.append([float(score) for score in proxy_scores])
            candidate_geom.append(float(np.mean(proxy_scores)) if proxy_scores else 0.0)

        geom_tensor = torch.tensor(candidate_geom, dtype=logits.dtype, device=logits.device)
        if geom_tensor.numel() > 1 and torch.isfinite(geom_tensor).all() and geom_tensor.std() > EPS:
            geom_tensor = (geom_tensor - geom_tensor.mean()) / (geom_tensor.std() + EPS)
        else:
            geom_tensor = torch.zeros_like(geom_tensor)

        structural_tensor = torch.tensor(
            [structural_energy(prompt, text) for text in candidate_texts],
            dtype=logits.dtype,
            device=logits.device,
        )
        provenance_tensor = torch.tensor(
            [provenance_energy(prompt, text) for text in candidate_texts],
            dtype=logits.dtype,
            device=logits.device,
        )

        composite_energy = (
            config.lambda_geom * geom_tensor
            + config.lambda_struct * structural_tensor
            + config.lambda_prov * provenance_tensor
        )

        shortlist_logits = logits[0, candidate_ids] - config.beta * composite_energy
        if sample:
            shortlist_probs = torch.softmax(shortlist_logits, dim=-1)
            shortlist_choice = int(torch.multinomial(shortlist_probs, 1).item())
        else:
            shortlist_choice = int(torch.argmax(shortlist_logits, dim=-1).item())
        next_token = candidate_ids[shortlist_choice]

        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
        decoded_text = teacher_tokenizer.decode(input_ids[0], skip_special_tokens=True)

        chosen_token_id = int(next_token.item())
        step_records.append(
            {
                "step": int(step),
                "candidates": [teacher_tokenizer.decode([int(token.item())], skip_special_tokens=True) for token in candidate_ids],
                "candidate_texts": candidate_texts,
                "candidate_geom": [float(value) for value in candidate_geom],
                "candidate_proxy_breakdown": candidate_proxy_breakdown,
                "candidate_structural": [float(value) for value in structural_tensor.tolist()],
                "candidate_provenance": [float(value) for value in provenance_tensor.tolist()],
                "candidate_composite": [float(value) for value in composite_energy.tolist()],
                "chosen_token_id": chosen_token_id,
                "chosen_token_text": teacher_tokenizer.decode([chosen_token_id], skip_special_tokens=True),
                "chosen_energy": float(candidate_geom[shortlist_choice]),
                "step_time_sec": float(time.perf_counter() - step_start),
            }
        )

        if teacher_tokenizer.eos_token_id is not None and chosen_token_id == teacher_tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - start
    chosen_energies = [record["chosen_energy"] for record in step_records if record["chosen_energy"] is not None]
    diagnostics = {
        "elapsed_sec": float(elapsed),
        "num_steps": int(len(step_records)),
        "mean_step_time_sec": float(np.mean([record["step_time_sec"] for record in step_records])) if step_records else 0.0,
        "mean_candidate_energy": float(np.mean([np.mean(record["candidate_geom"]) for record in step_records])) if step_records else 0.0,
        "mean_chosen_energy": float(np.mean(chosen_energies)) if chosen_energies else 0.0,
        "step_records": step_records,
    }
    if return_stats:
        return decoded_text, diagnostics
    return decoded_text
