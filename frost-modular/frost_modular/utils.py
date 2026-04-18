from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def ensure_dir(path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
