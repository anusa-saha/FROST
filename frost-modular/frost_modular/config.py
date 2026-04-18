from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    role: str = "proxy"
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    notes: str = ""
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


def load_model_zoo(path) -> Dict[str, Any]:
    zoo_path = Path(path)
    with zoo_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_model_zoo(zoo: Mapping[str, Any], path) -> None:
    zoo_path = Path(path)
    with zoo_path.open("w", encoding="utf-8") as f:
        json.dump(zoo, f, indent=2, ensure_ascii=False)


def _models_section(zoo: Mapping[str, Any]) -> Mapping[str, Any]:
    if "models" in zoo and isinstance(zoo["models"], Mapping):
        return zoo["models"]
    return zoo


def default_teacher_key(zoo: Mapping[str, Any]) -> Optional[str]:
    defaults = zoo.get("defaults", {})
    return defaults.get("teacher_key")


def default_proxy_keys(zoo: Mapping[str, Any]) -> List[str]:
    defaults = zoo.get("defaults", {})
    value = defaults.get("proxy_keys", [])
    return list(value) if isinstance(value, list) else [value]


def available_model_keys(zoo: Mapping[str, Any], role: Optional[str] = None) -> List[str]:
    models = _models_section(zoo)
    keys = []
    for key, raw in models.items():
        if not isinstance(raw, Mapping):
            continue
        if role is None or raw.get("role") == role:
            keys.append(key)
    return sorted(keys)


def resolve_model_spec(zoo: Mapping[str, Any], key: str, expected_role: Optional[str] = None) -> ModelSpec:
    models = _models_section(zoo)
    if key not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Unknown model key '{key}'. Available keys: {available}")

    raw = dict(models[key])
    role = raw.get("role", "proxy")
    if expected_role is not None and role != expected_role:
        raise ValueError(f"Model key '{key}' has role '{role}', expected '{expected_role}'")

    extra_kwargs = dict(raw.get("extra_kwargs", {}))
    for extra_key in ("load_kwargs", "kwargs"):
        value = raw.get(extra_key)
        if isinstance(value, Mapping):
            extra_kwargs.update(value)

    return ModelSpec(
        key=key,
        hf_id=raw.get("hf_id", raw.get("model_name", key)),
        role=role,
        dtype=raw.get("dtype", "float16"),
        device_map=raw.get("device_map", "auto"),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        notes=raw.get("notes", ""),
        extra_kwargs=extra_kwargs,
    )
