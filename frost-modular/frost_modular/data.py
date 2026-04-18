from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def load_gsm8k_dataset(split: str = "train", limit: int | None = None):
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main")[split]
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def calibration_pairs_from_examples(examples, limit: int):
    if limit <= 0:
        return []
    pairs = []
    for example in examples.select(range(min(limit, len(examples)))):
        pairs.append((build_prompt(example["question"]), example["answer"]))
    return pairs
