from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from tqdm import tqdm

from .data import build_prompt
from .decoding import DecodeConfig, frost_generate, teacher_generate
from .metrics import gsm8k_correct, response_length, save_results_json, summarize_results


def take_examples(examples, limit: int | None):
    if limit is None:
        return examples
    try:
        return examples.select(range(min(limit, len(examples))))
    except AttributeError:
        return list(examples)[:limit]


def run_gsm8k_evaluation(
    teacher_model,
    teacher_tokenizer,
    proxy_states,
    examples,
    decode_config: DecodeConfig,
    eval_limit: int | None = None,
    beta_values: Sequence[float] | None = None,
    sample: bool = False,
):
    eval_examples = take_examples(examples, eval_limit)
    beta_values = list(beta_values) if beta_values else [decode_config.beta]
    results: List[Dict[str, Any]] = []

    for example in tqdm(eval_examples):
        prompt = build_prompt(example["question"])
        teacher_out, teacher_stats = teacher_generate(
            teacher_model,
            teacher_tokenizer,
            prompt,
            decode_config,
            sample=sample,
            return_stats=True,
        )

        for beta in beta_values:
            beta_config = DecodeConfig(
                max_sequence_length=decode_config.max_sequence_length,
                max_new_tokens=decode_config.max_new_tokens,
                shortlist_k=decode_config.shortlist_k,
                beta=float(beta),
                sample=sample,
                lambda_geom=decode_config.lambda_geom,
                lambda_struct=decode_config.lambda_struct,
                lambda_prov=decode_config.lambda_prov,
            )
            frost_out, frost_stats = frost_generate(
                teacher_model,
                teacher_tokenizer,
                proxy_states,
                prompt,
                beta_config,
                sample=sample,
                return_stats=True,
            )

            row = {
                "question": example["question"],
                "ground_truth": example["answer"],
                "beta": float(beta),
                "top_k": int(decode_config.shortlist_k),
                "teacher": teacher_out,
                "frost": frost_out,
                "teacher_correct": bool(gsm8k_correct(teacher_out, example["answer"])),
                "frost_correct": bool(gsm8k_correct(frost_out, example["answer"])),
                "teacher_length": int(response_length(teacher_out, teacher_tokenizer)),
                "frost_length": int(response_length(frost_out, teacher_tokenizer)),
                "teacher_time_sec": float(teacher_stats["elapsed_sec"]),
                "frost_time_sec": float(frost_stats["elapsed_sec"]),
                "frost_mean_candidate_energy": float(frost_stats["mean_candidate_energy"]),
                "frost_mean_chosen_energy": float(frost_stats["mean_chosen_energy"]),
                "frost_num_steps": int(frost_stats["num_steps"]),
                "frost_step_records": frost_stats["step_records"],
            }
            results.append(row)
            print(
                f"beta={beta:.4f} | teacher_correct={row['teacher_correct']} | "
                f"frost_correct={row['frost_correct']}"
            )

        print("Teacher:", teacher_out)
        print("-" * 80)

    return results, summarize_results(results)


def save_evaluation_results(rows, path) -> None:
    save_results_json(rows, path)
