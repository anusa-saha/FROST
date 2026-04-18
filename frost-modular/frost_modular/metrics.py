from __future__ import annotations

import json
import re
from collections import Counter
from math import exp, log
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


ANSWER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
BLEU_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def extract_number(text):
    if text is None:
        return None
    text = str(text)
    if "####" in text:
        text = text.split("####")[-1]
    matches = ANSWER_RE.findall(text.replace("$", ""))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def gsm8k_correct(prediction, gold):
    pred = extract_number(prediction)
    truth = extract_number(gold)
    if pred is None or truth is None:
        return False
    try:
        return abs(float(pred) - float(truth)) < 1e-6
    except ValueError:
        return pred == truth


def response_length(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def strip_prompt_prefix(full_text, prompt):
    if full_text is None:
        return ""
    text = str(full_text)
    if prompt is None:
        return text
    prompt_text = str(prompt)
    if prompt_text and text.startswith(prompt_text):
        return text[len(prompt_text) :]
    return text


def bleu_tokenize(text):
    if text is None:
        return []
    return BLEU_TOKEN_RE.findall(str(text).lower())


def sentence_bleu_score(reference, hypothesis, max_order: int = 4, smooth: float = 1e-9):
    reference_tokens = bleu_tokenize(reference)
    hypothesis_tokens = bleu_tokenize(hypothesis)

    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    precisions = []
    for order in range(1, max_order + 1):
        ref_ngrams = Counter(
            tuple(reference_tokens[i : i + order])
            for i in range(max(len(reference_tokens) - order + 1, 0))
        )
        hyp_ngrams = Counter(
            tuple(hypothesis_tokens[i : i + order])
            for i in range(max(len(hypothesis_tokens) - order + 1, 0))
        )
        total = sum(hyp_ngrams.values())
        if total == 0:
            precisions.append(smooth)
            continue
        overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        precisions.append(max(overlap / total, smooth))

    geo_mean = exp(sum(log(p) for p in precisions) / max_order)
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    if hyp_len == 0:
        return 0.0
    brevity_penalty = 1.0 if hyp_len > ref_len else exp(1.0 - (ref_len / max(hyp_len, 1)))
    return float(brevity_penalty * geo_mean)


def continuation_bleu_score(prompt, reference_text, hypothesis_text, max_order: int = 4, smooth: float = 1e-9):
    reference_continuation = strip_prompt_prefix(reference_text, prompt).strip()
    hypothesis_continuation = strip_prompt_prefix(hypothesis_text, prompt).strip()
    return sentence_bleu_score(reference_continuation, hypothesis_continuation, max_order=max_order, smooth=smooth)


def roc_curve_manual(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def threshold_f1_curve(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0

    thresholds = np.unique(y_score)
    if thresholds.size == 1:
        thresholds = np.array([thresholds[0] - 1e-9, thresholds[0], thresholds[0] + 1e-9])
    else:
        thresholds = np.concatenate(([thresholds.min() - 1e-9], thresholds, [thresholds.max() + 1e-9]))

    precision_list = []
    recall_list = []
    f1_list = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    best_idx = int(np.argmax(f1_list)) if f1_list else 0
    return thresholds, np.array(precision_list), np.array(recall_list), np.array(f1_list), best_idx


def summarize_results(rows):
    if not rows:
        return {}

    teacher_correct = np.array([int(r["teacher_correct"]) for r in rows], dtype=int)
    frost_correct = np.array([int(r["frost_correct"]) for r in rows], dtype=int)
    teacher_length = np.array([r["teacher_length"] for r in rows], dtype=float)
    frost_length = np.array([r["frost_length"] for r in rows], dtype=float)
    teacher_time = np.array([r["teacher_time_sec"] for r in rows], dtype=float)
    frost_time = np.array([r["frost_time_sec"] for r in rows], dtype=float)
    energy_score = -np.array([r["frost_mean_candidate_energy"] for r in rows], dtype=float)
    bleu_values = np.array([r.get("frost_bleu_vs_teacher", np.nan) for r in rows], dtype=float)

    summary = {
        "teacher_accuracy": float(teacher_correct.mean()),
        "frost_accuracy": float(frost_correct.mean()),
        "teacher_mean_length": float(teacher_length.mean()),
        "frost_mean_length": float(frost_length.mean()),
        "teacher_mean_latency_sec": float(teacher_time.mean()),
        "frost_mean_latency_sec": float(frost_time.mean()),
        "mean_energy_score": float(energy_score.mean()),
    }
    if np.isfinite(bleu_values).any():
        summary["mean_frost_bleu_vs_teacher"] = float(np.nanmean(bleu_values))
    else:
        summary["mean_frost_bleu_vs_teacher"] = None

    roc = roc_curve_manual(frost_correct, energy_score)
    if roc is not None:
        _, _, auc = roc
        summary["frost_auc_roc"] = float(auc)
    else:
        summary["frost_auc_roc"] = None

    thresholds, precision, recall, f1, best_idx = threshold_f1_curve(frost_correct, energy_score)
    summary["best_f1"] = float(f1[best_idx]) if f1.size else None
    summary["best_f1_threshold"] = float(thresholds[best_idx]) if thresholds.size else None
    return summary


def save_results_json(rows, path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_results_json(path):
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as f:
        return json.load(f)
