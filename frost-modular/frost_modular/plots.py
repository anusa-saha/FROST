from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .metrics import roc_curve_manual, summarize_results, threshold_f1_curve
from .utils import ensure_dir


def plot_results(rows, output_dir="."):
    if not rows:
        print("No results to plot.")
        return {}

    output_dir = ensure_dir(output_dir)
    summary = summarize_results(rows)

    teacher_correct = np.array([int(r["teacher_correct"]) for r in rows], dtype=int)
    frost_correct = np.array([int(r["frost_correct"]) for r in rows], dtype=int)
    teacher_length = np.array([r["teacher_length"] for r in rows], dtype=float)
    frost_length = np.array([r["frost_length"] for r in rows], dtype=float)
    teacher_time = np.array([r["teacher_time_sec"] for r in rows], dtype=float)
    frost_time = np.array([r["frost_time_sec"] for r in rows], dtype=float)
    energy_score = -np.array([r["frost_mean_candidate_energy"] for r in rows], dtype=float)
    bleu_values = np.array([r.get("frost_bleu_vs_teacher", np.nan) for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = ["Teacher", "FROST"]
    accs = [float(teacher_correct.mean()), float(frost_correct.mean())]
    lengths = [float(teacher_length.mean()), float(frost_length.mean())]

    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, accs, width=0.36, label="Accuracy", color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax2 = ax.twinx()
    ax2.plot(x, lengths, marker="o", linewidth=2, color="#d95f0e", label="Mean length")
    ax2.set_ylabel("Mean response length")
    ax.set_title("Accuracy / Length tradeoff")

    ax = axes[1]
    roc = roc_curve_manual(frost_correct, energy_score)
    if roc is not None:
        fpr, tpr, auc = roc
        ax.plot(fpr, tpr, color="#31a354", linewidth=2, label=f"ROC AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("AUC-ROC on FROST correctness")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC curve unavailable\n(single class only)", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2]
    thresholds, precision, recall, f1, best_idx = threshold_f1_curve(frost_correct, energy_score)
    if thresholds.size:
        ax.plot(thresholds, precision, label="Precision", linewidth=2)
        ax.plot(thresholds, recall, label="Recall", linewidth=2)
        ax.plot(thresholds, f1, label="F1", linewidth=3)
        ax.axvline(thresholds[best_idx], linestyle="--", color="black", alpha=0.6)
        ax.set_xlabel("Energy-score threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold sweep")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Threshold curve unavailable", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_dir / "frost_kfac_metrics.png", dpi=200, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Teacher", "FROST"], [float(teacher_time.mean()), float(frost_time.mean())], color=["#2c7fb8", "#d95f0e"])
    ax.set_ylabel("Mean latency (sec)")
    ax.set_title("Latency comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "frost_kfac_latency.png", dpi=200, bbox_inches="tight")
    plt.show()

    beta_groups = {}
    for row in rows:
        beta_groups.setdefault(row["beta"], []).append(row)

    if len(beta_groups) > 1:
        betas = sorted(beta_groups)
        beta_accuracy = [float(np.mean([int(r["frost_correct"]) for r in beta_groups[b]])) for b in betas]
        beta_lengths = [float(np.mean([r["frost_length"] for r in beta_groups[b]])) for b in betas]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(betas, beta_accuracy, marker="o", linewidth=2, color="#2c7fb8", label="Accuracy")
        ax1.set_xlabel("Beta")
        ax1.set_ylabel("FROST accuracy")
        ax1.set_ylim(0.0, 1.0)
        ax2 = ax1.twinx()
        ax2.plot(betas, beta_lengths, marker="s", linewidth=2, color="#d95f0e", label="Mean length")
        ax2.set_ylabel("Mean response length")
        ax1.set_title("Beta tradeoff curve")
        fig.tight_layout()
        plt.savefig(output_dir / "frost_kfac_beta_tradeoff.png", dpi=200, bbox_inches="tight")
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(teacher_length.mean(), float(teacher_correct.mean()), s=120, label="Teacher", color="#2c7fb8")
        ax.scatter(frost_length.mean(), float(frost_correct.mean()), s=120, label="FROST", color="#d95f0e")
        ax.set_xlabel("Mean response length")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs response length")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "frost_kfac_accuracy_length.png", dpi=200, bbox_inches="tight")
        plt.show()

    if np.isfinite(bleu_values).any():
        bleu_groups = {}
        for row in rows:
            value = row.get("frost_bleu_vs_teacher")
            if value is None:
                continue
            bleu_groups.setdefault(row["beta"], []).append(float(value))

        if len(bleu_groups) > 1:
            betas = sorted(bleu_groups)
            bleu_means = [float(np.mean(bleu_groups[b])) for b in betas]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(betas, bleu_means, marker="o", linewidth=2, color="#4daf4a")
            ax.set_xlabel("Beta")
            ax.set_ylabel("Mean BLEU vs teacher")
            ax.set_ylim(0.0, 1.0)
            ax.set_title("BLEU tradeoff curve")
            plt.tight_layout()
            plt.savefig(output_dir / "frost_kfac_bleu_tradeoff.png", dpi=200, bbox_inches="tight")
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(["FROST"], [float(np.nanmean(bleu_values))], color="#4daf4a")
            ax.set_ylabel("Mean BLEU vs teacher")
            ax.set_ylim(0.0, 1.0)
            ax.set_title("BLEU similarity")
            plt.tight_layout()
            plt.savefig(output_dir / "frost_kfac_bleu.png", dpi=200, bbox_inches="tight")
            plt.show()

    print(json.dumps(summary, indent=2))
    return summary
