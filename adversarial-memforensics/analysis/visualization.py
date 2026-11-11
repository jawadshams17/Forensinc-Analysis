"""
Visualization Module
=====================
Generates all figures for the paper and analysis reports.

Plot types:
    1. Degradation curves (RQ1): accuracy/F1 vs truncation %
    2. Evasion heatmaps (RQ2): evasion rate matrix
    3. SHAP drift heatmaps (RQ3): feature-rank stability
    4. Feature importance comparison bar charts
    5. Confusion matrix heatmaps
    6. Interaction effect bar charts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "accuracy": "#2196F3",
    "f1": "#FF5722",
    "precision": "#4CAF50",
    "recall": "#9C27B0",
    "fpr": "#F44336",
    "fnr": "#FF9800",
    "clean": "#2196F3",
    "adversarial": "#F44336",
    "interaction": "#9C27B0",
}


def plot_degradation_curves(
    results: Dict[str, Any],
    output_path: str | Path,
    title: str = "RQ1: Ensemble Performance vs. Capture Truncation",
) -> None:
    """
    Plot accuracy and F1 degradation curves with confidence intervals.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    levels = results["truncation_levels"]
    metrics = results["metrics_by_level"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Extract data
    x = [m["truncation_pct"] * 100 for m in metrics]
    acc = [m["accuracy"] for m in metrics]
    f1 = [m["f1_macro"] for m in metrics]
    acc_ci_lo = [m["accuracy_ci"][0] for m in metrics]
    acc_ci_hi = [m["accuracy_ci"][1] for m in metrics]
    f1_ci_lo = [m["f1_ci"][0] for m in metrics]
    f1_ci_hi = [m["f1_ci"][1] for m in metrics]

    # Accuracy curve
    ax1.plot(x, acc, "o-", color=COLORS["accuracy"], linewidth=2.5, markersize=8)
    ax1.fill_between(x, acc_ci_lo, acc_ci_hi, alpha=0.2, color=COLORS["accuracy"])
    ax1.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax1.set_xlabel("Capture Completeness (%)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Degradation")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # F1 curve
    ax2.plot(x, f1, "s-", color=COLORS["f1"], linewidth=2.5, markersize=8)
    ax2.fill_between(x, f1_ci_lo, f1_ci_hi, alpha=0.2, color=COLORS["f1"])
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax2.set_xlabel("Capture Completeness (%)")
    ax2.set_ylabel("F1 Score (Macro)")
    ax2.set_title("F1 Score Degradation")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Degradation curve plot saved to {output_path}")


def plot_evasion_interaction(
    results: Dict[str, Any],
    output_path: str | Path,
    title: str = "RQ2: Evasion Rate & Interaction Effect",
) -> None:
    """Plot evasion rates and interaction Δ across truncation levels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    levels = results["levels"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = [l["truncation_pct"] * 100 for l in levels]
    clean_err = [l["clean_error_rate"] for l in levels]
    adv_evasion = [l["adv_evasion_rate"] for l in levels]
    interaction = [l.get("interaction_delta", 0) for l in levels]

    # Evasion rates
    ax1.plot(x, clean_err, "o-", color=COLORS["clean"], linewidth=2, label="Clean error rate")
    ax1.plot(x, adv_evasion, "s-", color=COLORS["adversarial"], linewidth=2, label="Adversarial evasion")
    ax1.fill_between(x, clean_err, adv_evasion, alpha=0.15, color=COLORS["adversarial"])
    ax1.set_xlabel("Capture Completeness (%)")
    ax1.set_ylabel("Rate")
    ax1.set_title("Evasion vs. Clean Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Interaction Δ
    bars = ax2.bar(range(len(x)), interaction, color=[
        COLORS["interaction"] if d > 0 else "gray" for d in interaction
    ], alpha=0.8)
    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels([f"{v:.0f}%" for v in x])
    ax2.set_xlabel("Capture Completeness")
    ax2.set_ylabel("Interaction Effect Δ")
    ax2.set_title("Disproportionate Amplification")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Evasion interaction plot saved to {output_path}")


def plot_shap_drift_heatmap(
    comparison_data: List[Dict],
    feature_names: List[str],
    output_path: str | Path,
    title: str = "RQ3: SHAP Feature-Importance Rank Stability",
) -> None:
    """Plot Spearman's ρ heatmap across conditions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(comparison_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = df["condition"].tolist()
    rhos = df["spearman_rho"].tolist()

    colors = ["#4CAF50" if r > 0.9 else "#FF9800" if r > 0.7 else "#F44336" for r in rhos]
    bars = ax.barh(conditions, rhos, color=colors, alpha=0.85)

    ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.5, label="Strong (ρ>0.9)")
    ax.axvline(x=0.7, color="orange", linestyle="--", alpha=0.5, label="Moderate (ρ>0.7)")

    ax.set_xlabel("Spearman's ρ (vs. clean-complete)")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")

    # Add value labels
    for bar, rho in zip(bars, rhos):
        ax.text(rho + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rho:.3f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"SHAP drift heatmap saved to {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {output_path}")
