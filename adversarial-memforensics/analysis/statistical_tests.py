"""
Statistical Tests Module
=========================
Shared statistical utilities for all RQ analyses.

Includes:
    - Bootstrap resampling for confidence intervals
    - Friedman test + Nemenyi post-hoc
    - Effect sizes (Cohen's d, Cliff's delta)
    - Permutation tests for Spearman's ρ
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats as scipy_stats


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a classification metric.

    Returns: (lower_bound, upper_bound)
    """
    rng = rng or np.random.default_rng(42)
    n = len(y_true)
    scores = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            scores[i] = np.nan

    scores = scores[~np.isnan(scores)]
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return (lower, upper)


def friedman_nemenyi_test(
    measurements: List[List[float]],
    group_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Friedman test for repeated measures, with Nemenyi post-hoc.

    Args:
        measurements: List of lists, each inner list = one treatment's measurements
        group_names:  Names for each treatment group
        alpha:        Significance level

    Returns:
        Dict with statistic, p_value, significant, post_hoc
    """
    if len(measurements) < 3:
        return {"error": "Need ≥ 3 groups for Friedman test"}

    try:
        stat, p = scipy_stats.friedmanchisquare(*measurements)
    except Exception as e:
        return {"error": str(e)}

    result = {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < alpha,
        "n_groups": len(measurements),
    }

    if group_names:
        result["group_names"] = group_names

    logger.info(f"Friedman: χ²={stat:.4f}, p={p:.6f}, significant={p < alpha}")
    return result


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's delta (non-parametric effect size).

    Returns: (delta, magnitude)
        magnitude: 'negligible', 'small', 'medium', 'large'
    """
    n_a, n_b = len(group_a), len(group_b)
    dominance = 0
    for a in group_a:
        for b in group_b:
            if a > b:
                dominance += 1
            elif a < b:
                dominance -= 1

    delta = dominance / (n_a * n_b)

    # Magnitude thresholds (Romano et al., 2006)
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return float(delta), magnitude


def compute_effect_size(
    group_a: np.ndarray,
    group_b: np.ndarray,
    method: str = "cohens_d",
) -> Dict[str, Any]:
    """Compute effect size with interpretation."""
    if method == "cohens_d":
        d = cohens_d(group_a, group_b)
        abs_d = abs(d)
        if abs_d < 0.2:
            interp = "negligible"
        elif abs_d < 0.5:
            interp = "small"
        elif abs_d < 0.8:
            interp = "medium"
        else:
            interp = "large"
        return {"method": "cohens_d", "value": d, "interpretation": interp}

    elif method == "cliffs_delta":
        delta, mag = cliffs_delta(group_a, group_b)
        return {"method": "cliffs_delta", "value": delta, "interpretation": mag}

    return {"error": f"Unknown method: {method}"}


def permutation_test_spearman(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Permutation test for Spearman's ρ being significantly less than 1.0.
    Tests H3: ρ < 1.0 (feature rankings are unstable).

    Returns: {observed_rho, p_value, significant, n_permutations}
    """
    rng = rng or np.random.default_rng(42)

    observed_rho, _ = scipy_stats.spearmanr(ranks_a, ranks_b)

    # Permutation distribution under the null (ρ = 1.0 → same ranking)
    perm_rhos = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_b = rng.permutation(ranks_b)
        perm_rhos[i], _ = scipy_stats.spearmanr(ranks_a, perm_b)

    # One-sided test: P(ρ_perm ≤ observed_ρ)
    p_value = np.mean(perm_rhos <= observed_rho)

    return {
        "observed_rho": float(observed_rho),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "n_permutations": n_permutations,
        "null_rho_mean": float(np.mean(perm_rhos)),
        "null_rho_std": float(np.std(perm_rhos)),
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    corrected_alpha = alpha / n

    results = []
    for i, p in enumerate(p_values):
        results.append({
            "original_p": p,
            "corrected_alpha": corrected_alpha,
            "significant_after_correction": p < corrected_alpha,
        })

    return results
