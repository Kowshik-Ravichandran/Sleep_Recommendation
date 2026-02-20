"""
Model Evaluation Module for AI Sleep Lab

Provides publication-grade evaluation metrics including:
- Stratified K-Fold Cross-Validation
- Confidence Intervals for all metrics
- Model Comparison Tests (statistical significance)
- Effect Size Calculations

Designed for research paper standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.stats import sem, wilcoxon, friedmanchisquare

from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score, 
    cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    make_scorer
)


@dataclass
class CVResults:
    """Container for cross-validation results with statistics."""
    metric_name: str
    scores: np.ndarray
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    
    def __str__(self) -> str:
        return f"{self.metric_name}: {self.mean:.4f} ± {self.std:.4f} (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"


def compute_confidence_interval(scores: np.ndarray, 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for cross-validation scores.
    
    Uses t-distribution for small sample sizes (typical in CV).
    
    Args:
        scores: Array of CV fold scores
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(scores)
    mean = np.mean(scores)
    se = sem(scores)
    
    # t-critical value for (1 - confidence)/2 in each tail
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    
    margin = t_crit * se
    return (mean - margin, mean + margin)


def stratified_kfold_evaluate(model, X: pd.DataFrame, y: pd.Series,
                               n_splits: int = 10,
                               random_state: int = 42) -> Dict[str, CVResults]:
    """
    Perform stratified K-fold cross-validation with comprehensive metrics.
    
    Args:
        model: Sklearn-compatible model (must have fit and predict methods)
        X: Feature DataFrame
        y: Target Series
        n_splits: Number of CV folds (default 10 for publication standard)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary of metric name -> CVResults
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Run cross-validation
    cv_results = cross_validate(
        model, X, y, 
        cv=skf, 
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Process results
    results = {}
    for metric_name in scoring.keys():
        scores = cv_results[f'test_{metric_name}']
        mean = np.mean(scores)
        std = np.std(scores)
        ci_lower, ci_upper = compute_confidence_interval(scores)
        
        results[metric_name] = CVResults(
            metric_name=metric_name,
            scores=scores,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )
    
    return results


def print_cv_results_table(results: Dict[str, CVResults]) -> str:
    """
    Format CV results as a publication-ready table.
    
    Args:
        results: Dictionary of CVResults from stratified_kfold_evaluate
    
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("-" * 70)
    lines.append(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'95% CI':>20}")
    lines.append("-" * 70)
    
    for metric_name, result in results.items():
        ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        lines.append(f"{metric_name:<15} {result.mean:>10.4f} {result.std:>10.4f} {ci_str:>20}")
    
    lines.append("-" * 70)
    return "\n".join(lines)


def results_to_dataframe(results: Dict[str, CVResults]) -> pd.DataFrame:
    """
    Convert CV results to DataFrame for easy export/display.
    
    Args:
        results: Dictionary of CVResults
    
    Returns:
        DataFrame with columns: Metric, Mean, Std, CI_Lower, CI_Upper
    """
    data = []
    for metric_name, result in results.items():
        data.append({
            'Metric': metric_name.upper(),
            'Mean': result.mean,
            'Std': result.std,
            'CI_Lower': result.ci_lower,
            'CI_Upper': result.ci_upper,
            'CI_String': f"{result.mean:.4f} ± {result.std:.4f}"
        })
    
    return pd.DataFrame(data)


# ============================================================================
# MODEL COMPARISON TESTS
# ============================================================================

def wilcoxon_compare_models(scores_model1: np.ndarray, 
                            scores_model2: np.ndarray,
                            model1_name: str = "Model 1",
                            model2_name: str = "Model 2") -> Dict[str, Any]:
    """
    Compare two models using Wilcoxon signed-rank test.
    
    Appropriate when comparing CV scores from same folds (paired data).
    Non-parametric alternative to paired t-test.
    
    Args:
        scores_model1: CV scores from first model
        scores_model2: CV scores from second model
        model1_name: Name of first model
        model2_name: Name of second model
    
    Returns:
        Dictionary with test results
    """
    if len(scores_model1) != len(scores_model2):
        raise ValueError("Score arrays must have same length (from same CV folds)")
    
    statistic, p_value = wilcoxon(scores_model1, scores_model2)
    
    # Determine which model is better
    mean1 = np.mean(scores_model1)
    mean2 = np.mean(scores_model2)
    better_model = model1_name if mean1 > mean2 else model2_name
    
    return {
        'test': 'Wilcoxon Signed-Rank',
        'model1': model1_name,
        'model2': model2_name,
        'mean1': mean1,
        'mean2': mean2,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'better_model': better_model if p_value < 0.05 else 'No significant difference'
    }


def friedman_compare_models(model_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Compare multiple models using Friedman test.
    
    Non-parametric alternative to repeated-measures ANOVA.
    Use when comparing 3+ models on same CV folds.
    
    Args:
        model_scores: Dictionary of model_name -> CV scores array
    
    Returns:
        Dictionary with test results
    """
    model_names = list(model_scores.keys())
    score_arrays = [model_scores[name] for name in model_names]
    
    # Check all have same length
    lengths = [len(s) for s in score_arrays]
    if len(set(lengths)) > 1:
        raise ValueError("All score arrays must have same length")
    
    statistic, p_value = friedmanchisquare(*score_arrays)
    
    # Get rankings
    means = {name: np.mean(scores) for name, scores in model_scores.items()}
    ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'test': 'Friedman',
        'models': model_names,
        'means': means,
        'ranking': [name for name, _ in ranked],
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# ============================================================================
# EFFECT SIZE CALCULATIONS
# ============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    
    Args:
        group1: Data from first group
        group2: Data from second group
    
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
    
    Returns:
        String interpretation
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def eta_squared(groups: List[np.ndarray]) -> float:
    """
    Calculate eta-squared (η²) effect size for ANOVA.
    
    Interpretation:
    - η² < 0.01: negligible
    - 0.01 ≤ η² < 0.06: small
    - 0.06 ≤ η² < 0.14: medium
    - η² ≥ 0.14: large
    
    Args:
        groups: List of arrays, one per group
    
    Returns:
        Eta-squared value
    """
    # Flatten all data
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # Sum of squares between groups
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    
    # Total sum of squares
    ss_total = np.sum((all_data - grand_mean)**2)
    
    if ss_total == 0:
        return 0.0
    
    return ss_between / ss_total


def interpret_eta_squared(eta2: float) -> str:
    """
    Interpret eta-squared effect size.
    
    Args:
        eta2: Eta-squared value
    
    Returns:
        String interpretation
    """
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"


# ============================================================================
# MULTIPLE TESTING CORRECTION
# ============================================================================

def bonferroni_correction(p_values: List[float], 
                          alpha: float = 0.05) -> Tuple[float, List[bool]]:
    """
    Apply Bonferroni correction for multiple testing.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Original significance level
    
    Returns:
        Tuple of (corrected_alpha, list of significant flags)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in p_values]
    
    return corrected_alpha, significant


def benjamini_hochberg_fdr(p_values: List[float], 
                           alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Preferred for exploratory analysis (less conservative than Bonferroni).
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired FDR level
    
    Returns:
        Tuple of (adjusted p-values, list of significant flags)
    """
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]
    
    # Calculate BH critical values
    bh_critical = [(i + 1) / n * alpha for i in range(n)]
    
    # Find largest k where p(k) <= k/n * alpha
    significant_sorted = [False] * n
    for k in range(n - 1, -1, -1):
        if sorted_pvals[k] <= bh_critical[k]:
            # All tests up to this one are significant
            for j in range(k + 1):
                significant_sorted[j] = True
            break
    
    # Map back to original order
    significant = [False] * n
    adjusted_pvals = [0.0] * n
    
    for i, orig_idx in enumerate(sorted_indices):
        significant[orig_idx] = significant_sorted[i]
        # Adjusted p-value (simplified)
        adjusted_pvals[orig_idx] = min(sorted_pvals[i] * n / (i + 1), 1.0)
    
    return adjusted_pvals, significant


# ============================================================================
# CONVENIENCE FUNCTIONS FOR STREAMLIT
# ============================================================================

def evaluate_ensemble_models(models: Dict[str, Any], 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             n_splits: int = 10) -> pd.DataFrame:
    """
    Evaluate multiple models and return comparison DataFrame.
    
    Designed for easy Streamlit display.
    
    Args:
        models: Dictionary of model_name -> model object
        X: Feature DataFrame
        y: Target Series
        n_splits: Number of CV folds
    
    Returns:
        DataFrame with rows per model, columns per metric
    """
    all_results = []
    
    for model_name, model in models.items():
        try:
            cv_results = stratified_kfold_evaluate(model, X, y, n_splits=n_splits)
            
            row = {'Model': model_name}
            for metric_name, result in cv_results.items():
                row[f'{metric_name.upper()}_Mean'] = result.mean
                row[f'{metric_name.upper()}_Std'] = result.std
                row[f'{metric_name.upper()}_CI'] = f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
            
            all_results.append(row)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Model Evaluation Module - Demo")
    print("=" * 50)
    
    # Generate synthetic CV scores for demonstration
    np.random.seed(42)
    
    xgb_scores = np.random.normal(0.85, 0.03, 10)
    lgb_scores = np.random.normal(0.83, 0.04, 10)
    rf_scores = np.random.normal(0.81, 0.035, 10)
    
    print("\n1. Model Comparison (Friedman Test):")
    model_scores = {
        'XGBoost': xgb_scores,
        'LightGBM': lgb_scores,
        'Random Forest': rf_scores
    }
    friedman_result = friedman_compare_models(model_scores)
    print(f"   Statistic: {friedman_result['statistic']:.4f}")
    print(f"   P-value: {friedman_result['p_value']:.4f}")
    print(f"   Ranking: {friedman_result['ranking']}")
    
    print("\n2. Pairwise Comparison (Wilcoxon):")
    wilcox_result = wilcoxon_compare_models(xgb_scores, lgb_scores, "XGBoost", "LightGBM")
    print(f"   {wilcox_result}")
    
    print("\n3. Effect Size (Cohen's d):")
    d = cohens_d(xgb_scores, lgb_scores)
    print(f"   Cohen's d: {d:.4f} ({interpret_cohens_d(d)})")
