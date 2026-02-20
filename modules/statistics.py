import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
import streamlit as st

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two groups.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d, effect_type='cohens_d'):
    """Interpret effect size magnitude."""
    d_abs = abs(d)
    if effect_type == 'cohens_d':
        if d_abs < 0.2: return "negligible"
        elif d_abs < 0.5: return "small"
        elif d_abs < 0.8: return "medium"
        else: return "large"
    elif effect_type == 'eta_squared':
        if d_abs < 0.01: return "negligible"
        elif d_abs < 0.06: return "small"
        elif d_abs < 0.14: return "medium"
        else: return "large"
    return "unknown"


def eta_squared(groups_data):
    """
    Calculate eta-squared (η²) effect size for ANOVA.
    
    Interpretation:
    - η² < 0.01: negligible
    - 0.01 ≤ η² < 0.06: small
    - 0.06 ≤ η² < 0.14: medium
    - η² ≥ 0.14: large
    """
    all_data = np.concatenate([g.values for g in groups_data])
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups_data)
    ss_total = np.sum((all_data - grand_mean)**2)
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def perform_ttest(df, group_col, value_col):
    """
    Performs t-test between two groups.
    Returns test statistic, p-value, effect size (Cohen's d), and interpretation.
    
    Publication-ready: includes effect sizes for research papers.
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        return None
    
    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()
    
    if len(group1) < 2 or len(group2) < 2:
        return None
    
    t_stat, p_value = ttest_ind(group1, group2)
    
    # Calculate effect size
    d = cohens_d(group1, group2)
    effect_interpretation = interpret_effect_size(d, 'cohens_d')
    
    return {
        'group1': groups[0],
        'group2': groups[1],
        'n1': len(group1),
        'n2': len(group2),
        'mean1': group1.mean(),
        'mean2': group2.mean(),
        'std1': group1.std(),
        'std2': group2.std(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': d,
        'effect_size': effect_interpretation,
        'stars': add_significance_stars(p_value)
    }


def perform_anova(df, group_col, value_col):
    """
    Performs one-way ANOVA for multiple groups.
    
    Publication-ready: includes eta-squared effect size.
    """
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    
    group_data = [df[df[group_col] == g][value_col].dropna() for g in groups]
    group_data = [g for g in group_data if len(g) >= 2]
    
    if len(group_data) < 2:
        return None
    
    f_stat, p_value = f_oneway(*group_data)
    
    # Calculate effect size
    eta2 = eta_squared(group_data)
    effect_interpretation = interpret_effect_size(eta2, 'eta_squared')
    
    return {
        'groups': groups.tolist(),
        'group_means': {str(groups[i]): np.mean(g) for i, g in enumerate(group_data)},
        'group_ns': {str(groups[i]): len(g) for i, g in enumerate(group_data)},
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'eta_squared': eta2,
        'effect_size': effect_interpretation,
        'stars': add_significance_stars(p_value)
    }

def correlation_heatmap(df):
    """
    Creates correlation heatmap for numeric variables.
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns
    numeric_cols = [c for c in numeric_cols if 'ID' not in c.upper()]
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap - Numeric Variables',
        xaxis_title='',
        yaxis_title='',
        height=600,
        width=800
    )
    
    return fig

def add_significance_stars(p_value):
    """
    Converts p-value to significance stars.
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def compare_shifts(df):
    """
    Compares sleep metrics across shift types with statistical tests.
    """
    if 'Shift_Rotation' not in df.columns or 'Actual_sleep_hours' not in df.columns:
        return None
    
    results = []
    
    # ANOVA for sleep hours across shifts
    anova_result = perform_anova(df, 'Shift_Rotation', 'Actual_sleep_hours')
    if anova_result:
        results.append({
            'test': 'ANOVA',
            'variable': 'Sleep Hours',
            'groups': 'All Shifts',
            'p_value': anova_result['p_value'],
            'significant': anova_result['significant'],
            'stars': add_significance_stars(anova_result['p_value'])
        })
    
    return results

def compare_chronotypes(df):
    """
    Compares sleep quality across chronotypes.
    """
    if 'Chronotype_Shift' not in df.columns or 'Total_scorePSQI' not in df.columns:
        return None
    
    anova_result = perform_anova(df, 'Chronotype_Shift', 'Total_scorePSQI')
    if anova_result:
        return {
            'test': 'ANOVA',
            'variable': 'PSQI Score',
            'p_value': anova_result['p_value'],
            'significant': anova_result['significant'],
            'stars': add_significance_stars(anova_result['p_value'])
        }
    
    return None
