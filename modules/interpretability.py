"""
SHAP Interpretability Module for AI Sleep Lab

This module provides SHAP-based model interpretability for understanding
how circadian rhythm features influence sleep quality predictions.

Publication-ready: Generates high-resolution plots suitable for journal submission.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from typing import Dict, Any, Optional, Tuple

# Configure matplotlib for publication-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Output directory for SHAP plots
SHAP_OUTPUT_DIR = "artifacts/shap_plots"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)


def get_circadian_features() -> list:
    """
    Returns list of circadian-related feature names for highlighting.
    These are the key features for the publication focus.
    """
    return [
        'Chronotype_MEQ', 'ScoreMEQ', 'Bedtime', 'Wake_up_time',
        'Actual_sleep_hours', 'Sleep_latency', 
        'Eating_window_workday', 'Eating_window_freeday',
        'Last_eating_workday', 'Last_eating_freeday',
        'Chronotype_Shift_1', 'Chronotype_Shift_2', 'Chronotype_Shift_3',
        'Shift_Rotation_2', 'Shift_Rotation_3'
    ]


def generate_shap_explainer(model, X: pd.DataFrame, model_type: str = "tree"):
    """
    Create SHAP explainer based on model type.
    
    Args:
        model: Trained ML model
        X: Feature DataFrame for background distribution
        model_type: "tree" for XGBoost/LightGBM/RF, "kernel" for MLP
    
    Returns:
        SHAP Explainer object
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "kernel":
        # For neural networks, use a sample of background data
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
    else:
        # Default to Explainer which auto-detects
        explainer = shap.Explainer(model, X)
    
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame) -> shap.Explanation:
    """
    Compute SHAP values for the given data.
    
    Args:
        explainer: SHAP Explainer object
        X: DataFrame with features
    
    Returns:
        SHAP Explanation object
    """
    shap_values = explainer(X)
    return shap_values


def plot_summary(shap_values, X: pd.DataFrame, 
                 title: str = "SHAP Feature Importance Summary",
                 max_display: int = 20,
                 save: bool = True) -> plt.Figure:
    """
    Generate SHAP summary plot (beeswarm).
    
    This plot shows:
    - Feature importance ranking
    - Impact direction (positive/negative)
    - Feature value effects
    
    Args:
        shap_values: Computed SHAP values
        X: Original feature DataFrame
        title: Plot title
        max_display: Maximum features to display
        save: Whether to save the figure
    
    Returns:
        Matplotlib figure
    """
    ensure_output_dir()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(SHAP_OUTPUT_DIR, "shap_summary.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Saved: {filepath}")
    
    return fig


def plot_bar_importance(shap_values, X: pd.DataFrame,
                        title: str = "Mean Absolute SHAP Values",
                        max_display: int = 15,
                        save: bool = True) -> plt.Figure:
    """
    Generate bar plot of mean absolute SHAP values.
    
    Simpler than beeswarm, good for quick reference.
    
    Args:
        shap_values: Computed SHAP values
        X: Original feature DataFrame
        title: Plot title
        max_display: Maximum features to display
        save: Whether to save the figure
    
    Returns:
        Matplotlib figure
    """
    ensure_output_dir()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(SHAP_OUTPUT_DIR, "shap_bar.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Saved: {filepath}")
    
    return fig


def plot_dependence(shap_values, X: pd.DataFrame, 
                    feature: str,
                    interaction_feature: Optional[str] = None,
                    title: Optional[str] = None,
                    save: bool = True) -> plt.Figure:
    """
    Generate SHAP dependence plot for a specific feature.
    
    Shows how a feature's value affects the prediction,
    optionally colored by an interaction feature.
    
    Args:
        shap_values: Computed SHAP values
        X: Original feature DataFrame
        feature: Feature name to analyze
        interaction_feature: Optional feature for color coding
        title: Plot title (auto-generated if None)
        save: Whether to save the figure
    
    Returns:
        Matplotlib figure
    """
    ensure_output_dir()
    
    if feature not in X.columns:
        print(f"Feature '{feature}' not found in data")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if interaction_feature and interaction_feature in X.columns:
        shap.dependence_plot(feature, shap_values.values, X, 
                            interaction_index=interaction_feature, 
                            ax=ax, show=False)
    else:
        shap.dependence_plot(feature, shap_values.values, X, 
                            ax=ax, show=False)
    
    plot_title = title or f"SHAP Dependence: {feature}"
    plt.title(plot_title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save:
        safe_name = feature.replace("/", "_").replace(" ", "_")
        filepath = os.path.join(SHAP_OUTPUT_DIR, f"shap_dep_{safe_name}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Saved: {filepath}")
    
    return fig


def plot_circadian_dependence_grid(shap_values, X: pd.DataFrame,
                                   save: bool = True) -> plt.Figure:
    """
    Generate grid of dependence plots for key circadian features.
    
    This is specifically designed for the publication to show
    how circadian rhythm features affect sleep quality predictions.
    
    Args:
        shap_values: Computed SHAP values
        X: Original feature DataFrame
        save: Whether to save the figure
    
    Returns:
        Matplotlib figure
    """
    ensure_output_dir()
    
    # Key circadian features to visualize
    circadian_features = [
        'Chronotype_MEQ', 'ScoreMEQ', 'Bedtime', 
        'Actual_sleep_hours', 'Sleep_latency', 'Wake_up_time'
    ]
    
    # Filter to features that exist in the data
    available_features = [f for f in circadian_features if f in X.columns]
    
    if len(available_features) < 2:
        print("Not enough circadian features available for grid plot")
        return None
    
    # Create subplot grid
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.array(axes).flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        shap.dependence_plot(feature, shap_values.values, X, 
                            ax=ax, show=False)
        ax.set_title(f"{feature}", fontsize=10, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(len(available_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Circadian Feature Effects on Sleep Quality Prediction", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(SHAP_OUTPUT_DIR, "shap_circadian_grid.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Saved: {filepath}")
    
    return fig


def plot_force_single(explainer, shap_values, X: pd.DataFrame, 
                      index: int = 0,
                      save: bool = True) -> None:
    """
    Generate force plot for a single prediction (case study).
    
    Args:
        explainer: SHAP Explainer object
        shap_values: Computed SHAP values
        X: Original feature DataFrame
        index: Row index to explain
        save: Whether to save the figure
    """
    ensure_output_dir()
    
    force_plot = shap.force_plot(
        explainer.expected_value if hasattr(explainer, 'expected_value') else shap_values.base_values[index],
        shap_values.values[index],
        X.iloc[index],
        matplotlib=True,
        show=False
    )
    
    if save:
        filepath = os.path.join(SHAP_OUTPUT_DIR, f"shap_force_case_{index}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {filepath}")


def get_top_features(shap_values, X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Get top N most important features based on mean absolute SHAP value.
    
    Args:
        shap_values: Computed SHAP values
        X: Original feature DataFrame
        n: Number of top features to return
    
    Returns:
        DataFrame with feature names and importance scores
    """
    importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(n)


def generate_full_shap_report(model, X: pd.DataFrame, 
                              model_name: str = "XGBoost",
                              model_type: str = "tree") -> Dict[str, Any]:
    """
    Generate complete SHAP analysis report for publication.
    
    This is the main function to call for comprehensive analysis.
    
    Args:
        model: Trained ML model
        X: Feature DataFrame
        model_name: Name of the model for labeling
        model_type: "tree" or "kernel"
    
    Returns:
        Dictionary with SHAP values, figures, and feature importance
    """
    print(f"\n{'='*60}")
    print(f"Generating SHAP Report for {model_name}")
    print(f"{'='*60}\n")
    
    # Create explainer
    print("1. Creating SHAP explainer...")
    explainer = generate_shap_explainer(model, X, model_type)
    
    # Compute SHAP values
    print("2. Computing SHAP values (this may take a moment)...")
    shap_values = compute_shap_values(explainer, X)
    
    # Generate plots
    print("\n3. Generating plots...")
    
    print("   - Summary plot (beeswarm)...")
    fig_summary = plot_summary(shap_values, X, 
                               title=f"{model_name}: Feature Impact on Sleep Quality")
    
    print("   - Bar importance plot...")
    fig_bar = plot_bar_importance(shap_values, X,
                                  title=f"{model_name}: Mean Feature Importance")
    
    print("   - Circadian features grid...")
    fig_circadian = plot_circadian_dependence_grid(shap_values, X)
    
    # Get feature importance ranking
    print("\n4. Computing feature importance ranking...")
    top_features = get_top_features(shap_values, X, n=15)
    
    print("\n✅ SHAP analysis complete!")
    print(f"\nTop 10 Most Important Features:")
    print(top_features.head(10).to_string(index=False))
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_importance': top_features,
        'figures': {
            'summary': fig_summary,
            'bar': fig_bar,
            'circadian_grid': fig_circadian
        }
    }


# Convenience function for Streamlit integration
def get_shap_summary_for_streamlit(model, X: pd.DataFrame, 
                                   model_type: str = "tree") -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Quick SHAP summary for Streamlit display.
    
    Returns figure and importance DataFrame suitable for st.pyplot() and st.dataframe().
    """
    explainer = generate_shap_explainer(model, X, model_type)
    shap_values = compute_shap_values(explainer, X)
    
    fig = plot_summary(shap_values, X, save=False)
    importance_df = get_top_features(shap_values, X, n=10)
    
    return fig, importance_df


if __name__ == "__main__":
    # Test with XGBoost model if available
    import os
    
    model_path = "project_models/xgboost_bsq_model.pkl"
    data_path = "data/DATA_COMPILATION.xlsx"
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        print("Loading model and data for SHAP analysis demonstration...")
        model = joblib.load(model_path)
        
        # Load and preprocess data (simplified - adjust as needed)
        df = pd.read_excel(data_path)
        
        print(f"Data loaded: {len(df)} samples")
        print("\nRun generate_full_shap_report() with processed features for full analysis.")
    else:
        print("Model or data not found. Please ensure paths are correct.")
