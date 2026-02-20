"""
Research Analysis Page for AI Sleep Lab

Publication-ready analysis combining:
- SHAP Interpretability
- Cross-Validation with Confidence Intervals
- Circadian Rhythm Feature Analysis
- Statistical Tests with Effect Sizes

Designed for: "How ML Models Help Understand Circadian Rhythm"
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from modules.theme import apply_theme

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Research Analysis | AI Sleep Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_theme()

# Additional styling for research page
st.markdown("""
<style>
    .research-header {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.2) 0%, rgba(29, 209, 161, 0.1) 100%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .significance-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .sig-yes {
        background: rgba(29, 209, 161, 0.2);
        color: #1DD1A1;
    }
    .sig-no {
        background: rgba(255, 107, 107, 0.2);
        color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="research-header">
    <div style="font-size: 2.5rem; margin-bottom: 10px;">🔬</div>
    <h1 style="margin: 0; color: white;">Research Analysis Dashboard</h1>
    <p style="color: rgba(255,255,255,0.7); margin-top: 10px;">
        Publication-Ready Statistical Analysis & Model Interpretability
    </p>
</div>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Load and cache the dataset."""
    data_path = "data/DATA_COMPILATION.xlsx"
    if os.path.exists(data_path):
        return pd.read_excel(data_path)
    return None

df = load_data()

if df is None:
    st.error("❌ Data file not found. Please ensure DATA_COMPILATION.xlsx is in the data/ folder.")
    st.stop()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### 🔬 Research Analysis")
analysis_type = st.sidebar.selectbox(
    "Select Analysis",
    ["📊 Model Performance (CV)", "🧠 SHAP Interpretability", "⏰ Circadian Features", "📈 Statistical Tests"]
)

# ========================================================================
# SECTION 1: MODEL PERFORMANCE WITH CROSS-VALIDATION
# ========================================================================
if analysis_type == "📊 Model Performance (CV)":
    st.markdown("## 📊 Model Performance with Cross-Validation")
    st.markdown("""
    <div class="stat-card">
        <p style="color: rgba(255,255,255,0.8);">
            📌 <strong>Publication Standard:</strong> 10-fold Stratified Cross-Validation with 95% Confidence Intervals.
            All metrics include effect sizes for proper academic reporting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from modules.evaluation import (
            stratified_kfold_evaluate, 
            results_to_dataframe,
            friedman_compare_models,
            wilcoxon_compare_models
        )
        import joblib
        
        # Load models
        models = {}
        model_files = {
            "XGBoost": "project_models/xgboost_bsq_model.pkl",
            "LightGBM": "project_models/lightgbm_bsq_model.pkl",
            "Random Forest": "project_models/random_forest_bsq_model.pkl"
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                models[name] = joblib.load(path)
        
        if not models:
            st.warning("No models found in project_models/ directory.")
        else:
            st.success(f"✅ Loaded {len(models)} models: {', '.join(models.keys())}")
            
            # Prepare data (simplified - you may need to adjust preprocessing)
            st.info("💡 This section displays cross-validation methodology. Full CV requires preprocessing pipeline.")
            
            # Display methodology
            st.markdown("### Methodology")
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | **CV Strategy** | Stratified K-Fold |
            | **Folds** | 10 |
            | **Metrics** | Accuracy, Precision, Recall, F1, AUC-ROC |
            | **Confidence Interval** | 95% (t-distribution) |
            | **Effect Size** | Eta-squared for multi-model comparison |
            """)
            
            # Sample metrics display (replace with actual CV when data is preprocessed)
            st.markdown("### Example Metrics Format")
            
            sample_data = {
                'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'MLP'],
                'AUC-ROC': ['0.87 ± 0.03', '0.85 ± 0.04', '0.83 ± 0.03', '0.82 ± 0.05'],
                '95% CI': ['[0.82, 0.92]', '[0.79, 0.91]', '[0.78, 0.88]', '[0.75, 0.89]'],
                'Accuracy': ['0.84 ± 0.02', '0.82 ± 0.03', '0.81 ± 0.02', '0.80 ± 0.04']
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
            
    except ImportError as e:
        st.error(f"Module import error: {e}")

# ========================================================================
# SECTION 2: SHAP INTERPRETABILITY
# ========================================================================
elif analysis_type == "🧠 SHAP Interpretability":
    st.markdown("## 🧠 SHAP Feature Importance Analysis")
    st.markdown("""
    <div class="stat-card">
        <p style="color: rgba(255,255,255,0.8);">
            🎯 <strong>Research Focus:</strong> Understanding which circadian rhythm features 
            most influence sleep quality predictions using SHAP (SHapley Additive exPlanations).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from modules.interpretability import (
            generate_shap_explainer,
            compute_shap_values,
            get_top_features,
            get_circadian_features
        )
        import joblib
        import shap
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model for SHAP Analysis",
            ["XGBoost (Recommended)", "LightGBM", "Random Forest"]
        )
        
        model_map = {
            "XGBoost (Recommended)": "project_models/xgboost_bsq_model.pkl",
            "LightGBM": "project_models/lightgbm_bsq_model.pkl",
            "Random Forest": "project_models/random_forest_bsq_model.pkl"
        }
        
        model_path = model_map[model_choice]
        
        if os.path.exists(model_path):
            with st.spinner("Loading model and preparing SHAP analysis..."):
                model = joblib.load(model_path)
                st.success(f"✅ Model loaded: {model_choice}")
            
            # Display circadian features of interest
            st.markdown("### 🕐 Key Circadian Features")
            circadian_features = get_circadian_features()
            cols = st.columns(3)
            for i, feat in enumerate(circadian_features[:9]):
                with cols[i % 3]:
                    st.markdown(f"• `{feat}`")
            
            st.markdown("---")
            
            # SHAP plot generation button
            if st.button("🚀 Generate SHAP Analysis", use_container_width=True):
                st.info("⏳ SHAP analysis requires preprocessed data. Please run the full analysis script.")
                st.markdown("""
                **To generate SHAP plots:**
                ```python
                from modules.interpretability import generate_full_shap_report
                
                # Load your preprocessed data
                X = preprocess_data(df)
                
                # Generate complete report
                report = generate_full_shap_report(model, X, "XGBoost")
                ```
                
                Plots will be saved to: `artifacts/shap_plots/`
                """)
                
            # Check for existing SHAP plots
            shap_dir = "artifacts/shap_plots"
            if os.path.exists(shap_dir):
                plots = [f for f in os.listdir(shap_dir) if f.endswith('.png')]
                if plots:
                    st.markdown("### 📊 Existing SHAP Plots")
                    for plot in plots:
                        st.image(os.path.join(shap_dir, plot), caption=plot)
        else:
            st.warning(f"Model not found at: {model_path}")
            
    except ImportError as e:
        st.error(f"SHAP module not available: {e}")
        st.info("Install with: `pip install shap`")

# ========================================================================
# SECTION 3: CIRCADIAN FEATURE ANALYSIS
# ========================================================================
elif analysis_type == "⏰ Circadian Features":
    st.markdown("## ⏰ Circadian Rhythm Feature Analysis")
    st.markdown("""
    <div class="stat-card">
        <p style="color: rgba(255,255,255,0.8);">
            🌙 <strong>Novel Contribution:</strong> Quantifying circadian rhythm disruption through 
            engineered features: Social Jetlag, Phase Delay Index, and Circadian Misalignment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from modules.circadian_analysis import (
            engineer_circadian_features,
            circadian_summary_statistics,
            calculate_social_jetlag,
            calculate_phase_delay_index,
            calculate_circadian_misalignment
        )
        
        # Engineer features
        with st.spinner("Engineering circadian features..."):
            df_circadian = engineer_circadian_features(df)
        
        st.success("✅ Circadian features engineered successfully!")
        
        # Display new features
        new_features = ['Midsleep', 'Phase_Delay_Index', 'Circadian_Misalignment', 
                        'Chronotype_Shift_Match', 'Sleep_Efficiency_Proxy', 'Late_Eating_Flag']
        available_new = [f for f in new_features if f in df_circadian.columns]
        
        if available_new:
            st.markdown("### 📊 Engineered Features Summary")
            
            # Summary statistics
            summary = circadian_summary_statistics(df_circadian)
            if not summary.empty:
                st.dataframe(summary.round(2), use_container_width=True)
            
            # Visualizations
            st.markdown("### 📈 Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Phase_Delay_Index' in df_circadian.columns:
                    fig = px.histogram(
                        df_circadian, 
                        x='Phase_Delay_Index',
                        title='Phase Delay Index Distribution',
                        labels={'Phase_Delay_Index': 'Hours Delayed from 10 PM'},
                        color_discrete_sequence=['#6C63FF']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Circadian_Misalignment' in df_circadian.columns:
                    fig = px.histogram(
                        df_circadian.dropna(subset=['Circadian_Misalignment']), 
                        x='Circadian_Misalignment',
                        title='Circadian Misalignment Score Distribution',
                        labels={'Circadian_Misalignment': 'Misalignment Score (0-100)'},
                        color_discrete_sequence=['#FF6584']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Chronotype-Shift Match Analysis
            if 'Chronotype_Shift_Match' in df_circadian.columns and 'Total_scorePSQI' in df_circadian.columns:
                st.markdown("### 🎯 Chronotype-Shift Match vs Sleep Quality")
                
                match_analysis = df_circadian.groupby('Chronotype_Shift_Match')['Total_scorePSQI'].agg(['mean', 'std', 'count']).reset_index()
                match_analysis.columns = ['Matched', 'Mean PSQI', 'Std', 'N']
                match_analysis['Matched'] = match_analysis['Matched'].map({1: 'Matched', 0: 'Mismatched'})
                
                fig = px.bar(
                    match_analysis,
                    x='Matched',
                    y='Mean PSQI',
                    error_y='Std',
                    title='PSQI Score by Chronotype-Shift Match (Lower = Better Sleep)',
                    color='Matched',
                    color_discrete_map={'Matched': '#1DD1A1', 'Mismatched': '#FF6B6B'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning("Unable to engineer circadian features. Check if required columns exist in dataset.")
            st.write("Required columns: Bedtime, Wake_up_time, Chronotype_MEQ, Shift_Rotation")
            
    except ImportError as e:
        st.error(f"Circadian analysis module error: {e}")

# ========================================================================
# SECTION 4: STATISTICAL TESTS WITH EFFECT SIZES
# ========================================================================
elif analysis_type == "📈 Statistical Tests":
    st.markdown("## 📈 Statistical Tests with Effect Sizes")
    st.markdown("""
    <div class="stat-card">
        <p style="color: rgba(255,255,255,0.8);">
            📐 <strong>Publication Standard:</strong> All statistical tests include effect sizes 
            (Cohen's d for t-tests, η² for ANOVA) and significance stars (*p<0.05, **p<0.01, ***p<0.001).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from modules.statistics import perform_ttest, perform_anova, add_significance_stars
        
        # Select test type
        test_type = st.radio("Select Test", ["T-Test (2 Groups)", "ANOVA (3+ Groups)"])
        
        # Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if 'ID' not in c.upper()]
        
        # Get categorical columns for grouping
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols += [c for c in df.columns if df[c].nunique() <= 5 and df[c].dtype in ['int64', 'float64']]
        categorical_cols = list(set(categorical_cols))
        
        col1, col2 = st.columns(2)
        
        with col1:
            group_var = st.selectbox("Grouping Variable", categorical_cols)
        with col2:
            outcome_var = st.selectbox("Outcome Variable", numeric_cols)
        
        if st.button("🔍 Run Analysis", use_container_width=True):
            if test_type == "T-Test (2 Groups)":
                result = perform_ttest(df, group_var, outcome_var)
                
                if result:
                    st.markdown("### Results")
                    
                    # Display results in formatted cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">T-Statistic</div>
                            <div class="metric-value">{result['t_statistic']:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        sig_class = "sig-yes" if result['significant'] else "sig-no"
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">P-Value</div>
                            <div class="metric-value">{result['p_value']:.4f}</div>
                            <span class="significance-badge {sig_class}">{result['stars']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">Cohen's d</div>
                            <div class="metric-value">{result['cohens_d']:.3f}</div>
                            <div style="color: rgba(255,255,255,0.6);">({result['effect_size']} effect)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Group comparison table
                    st.markdown("### Group Statistics")
                    group_stats = pd.DataFrame({
                        'Group': [result['group1'], result['group2']],
                        'N': [result['n1'], result['n2']],
                        'Mean': [result['mean1'], result['mean2']],
                        'SD': [result['std1'], result['std2']]
                    })
                    st.dataframe(group_stats.round(3), use_container_width=True)
                    
                else:
                    st.warning("T-test requires exactly 2 groups. Please check your grouping variable.")
                    
            else:  # ANOVA
                result = perform_anova(df, group_var, outcome_var)
                
                if result:
                    st.markdown("### Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">F-Statistic</div>
                            <div class="metric-value">{result['f_statistic']:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        sig_class = "sig-yes" if result['significant'] else "sig-no"
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">P-Value</div>
                            <div class="metric-value">{result['p_value']:.4f}</div>
                            <span class="significance-badge {sig_class}">{result['stars']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="metric-label">Eta-Squared (η²)</div>
                            <div class="metric-value">{result['eta_squared']:.3f}</div>
                            <div style="color: rgba(255,255,255,0.6);">({result['effect_size']} effect)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Group means table
                    st.markdown("### Group Means")
                    group_means = pd.DataFrame({
                        'Group': list(result['group_means'].keys()),
                        'Mean': list(result['group_means'].values()),
                        'N': list(result['group_ns'].values())
                    })
                    st.dataframe(group_means.round(3), use_container_width=True)
                    
                else:
                    st.warning("ANOVA requires at least 2 groups with sufficient data.")
                    
    except ImportError as e:
        st.error(f"Statistics module error: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.5);">
    🔬 Research Analysis Module | AI Sleep Lab | Publication-Ready Analytics
</div>
""", unsafe_allow_html=True)
