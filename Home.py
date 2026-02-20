import streamlit as st
from modules.theme import apply_theme

st.set_page_config(
    page_title="Sleep & Health AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Premium Theme
apply_theme()

def main():
    # Hero Section with Premium Styling
    st.markdown('''
    <div class="hero-section animate-fade-in">
        <div class="hero-title">🌙 AI Sleep Lab</div>
        <div class="hero-subtitle">
            Advanced Sleep Quality & Health Analysis System<br>
            <span style="font-size: 1rem;">Data-Driven Insights for Shift Workers</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="feature-card animate-fade-in">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">ML Prediction</div>
            <div class="feature-desc">
                Ensemble of XGBoost, LightGBM, Random Forest & Neural Networks for accurate sleep quality prediction.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown('''
        <div class="feature-card animate-fade-in" style="animation-delay: 0.1s;">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visual Analytics</div>
            <div class="feature-desc">
                Interactive dashboards with deep physiological and behavioral analysis of sleep patterns.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col3:
        st.markdown('''
        <div class="feature-card animate-fade-in" style="animation-delay: 0.2s;">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">XAI Insights</div>
            <div class="feature-desc">
                Explainable AI with SHAP analysis, statistical tests (ANOVA, T-Tests) for scientific validation.
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Problem & Solution Section
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('''
        <div class="glass-card">
            <div class="card-header">🏥 The Problem</div>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.8;">
                Shift work often disrupts the circadian rhythm, leading to <strong style="color: #FF6584;">Shift Work Sleep Disorder (SWSD)</strong>, 
                metabolic issues, and decreased performance in healthcare professionals.
            </p>
            
            <div class="card-header" style="margin-top: 25px;">💡 Our Solution</div>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.8;">
                This application uses <strong style="color: #6C63FF;">Advanced Machine Learning</strong> and 
                <strong style="color: #1DD1A1;">Chronobiology</strong> to:
            </p>
            <ul style="color: rgba(255,255,255,0.7); line-height: 2;">
                <li>📈 <strong>Analyze</strong> sleep patterns across different demographics</li>
                <li>🎯 <strong>Predict</strong> the risk of poor sleep quality (PSQI)</li>
                <li>⚡ <strong>Optimize</strong> daily schedules to minimize health risks</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
    with col_right:
        st.markdown('''
        <div class="glass-card" style="height: 100%;">
            <div class="card-header">✨ Key Features</div>
            <div style="color: rgba(255,255,255,0.8);">
                <p style="margin: 15px 0; padding: 12px; background: rgba(108, 99, 255, 0.15); border-radius: 8px;">
                    <strong style="color: #6C63FF;">🤖 ML Models</strong><br>
                    <span style="font-size: 0.9rem;">Random Forest, XGBoost, CatBoost, MLP</span>
                </p>
                <p style="margin: 15px 0; padding: 12px; background: rgba(255, 101, 132, 0.15); border-radius: 8px;">
                    <strong style="color: #FF6584;">🔍 XAI Engine</strong><br>
                    <span style="font-size: 0.9rem;">SHAP Feature Importance</span>
                </p>
                <p style="margin: 15px 0; padding: 12px; background: rgba(29, 209, 161, 0.15); border-radius: 8px;">
                    <strong style="color: #1DD1A1;">📉 Statistics</strong><br>
                    <span style="font-size: 0.9rem;">ANOVA, T-Tests, Correlations</span>
                </p>
                <p style="margin: 15px 0; padding: 12px; background: rgba(252, 165, 165, 0.15); border-radius: 8px;">
                    <strong style="color: #FCA5A5;">⏰ Scheduler</strong><br>
                    <span style="font-size: 0.9rem;">AI Schedule Optimizer</span>
                </p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        st.info("👈 **Select a Module from the Sidebar to begin your analysis.**")

    # Footer
    st.markdown('''
    <div class="footer">
        <p>Project by <strong>Kowshik</strong> • Powered by Streamlit & Python</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown('''
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 2rem; margin-bottom: 10px;">🌙</div>
        <div style="font-weight: 600; color: #6C63FF;">AI Sleep Lab</div>
        <div style="font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 5px;">v2.0</div>
    </div>
    ''', unsafe_allow_html=True)
    st.sidebar.caption("© 2024 Kowshik")

if __name__ == "__main__":
    main()
