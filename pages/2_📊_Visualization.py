import streamlit as st
import pandas as pd
import plotly.express as px
from modules import data_loader, visualization, insights, statistics
from modules.theme import apply_theme

# Page Config
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Premium Theme
apply_theme()

def main():
    # Page Header
    st.markdown('<div class="main-title">📊 Sleep Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep physiological and behavioral analysis of shift workers</div>', unsafe_allow_html=True)

    # --- Data Loading ---
    with st.sidebar:
        st.markdown('''
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
            <div style="font-weight: 600; color: #6C63FF;">Analytics</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="card-header">📁 Data Configuration</div>', unsafe_allow_html=True)
        df = data_loader.load_data()
        
    if df is None:
        st.warning("⚠️ Please load the dataset to continue.")
        return

    df = data_loader.preprocess_data(df)
    
    # --- Filters ---
    with st.sidebar:
        st.markdown('<div class="card-header" style="margin-top: 20px;">🎛️ Filters</div>', unsafe_allow_html=True)
        if 'Industry' in df.columns:
            industries = ['All'] + list(df['Industry'].unique())
            sel_ind = st.selectbox("Industry", industries)
            if sel_ind != 'All':
                df = df[df['Industry'] == sel_ind]
                
    # --- KPIs ---
    total_participants = len(df)
    avg_sleep = df['Actual_sleep_hours'].mean() if 'Actual_sleep_hours' in df.columns else 0
    pct_poor = (len(df[df['Total_scorePSQI'] > 5]) / total_participants * 100) if 'Total_scorePSQI' in df.columns else 0
    
    k1, k2, k3 = st.columns(3)
    k1.metric("👥 Participants", total_participants)
    k2.metric("😴 Avg Sleep", f"{avg_sleep:.1f} hrs")
    k3.metric("⚠️ Poor Sleep Rate", f"{pct_poor:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Tabs ---
    t1, t2, t3, t4 = st.tabs(["📈 Interactive Charts", "💡 AI Insights", "📉 Statistics", "📋 Data View"])
    
    with t1:
        st.markdown('<div class="card-header">🔬 Deep Dive Analysis</div>', unsafe_allow_html=True)
        
        # 1. Demographics
        with st.expander("📊 A. Sociodemographic Insights", expanded=True):
            d_charts = visualization.plot_demographics(df)
            col1, col2 = st.columns(2)
            if 'gender_sleep' in d_charts: col1.plotly_chart(d_charts['gender_sleep'], use_container_width=True)
            if 'age_dist' in d_charts: col2.plotly_chart(d_charts['age_dist'], use_container_width=True)
            
            col3, col4 = st.columns(2)
            if 'bmi_sleep' in d_charts: col3.plotly_chart(d_charts['bmi_sleep'], use_container_width=True)
            if 'edu_sleep' in d_charts: col4.plotly_chart(d_charts['edu_sleep'], use_container_width=True)
            
        # 2. Shift Work
        with st.expander("🌙 B. Shift-Work & Sleep Patterns", expanded=True):
            s_charts = visualization.plot_shift_work(df)
            if 'shift_sleep_hours' in s_charts: st.plotly_chart(s_charts['shift_sleep_hours'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            if 'bedtime_heatmap' in s_charts: col1.plotly_chart(s_charts['bedtime_heatmap'], use_container_width=True)
            if 'latency_scatter' in s_charts: col2.plotly_chart(s_charts['latency_scatter'], use_container_width=True)
            if 'shift_trend' in s_charts: st.plotly_chart(s_charts['shift_trend'], use_container_width=True)
            
        # 3. Chronotype
        with st.expander("⏰ C. Chronotype Analysis", expanded=False):
            c_charts = visualization.plot_chronotype(df)
            col1, col2 = st.columns(2)
            if 'chrono_pie' in c_charts: col1.plotly_chart(c_charts['chrono_pie'], use_container_width=True)
            if 'meq_hist' in c_charts: col2.plotly_chart(c_charts['meq_hist'], use_container_width=True)
            
            col3, col4 = st.columns(2)
            if 'chrono_sleep_box' in c_charts: col3.plotly_chart(c_charts['chrono_sleep_box'], use_container_width=True)
            if 'msfsc_scatter' in c_charts: col4.plotly_chart(c_charts['msfsc_scatter'], use_container_width=True)
            
        # 4. Nutrition
        with st.expander("🍽️ D. Chrononutrition Behavior", expanded=False):
            n_charts = visualization.plot_nutrition(df)
            col1, col2 = st.columns(2)
            if 'meal_psqi' in n_charts: col1.plotly_chart(n_charts['meal_psqi'], use_container_width=True)
            if 'window_sleep' in n_charts: col2.plotly_chart(n_charts['window_sleep'], use_container_width=True)
            
            col3, col4 = st.columns(2)
            if 'eat_bed_scatter' in n_charts: col3.plotly_chart(n_charts['eat_bed_scatter'], use_container_width=True)
            if 'window_compare' in n_charts: col4.plotly_chart(n_charts['window_compare'], use_container_width=True)

    with t2:
        st.markdown('<div class="card-header">🤖 Automated Analyst</div>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing patterns..."):
            findings = insights.generate_insights(df)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('''
            <div class="glass-card">
                <div class="card-header">🌙 Shift Analysis</div>
            </div>
            ''', unsafe_allow_html=True)
            st.info(f"**Worst Shift:**\n{findings['worst_shift']}")
            st.info(f"**Risk Chronotype:**\n{findings['risk_chronotype']}")
        with c2:
            st.markdown('''
            <div class="glass-card">
                <div class="card-header">🍽️ Nutrition Impact</div>
            </div>
            ''', unsafe_allow_html=True)
            st.warning(f"**Nutrition:**\n{findings['nutrition']}")
            
        st.markdown('<div class="card-header" style="margin-top: 30px;">📈 Key Correlations</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        for c in findings['correlations']:
            st.markdown(f"• {c}")
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="card-header">📉 Statistical Tests</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255,255,255,0.7);">ANOVA & T-Tests for scientific validation</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('''
            <div class="glass-card">
                <div class="card-header">🏢 Shift Differences</div>
            </div>
            ''', unsafe_allow_html=True)
            res = statistics.compare_shifts(df)
            if res:
                for r in res:
                    significance = "✅" if r['p_value'] < 0.05 else "❌"
                    st.write(f"{significance} **{r['variable']}**: p={r['p_value']:.4f} ({r['stars']})")
                    
        with c2:
            st.markdown('''
            <div class="glass-card">
                <div class="card-header">⏰ Chronotype Differences</div>
            </div>
            ''', unsafe_allow_html=True)
            res2 = statistics.compare_chronotypes(df)
            if res2:
                significance = "✅" if res2['p_value'] < 0.05 else "❌"
                st.write(f"{significance} **{res2['variable']}**: p={res2['p_value']:.4f} ({res2['stars']})")

    with t4:
        st.markdown('<div class="card-header">📋 Full Dataset</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: rgba(255,255,255,0.7);">Showing all <strong>{len(df)}</strong> records</p>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=600)

    # Footer
    st.markdown('''
    <div class="footer">
        <p>AI Sleep Lab Analytics • Powered by Streamlit</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
