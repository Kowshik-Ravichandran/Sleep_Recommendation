"""
Unified Theme Module for AI Sleep Lab
Provides consistent premium styling across all pages.
"""

PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* ========== GLOBAL THEME ========== */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #FF6584;
        --accent-green: #1DD1A1;
        --accent-orange: #FCA5A5;
        --bg-gradient: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-bg-hover: rgba(255, 255, 255, 0.08);
        --glass-border: 1px solid rgba(255, 255, 255, 0.1);
        --text-color: #ffffff;
        --text-muted: rgba(255, 255, 255, 0.7);
        --card-radius: 16px;
        --shadow-glow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Apply to entire app */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-color);
    }
    
    .stApp {
        background: var(--bg-gradient);
        background-attachment: fixed;
    }

    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        color: var(--text-color) !important;
    }
    
    .main-title {
        font-size: 3.2rem;
        background: linear-gradient(90deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 0 4px 20px rgba(108, 99, 255, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 1.2rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }

    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
    }

    /* ========== GLASSMORPHISM CARDS ========== */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: var(--glass-border);
        border-radius: var(--card-radius);
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: var(--shadow-glow);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
        background: var(--glass-bg-hover);
    }

    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ========== CONTAINERS WITH BORDERS ========== */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(12px);
        border: var(--glass-border) !important;
        border-radius: var(--card-radius) !important;
    }

    /* ========== EXPANDERS ========== */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        border: var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-color) !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.2) !important;
        border: var(--glass-border) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* ========== METRICS ========== */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, #4834d4 100%) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(108, 99, 255, 0.3) !important;
        border: none !important;
    }
    
    div[data-testid="metric-container"] * {
        color: white !important;
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--glass-bg);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-muted);
        font-weight: 500;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: white !important;
    }

    /* ========== INPUT STYLING ========== */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTimeInput > div > div > input,
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.07) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input:focus, 
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 10px rgba(108, 99, 255, 0.3) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: var(--primary-color) !important;
    }

    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(90deg, #6C63FF 0%, #4834d4 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        border-radius: 50px;
        font-weight: 600;
        box-shadow: 0 10px 20px rgba(108, 99, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(108, 99, 255, 0.5);
    }

    /* ========== DATAFRAME ========== */
    .stDataFrame {
        background: var(--glass-bg) !important;
        border-radius: var(--card-radius) !important;
        border: var(--glass-border) !important;
    }

    /* ========== INFO/WARNING/SUCCESS BOXES ========== */
    .stAlert {
        background: var(--glass-bg) !important;
        border-radius: 12px !important;
        border-left: 4px solid var(--primary-color) !important;
    }
    
    [data-testid="stAlertContentInfo"] {
        background: rgba(108, 99, 255, 0.1) !important;
    }
    
    [data-testid="stAlertContentWarning"] {
        background: rgba(255, 193, 7, 0.1) !important;
        border-left-color: #ffc107 !important;
    }
    
    [data-testid="stAlertContentSuccess"] {
        background: rgba(29, 209, 161, 0.15) !important;
        border-left-color: var(--accent-green) !important;
    }

    /* ========== DIVIDERS ========== */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* ========== CHARTS (Plotly) ========== */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }
    
    .js-plotly-plot .plotly .modebar-btn path {
        fill: var(--text-muted) !important;
    }

    /* ========== FEATURE CARDS ========== */
    .feature-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border: var(--glass-border);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        border-color: var(--primary-color);
        box-shadow: 0 20px 40px rgba(108, 99, 255, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: var(--text-color);
    }
    
    .feature-desc {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* ========== HERO SECTION ========== */
    .hero-section {
        text-align: center;
        padding: 60px 20px;
        margin-bottom: 40px;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF 0%, #FF6584 50%, #1DD1A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(15deg); }
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: var(--text-muted);
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto 30px auto;
    }

    /* ========== FOOTER ========== */
    .footer {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    /* ========== RESULT CARD ========== */
    .result-container {
        text-align: center;
        padding: 40px;
        border-radius: 20px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    .score-display {
        font-size: 5rem;
        font-weight: 800;
        margin: 10px 0;
        background: linear-gradient(180deg, #ffffff, #a5a5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        margin-top: 10px;
    }
    
    .custom-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #6C63FF;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
"""

def apply_theme():
    """Apply the premium theme CSS to the current page."""
    import streamlit as st
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
